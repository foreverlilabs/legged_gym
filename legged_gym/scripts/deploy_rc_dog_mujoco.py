"""Run a trained RC-Dog policy in MuJoCo, with optional keyboard commands.

The observation layout mirrors ``LeggedRobot.compute_observations`` exactly:
base linear velocity, body angular velocity, projected gravity, three commands,
12 joint positions, 12 joint velocities, 12 previous actions, and 187 flat
ground height samples (235 values total).

Keyboard mode reads one key at a time from the terminal running this process:
W/S change forward speed, A/D change lateral speed, Q/E change yaw speed, X
stops, SPACE zeros all commands, and ESC exits. Keep the terminal focused while
the MuJoCo viewer is open.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.convert_rc_dog_urdf_to_mjcf import convert


JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
DEFAULT_ANGLES = np.zeros(12, dtype=np.float32)
KP = np.full(12, 20.0, dtype=np.float32)
KD = np.full(12, 0.5, dtype=np.float32)
TORQUE_LIMITS = np.array([23.7, 23.7, 35.55] * 4, dtype=np.float32)
ACTION_SCALE = 0.25
DECIMATION = 4
SIMULATION_DT = 0.005
POLICY_DT = SIMULATION_DT * DECIMATION
OBS_SIZE = 235
HEIGHT_COUNT = 17 * 11


class TerminalKeyboard:
    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdin.isatty()
        self._old = None
        if self.enabled:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("Keyboard: W/S vx, A/D vy, Q/E yaw, X stop, SPACE zero, ESC quit")

    def poll(self, command: np.ndarray) -> bool:
        if not self.enabled:
            return False
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = os.read(sys.stdin.fileno(), 1)
            if not key:
                break
            key = key.decode("utf-8", errors="ignore").lower()
            if key == "\x1b":
                return True
            step = 0.1
            if key == "w":
                command[0] = np.clip(command[0] + step, -1.0, 1.0)
            elif key == "s":
                command[0] = np.clip(command[0] - step, -1.0, 1.0)
            elif key == "a":
                command[1] = np.clip(command[1] + step, -1.0, 1.0)
            elif key == "d":
                command[1] = np.clip(command[1] - step, -1.0, 1.0)
            elif key == "q":
                command[2] = np.clip(command[2] + 0.1, -1.0, 1.0)
            elif key == "e":
                command[2] = np.clip(command[2] - 0.1, -1.0, 1.0)
            elif key == "x":
                command[:] = 0.0
            elif key == " ":
                command[:] = 0.0
        return False

    def close(self) -> None:
        if self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
            self._old = None


def _body_rotation_from_mujoco_quat(mujoco_quat: np.ndarray) -> np.ndarray:
    import mujoco
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, mujoco_quat)
    return rotation.reshape(3, 3)


def build_observation(model, data, command: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
    rotation_world_body = _body_rotation_from_mujoco_quat(data.qpos[3:7])
    base_lin_vel = rotation_world_body.T @ data.qvel[0:3]
    # MuJoCo stores the free-joint angular velocity in the local/body frame,
    # which is the same frame Isaac Gym uses for base_ang_vel. Its
    # translational free-joint velocity above is world-frame and is rotated.
    base_ang_vel = data.qvel[3:6]
    projected_gravity = rotation_world_body.T @ np.array([0.0, 0.0, -1.0])
    q = data.qpos[7:19]
    dq = data.qvel[6:18]

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0:3] = base_lin_vel * 2.0
    obs[3:6] = base_ang_vel * 0.25
    obs[6:9] = projected_gravity
    obs[9:12] = command * np.array([2.0, 2.0, 0.25], dtype=np.float32)
    obs[12:24] = (q - DEFAULT_ANGLES)
    obs[24:36] = dq * 0.05
    obs[36:48] = previous_action
    # The training environment measures terrain height relative to the root.
    # For the flat MuJoCo validation scene, all terrain samples are zero.
    flat_heights = np.clip(float(data.qpos[2]) - 0.5, -1.0, 1.0) * 5.0
    obs[48:] = flat_heights
    return np.clip(obs, -100.0, 100.0)


def _fallen(model, data) -> bool:
    """Match Isaac Gym's base-contact termination for the flat test scene."""
    if data.qpos[2] < 0.16:
        return True
    rotation_world_body = _body_rotation_from_mujoco_quat(data.qpos[3:7])
    projected_gravity = rotation_world_body.T @ np.array([0.0, 0.0, -1.0])
    if np.linalg.norm(projected_gravity[:2]) > 0.8:
        return True
    base_geom = model.geom("base_link_collision_0").id
    ground_geom = model.geom("ground").id
    for i in range(data.ncon):
        contact = data.contact[i]
        if {int(contact.geom1), int(contact.geom2)} == {base_geom, ground_geom}:
            return True
    return False


def _reset_state(model, data) -> None:
    import mujoco
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[2] = 0.33
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)



def _resolve(path: str) -> Path:
    return Path(path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)).expanduser().resolve()


def run(args: argparse.Namespace) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit("MuJoCo is not installed. Run: pip install mujoco") from exc

    xml_path = _resolve(args.xml)
    urdf_path = _resolve(args.urdf)
    if args.regenerate_xml or not xml_path.exists():
        convert(urdf_path, xml_path)

    policy_path = _resolve(args.policy)
    if not policy_path.exists():
        raise FileNotFoundError(f"policy not found: {policy_path}. Run legged_gym/scripts/play.py first.")
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_DT
    model_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                         for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    if model_joint_names != JOINT_NAMES:
        raise RuntimeError(f"MuJoCo hinge order differs from training order:\n{model_joint_names}\n{JOINT_NAMES}")
    if model.nq != 19 or model.nv != 18:
        raise RuntimeError(f"expected 19 qpos/18 qvel, got nq={model.nq}, nv={model.nv}")

    _reset_state(model, data)

    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)
    target = DEFAULT_ANGLES.copy()
    keyboard = TerminalKeyboard(args.keyboard)
    counter = 0
    policy_steps = 0
    started = time.time()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Isaac Gym follows the robot's reference environment. Keep the
            # passive viewer centered on the robot so a fallen/moving robot is
            # not mistaken for a closed simulation.
            if args.track_camera:
                viewer.cam.distance = max(viewer.cam.distance, 1.5)
            obs = build_observation(model, data, command, action)
            while viewer.is_running() and (args.seconds <= 0 or time.time() - started < args.seconds):
                step_start = time.time()
                if keyboard.poll(command):
                    break
                # Isaac Gym computes one action from the current observation,
                # then holds that target for exactly DECIMATION physics steps.
                if counter % DECIMATION == 0:
                    if obs.shape != (OBS_SIZE,):
                        raise RuntimeError(f"unexpected observation shape: {obs.shape}")
                    with torch.no_grad():
                        action = policy(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().reshape(-1)
                    if action.size != 12:
                        raise RuntimeError(f"policy output has {action.size} values, expected 12")
                    action = np.clip(action.astype(np.float32), -100.0, 100.0)
                    target = DEFAULT_ANGLES + ACTION_SCALE * action
                    policy_steps += 1
                tau = KP * (target - data.qpos[7:19]) - KD * data.qvel[6:18]
                data.ctrl[:] = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
                mujoco.mj_step(model, data)
                counter += 1
                if _fallen(model, data):
                    if args.debug:
                        print(f"[reset] t={time.time() - started:.2f}s z={data.qpos[2]:.3f}")
                    _reset_state(model, data)
                    action = np.zeros(12, dtype=np.float32)
                    target = DEFAULT_ANGLES.copy()
                    obs = build_observation(model, data, command, action)
                if counter % DECIMATION == 0:
                    obs = build_observation(model, data, command, action)
                if args.track_camera:
                    viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                if args.debug and counter % DECIMATION == 0:
                    print(f"[policy {policy_steps}] z={data.qpos[2]:.3f} "
                          f"obs=({obs.min():.3f},{obs.max():.3f}) "
                          f"action=({action.min():.3f},{action.max():.3f})")
                remaining = SIMULATION_DT - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        keyboard.close()


def main() -> None:
    root = Path(LEGGED_GYM_ROOT_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(root / "logs/rough_rc_dog/exported/policies/policy_1.pt"))
    parser.add_argument("--xml", default=str(root / "resources/robots/rc_dog/mujoco/rc_dog.xml"))
    parser.add_argument("--urdf", default=str(root / "resources/robots/rc_dog/urdf/rc_dog.urdf"))
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means run until the viewer closes")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--keyboard", action="store_true", help="enable terminal keyboard command input")
    parser.add_argument("--track-camera", action="store_true", default=True,
                        help="follow the robot with the MuJoCo camera (default: on)")
    parser.add_argument("--no-track-camera", dest="track_camera", action="store_false",
                        help="leave the viewer camera fixed")
    parser.add_argument("--debug", action="store_true", help="print policy/action and reset diagnostics")
    parser.add_argument("--regenerate-xml", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
