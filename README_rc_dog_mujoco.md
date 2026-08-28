# RC-Dog deployment in MuJoCo

Install MuJoCo in the environment that contains the exported policy:

```bash
pip install mujoco
```

Export the latest actor from a trained checkpoint using the existing Isaac Gym
play script:

```bash
python legged_gym/scripts/play.py --task rc_dog_rough --headless
```

For the exact checkpoint reported in the Isaac Gym test, export that same
checkpoint before launching MuJoCo:

```bash
python legged_gym/scripts/play.py --task rc_dog_rough --load_run=Aug20_18-03-44_ --checkpoint=8700 --headless
```

The exported policy is normally written to
`logs/rough_rc_dog/exported/policies/policy_1.pt`. Run it in MuJoCo with a
fixed command first:

```bash
python legged_gym/scripts/deploy_rc_dog_mujoco.py \
  --policy logs/rough_rc_dog/exported/policies/policy_1.pt \
  --vx 0.3 --seconds 20 --debug --regenerate-xml
```

`--seconds 20` intentionally closes the viewer after 20 seconds. Use
`--seconds 0` to leave it open until the window is closed. The default camera
follows the robot; pass `--no-track-camera` to disable this. A fallen robot is
reset in the same way as Isaac Gym's base-contact termination.

Then enable terminal keyboard commands:

```bash
python legged_gym/scripts/deploy_rc_dog_mujoco.py --keyboard
```

Keep the terminal focused. `W/S` changes forward speed, `A/D` changes lateral
speed, `Q/E` changes yaw speed, `X` or SPACE stops, and `ESC` exits. Each key
press changes the command by a small increment and the command is clipped to
the same `[-1, 1]` range used by the RC-Dog training configuration.

The deploy loop matches training at the policy boundary:

- MuJoCo timestep: `0.005 s`;
- four physics steps per policy action (`0.02 s` policy period);
- PD gains: `kp=20`, `kd=0.5`;
- action scale: `0.25`;
- observation size: `235`;
- joint order: `FR`, `FL`, `RR`, `RL`, three joints per leg.
- body linear velocity is transformed from MuJoCo's world frame, while its
  free-joint angular velocity (already local/body-frame) is passed through,
  matching Isaac Gym's observation convention;
- calf collision cylinders are converted to capsules, matching
  `replace_cylinder_with_capsule=True` in the training asset.

The 187 terrain observations are filled with flat-ground samples. This keeps
the policy input shape correct for the current rough-terrain policy. To test
rough terrain in MuJoCo, replace the ground plane and implement the same 17 x
11 height ray measurements before comparing behavior.
