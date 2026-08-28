#!/usr/bin/env python3
"""Inspect the RC Dog URDF in the native Isaac Gym viewer."""

import math
import os

import numpy as np
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR


def print_asset_summary(gym, asset):
    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)

    print("\n=== RC Dog asset summary ===")
    print("Rigid bodies (%d): %s" % (len(body_names), list(body_names)))
    print("DOFs (%d), in policy order:" % len(dof_names))
    for index, name in enumerate(dof_names):
        print(
            "  %2d %-18s lower=% .4f upper=% .4f effort=% .3f velocity=% .3f"
            % (
                index,
                name,
                dof_props["lower"][index],
                dof_props["upper"][index],
                dof_props["effort"][index],
                dof_props["velocity"][index],
            )
        )
    print("Controls: SPACE animate/pause, N next joint, R reset to q=0")
    print("The colored line shows the currently selected joint axis.\n")


def main():
    args = gymutil.parse_arguments(
        description="Inspect the RC Dog training URDF",
        custom_parameters=[
            {
                "name": "--height",
                "type": float,
                "default": 0.32,
                "help": "Initial base height in metres",
            },
            {
                "name": "--amplitude",
                "type": float,
                "default": 0.35,
                "help": "Joint inspection amplitude in radians",
            },
            {
                "name": "--free_base",
                "action": "store_true",
                "help": "Enable gravity and allow the robot base to fall",
            },
        ],
    )

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    viewer = None
    try:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane.static_friction = 1.0
        plane.dynamic_friction = 1.0
        gym.add_ground(sim, plane)

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")

        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "toggle_animation")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_N, "next_joint")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset_pose")

        urdf_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "resources",
            "robots",
            "rc_dog",
            "urdf",
            "rc_dog.urdf",
        )
        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = not args.free_base
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        # SolidWorks STL files are already z-up; flipping them scatters visuals.
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)

        print("Loading '%s' from '%s'" % (asset_file, asset_root))
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
        if asset is None:
            raise RuntimeError("Isaac Gym failed to load %s" % urdf_path)

        num_dofs = gym.get_asset_dof_count(asset)
        if num_dofs != 12:
            raise RuntimeError("Expected 12 DOFs, Isaac Gym loaded %d" % num_dofs)
        print_asset_summary(gym, asset)

        env = gym.create_env(
            sim,
            gymapi.Vec3(-1.5, -1.5, 0.0),
            gymapi.Vec3(1.5, 1.5, 1.5),
            1,
        )
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, args.height)
        actor = gym.create_actor(env, asset, pose, "rc_dog", 0, 1, 0)

        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
        positions = dof_states["pos"]
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

        cam_pos = gymapi.Vec3(1.25, 1.0, 0.75)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.22)
        gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

        dof_names = gym.get_asset_dof_names(asset)
        dof_props = gym.get_asset_dof_properties(asset)
        current_dof = 0
        phase = 0.0
        animate = False
        print("Selected DOF 0: %s" % dof_names[current_dof])

        while not gym.query_viewer_has_closed(viewer):
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                if event.action == "toggle_animation":
                    animate = not animate
                    phase = 0.0
                    positions[:] = 0.0
                    print("Animation:", "ON" if animate else "OFF")
                elif event.action == "next_joint":
                    current_dof = (current_dof + 1) % num_dofs
                    phase = 0.0
                    positions[:] = 0.0
                    print("Selected DOF %d: %s" % (current_dof, dof_names[current_dof]))
                elif event.action == "reset_pose":
                    animate = False
                    phase = 0.0
                    positions[:] = 0.0
                    print("Reset all joints to q=0")

            if animate:
                phase += sim_params.dt * 1.5
                lower = float(dof_props["lower"][current_dof])
                upper = float(dof_props["upper"][current_dof])
                amplitude = min(args.amplitude, max(0.0, upper), max(0.0, -lower))
                positions[current_dof] = amplitude * math.sin(phase)

            gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)

            gym.clear_lines(viewer)
            dof_handle = gym.get_actor_dof_handle(env, actor, current_dof)
            frame = gym.get_dof_frame(env, dof_handle)
            axis_start = frame.origin
            axis_end = frame.origin + frame.axis * 0.12
            gymutil.draw_line(
                axis_start,
                axis_end,
                gymapi.Vec3(1.0, 0.1, 0.1),
                gym,
                viewer,
                env,
            )

            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
