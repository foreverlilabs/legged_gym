# RC Dog robot assets

This directory contains two URDF variants:

- `urdf/rc_dog_raw.urdf`: original SolidWorks/Gazebo/ROS2 model, retained for reference.
- `urdf/rc_dog.urdf`: cleaned Isaac Gym training model with A1-compatible semantics.

## Training model conventions

- World/body convention: x forward, y left, z up.
- Leg order and names: `FR`, `FL`, `RR`, `RL`.
- Actuated joints per leg: `hip_joint`, `thigh_joint`, `calf_joint`.
- Joint axes use the A1 positive-direction convention: hip `+x`, thigh/calf `+y`.
- The four `*_foot` bodies remain separate when fixed joints are collapsed.
- Visual geometry uses the exported STL files.
- Collision geometry uses a box for the base and thighs, capsules (converted from
  URDF cylinders by Isaac Gym) for calves, and spheres for feet.

The simplified collision geometry is intentional. The A1 and ANYmal assets in
legged_gym use the same approach because primitive contact geometry is faster
and more stable than high-triangle CAD mesh collision in parallel simulation.

## Initial legged_gym asset settings

Use these values when adding `rc_dog_config.py`:

```python
class asset:
    file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/rc_dog/urdf/rc_dog.urdf"
    name = "rc_dog"
    foot_name = "foot"
    penalize_contacts_on = ["thigh", "calf"]
    terminate_after_contacts_on = ["base_link"]
    collapse_fixed_joints = True
    replace_cylinder_with_capsule = True
    flip_visual_attachments = False
    self_collisions = 1
```

The SolidWorks export encodes the nominal standing pose in the joint-origin
rotations. Start with all 12 `default_joint_angles` set to `0.0`; copying A1's
nonzero nominal thigh/calf angles would apply the standing offset twice.

## Provisional physical values

- Total URDF mass: `12.83209292 kg`.
- Base mass: `5.661 kg`, excluding future battery/payload additions.
- Joint limits, effort limits, and velocity limits currently retain the values
  from the supplied model and must be replaced with measured hardware limits
  before sim-to-real deployment.
- Base collision dimensions (`0.36 x 0.24 x 0.14 m`) and limb collision radii
  are initial approximations and should be checked against the model in the
  simulator viewer before long training runs.
