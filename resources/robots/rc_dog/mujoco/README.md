# RC-Dog MuJoCo sim-to-sim

The `rc_dog.xml` file is generated from the Isaac Gym training asset
`../urdf/rc_dog.urdf`. Regenerate it after changing the URDF with:

```bash
python legged_gym/scripts/convert_rc_dog_urdf_to_mjcf.py
```

The MJCF keeps the training model's 12-joint order (`FR`, `FL`, `RR`, `RL`),
collision primitives, inertial parameters, joint limits, and effort limits.
The MuJoCo ground is intentionally flat; this is a first sim-to-sim check, not
a replacement for the rough-terrain curriculum.
