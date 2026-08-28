# 将自定义四足机器人迁移到 legged_gym：训练与 Isaac Gym -> MuJoCo sim-to-sim 指南

本文针对本仓库当前代码（经典 Isaac Gym Preview 3 + `rsl_rl` PPO 结构），目标是：

1. 把自己建模的四足机器人接入 Isaac Gym；
2. 在平地上学会站立、起步、速度跟踪和稳定行走；
3. 再把同一个策略放进 MuJoCo 做 sim-to-sim 对比。

暂不讨论真实硬件、通信、实时线程、关节编码器和安全限位等 sim-to-real 问题。文中的“机器人”均指你的 URDF/MJCF、关节执行器参数和脚本，不是仓库自带的 ANYmal/A1/Cassie。

> **先读这个结论**：迁移成功的关键不是“复制一个配置文件”，而是让以下五个契约在两个仿真器中完全一致：
> `DOF 顺序`、`动作含义`、`观测顺序/尺度`、`控制周期`、`终止与奖励定义`。任何一个契约不一致，PPO 仍可能有 reward 曲线，但策略无法可靠行走，或 Isaac Gym 能走而 MuJoCo 立即倒地。

## 1. 本工程的执行链和改动边界

### 1.1 代码如何创建任务

当前工程的调用链是：

```text
scripts/train.py
  -> get_args()                         # --task, --num_envs, --headless ...
  -> task_registry.make_env()
       -> envs/__init__.py 中的任务注册
       -> LeggedRobot(cfg, sim_params, ...)
  -> task_registry.make_alg_runner()
       -> rsl_rl.runners.OnPolicyRunner
  -> ppo_runner.learn(...)
```

任务注册在 [legged_gym/envs/__init__.py](legged_gym/envs/__init__.py)；环境创建和 PPO runner 在 [legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py)。因此新机器人至少需要：

```text
resources/robots/<my_robot>/urdf/<my_robot>.urdf
legged_gym/envs/<my_robot>/<my_robot>_config.py
legged_gym/envs/__init__.py 中的一行 register(...)
```

如果通用 `LeggedRobot` 的状态、奖励和控制器足够，不需要新建环境 `.py`；只有在需要自定义观测、奖励、终止、特殊执行器网络或特殊 reset 时才继承它。

### 1.2 `LeggedRobot` 已经提供的功能

查看 [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) 可见基类已经实现：

- URDF/MJCF asset 加载、多个并行环境、平地/heightfield/trimesh；
- root state、DOF state、接触力 tensor 的刷新；
- P/V/T 三种基础控制模式；
- 根据配置字符串自动寻找脚端、碰撞和终止 body；
- 速度命令采样、heading 命令、terrain/command curriculum；
- 观测、奖励、终止、reset 和 episode logging；
- 摩擦、质量、推力和观测噪声随机化；
- `play.py` 中的 checkpoint 加载和 TorchScript 导出。

基类的默认观测拼接顺序（`compute_observations()`）是：

```text
[ base_lin_vel*scale (3),
  base_ang_vel*scale (3),
  projected_gravity (3),
  commands[:3]*commands_scale (3),
  (dof_pos-default_dof_pos)*scale (N),
  dof_vel*scale (N),
  last/current actions (N),
  optional height measurements (M) ]
```

这里的动作项实际是 `self.actions`（当前 action），不是 `last_actions`；仓库中 `_get_noise_scale_vec()` 对默认 12 DOF/179 个高度点写了固定切片，因此一旦改变 DOF 数量或高度点数量，必须同步修改它，不能只修改 `env.num_observations`。

### 1.3 重要时间和控制公式

在 [legged_robot.py] 的 `_parse_cfg()` 中：

```python
sim_dt    = cfg.sim.dt
policy_dt = cfg.control.decimation * sim_dt
```

默认值是 `sim.dt=0.005 s`、`decimation=4`，即策略频率 50 Hz。P 控制器在 `_compute_torques()` 中等价于：

```text
q_target = default_q + action_scale * action
tau      = Kp * (q_target - q) - Kd * qdot
tau      = clip(tau, -torque_limit, torque_limit)
```

V 控制和 T（直接力矩）模式也在同一函数中实现。Isaac Gym 每个 policy step 内执行 `decimation` 次仿真步，所以 MuJoCo 必须使用相同的 `sim_dt`、控制更新频率、动作缩放和力矩裁剪。

## 2. 迁移前的机器人规格表（先填完再写代码）

建议在 `resources/robots/<my_robot>/README.md` 或项目 issue 中维护一张“单一事实表”。至少记录：

| 类别 | 必须确定的内容 | 常见错误 |
|---|---|---|
| 坐标系 | Isaac Gym/MuJoCo 都使用 z-up；base 前方是 +x、左侧是 +y；四元数顺序 | 把 wxyz 当成 xyzw，或 y-up/z-up 混用 |
| root | base link 名称、初始高度、初始姿态 | 初始脚离地太高/穿地，第一步就产生爆炸接触 |
| DOF | 可驱动关节名称、数量、URDF 顺序、正方向、单位 rad | 配置字典顺序与 asset 顺序不同；左右腿符号相反 |
| 每个 DOF | lower/upper、velocity、effort、惯量、阻尼、摩擦、armature | URDF 没有有效 effort/velocity，导致限幅为 0 或异常大 |
| body | base、四个脚 body、需要惩罚碰撞的 thigh/calf/shank | `foot_name` 匹配不到 body，脚索引为空 |
| 执行器 | 实际是位置 PD、速度 PD、直接力矩还是 actuator network | 把电机额定力矩当成关节 effort，或重复施加 damping |
| 脚接触 | 脚的几何体是否真正参与碰撞；脚 body 是否是末端 body | 视觉 mesh 名称存在但 collision geometry 缺失 |
| 训练目标 | 速度范围、是否允许侧向运动/原地转向、目标 base 高度 | 直接套用 ANYmal 的高度和奖励尺度 |
| 计量 | 质量 kg、长度 m、角度 rad、力矩 N·m | mesh 使用 mm，惯量却按 m 写入 |

### 2.1 asset 导入前的独立检查

在 Isaac Gym 加载前，先用 URDF/MJCF 工具检查：

```bash
# ROS 环境可用 check_urdf；没有 ROS 时至少用你自己的 XML 解析脚本检查名称和 limit
check_urdf resources/robots/my_robot/urdf/my_robot.urdf
```

应人工确认：

1. 每个驱动 joint 都是预期的 revolute/prismatic 类型，axis 和 limit 正确；
2. link 的 collision geometry 存在且 mesh 路径可解析；
3. base 是浮动根而非 `fix_base_link=True`；
4. mass、inertia、collision 尺寸和摩擦处于合理物理量级；
5. URDF 的 joint 顺序与 Isaac Gym 打印出的 `gym.get_asset_dof_names()` 一致；
6. 脚 body 的名称是 link 名，不要只看 mesh 文件名。

## 3. 推荐的迁移目录和最小实现

### 3.1 目录结构

```text
resources/robots/my_robot/
  urdf/my_robot.urdf
  meshes/...
  README.md                    # 质量、DOF、坐标系、版本和校验记录
legged_gym/envs/my_robot/
  __init__.py
  my_robot_config.py
  my_robot.py                  # 第一步可不创建
```

### 3.2 从基类继承配置

不要修改 `LeggedRobotCfg` 或其他机器人配置来“临时适配”。新建独立配置，避免影响已有 task。下面是一个可直接改名的骨架：

```python
# legged_gym/envs/my_robot/my_robot_config.py
from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg, LeggedRobotCfgPPO,
)


class MyRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096             # 首次调试先用 1/4/16/64
        num_observations = 48 + 0   # 平地、无高度观测时：12+3*N
        num_privileged_obs = None
        num_actions = 12             # 必须等于可驱动 DOF 数
        episode_length_s = 20.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"         # 先平地，再 heightfield/trimesh
        measure_heights = False
        curriculum = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]       # 按你的 base 高度修改
        rot = [0.0, 0.0, 0.0, 1.0]
        default_joint_angles = {
            # 键必须覆盖 gym.get_asset_dof_names() 返回的每个名称
            "FL_hip_joint": 0.1,
            # ... 其余 DOF ...
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"hip": 20.0, "thigh": 25.0, "calf": 30.0}
        damping = {"hip": 0.5, "thigh": 0.7, "calf": 0.8}
        action_scale = 0.25
        decimation = 4
        # 仅当你实现了 actuator network 的机器人类时才打开
        use_actuator_network = False

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/urdf/my_robot.urdf"
        name = "my_robot"
        foot_name = "foot"             # 子串匹配；必须能找到四个 body
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1             # 1=禁用自碰撞，0=允许；按模型调试
        collapse_fixed_joints = False   # 第一次导入建议 False，确认后再优化
        flip_visual_attachments = False # 取决于你的 mesh 坐标系

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 4.0
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1.0, 1.0]

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False  # 先保留失败惩罚，便于诊断
        base_height_target = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -2.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -2.0e-4
            dof_acc = -2.5e-7
            action_rate = -0.01
            feet_air_time = 1.0
            collision = -1.0
            dof_pos_limits = -1.0


class MyRobotCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "my_robot_flat"
        run_name = "baseline"
        max_iterations = 1500
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
```

**观测维度公式**：无高度观测时 `3+3+3+3+N+N+N = 12+3N`；N=12 是 48。打开高度观测后，再加 `len(measured_points_x) * len(measured_points_y)`。例如默认 17×11=187，观测为 235。不要凭记忆写 235；用实际列表计算。

### 3.3 注册任务

在 [legged_gym/envs/__init__.py](legged_gym/envs/__init__.py) 增加 import 和 register。第一版使用基类：

```python
from .my_robot.my_robot_config import MyRobotCfg, MyRobotCfgPPO

task_registry.register(
    "my_robot_flat", LeggedRobot, MyRobotCfg(), MyRobotCfgPPO()
)
```

如果有自定义环境类：

```python
from .my_robot import MyRobot
task_registry.register("my_robot_flat", MyRobot, MyRobotCfg(), MyRobotCfgPPO())
```

不要遗漏 `__init__.py` 的导入。`train.py` 通过 `from legged_gym.envs import *` 触发注册；任务字符串不存在时会在 `TaskRegistry.make_env()` 抛出 `Task with name ... was not registered`。

## 4. DOF、body、脚端和执行器契约

### 4.1 先打印 asset 实际顺序

基类会通过 Isaac Gym 获取：

```python
self.dof_names = self.gym.get_asset_dof_names(robot_asset)
body_names = self.gym.get_asset_rigid_body_names(robot_asset)
```

这些顺序是所有 tensor 的真实索引。强烈建议在 `_create_envs()` 完成后临时加入：

```python
print("num_dof:", self.num_dof)
print("dof_names:", list(enumerate(self.dof_names)))
print("body_names:", list(enumerate(body_names)))
print("feet:", feet_names)
print("penalized:", penalized_contact_names)
print("termination:", termination_contact_names)
```

然后把输出保存到 `resources/robots/my_robot/README.md`。配置中的 `default_joint_angles` 必须按名称覆盖所有 DOF；缺键会在 `_init_buffers()` 直接 `KeyError`。

### 4.2 PD 增益字典是“子串匹配”

基类循环每个 DOF，并检查 `if dof_name in name`。所以：

- `{"hip": 20}` 可以匹配 `FL_hip_joint`；
- 关键字重叠时，后匹配的项可能覆盖先前增益；
- 没匹配到时，P/V 控制会把增益设为 0 并打印警告；
- 最稳妥的做法是使用不重叠的类别名，启动时检查所有 `p_gains > 0`。

若机器人各关节增益完全不同，可覆写 `_init_buffers()` 或先把字典写成每个完整 joint 名，再确认子串行为不会产生歧义。

### 4.3 action 到关节的映射

PPO actor 输出形状是 `[num_envs, num_actions]`，通常范围不是硬限制在 [-1,1]，环境会先按 `clip_actions` 裁剪。之后才应用 `action_scale` 和 default pose。必须明确记录：

```text
policy index 0 -> Isaac DOF name X -> MuJoCo qpos/dof index Y
policy index 1 -> Isaac DOF name Z -> MuJoCo qpos/dof index W
...
```

MuJoCo 的 joint、`qpos`、`qvel` 和 actuator 顺序不一定等于 URDF/Isaac 顺序，不能直接按数组下标复制。

### 4.4 什么时候写 `my_robot.py`

仅配置就够用的情况：标准浮动 base、标准 revolute 腿、P/V/T 控制、通用奖励和接触判定。

需要新类的情况：

- 关节是耦合的，action 需要重排/符号翻转；
- 执行器需要历史状态、LSTM 或 SEA；
- 观测不是基类顺序（例如加入相位、历史帧、IMU 特征）；
- 脚端接触必须改为 force sensor；
- 需要特殊终止条件（翻倒角度、膝盖接触、机身高度）；
- 默认 reward 与机器人结构不适配。

示例：

```python
from legged_gym.envs.base.legged_robot import LeggedRobot

class MyRobot(LeggedRobot):
    def check_termination(self):
        super().check_termination()
        tilt = self.projected_gravity[:, 2] < 0.2
        self.reset_buf |= tilt

    def _reward_my_contact(self):
        # rewards.scales 中设 my_contact != 0 后，基类会自动查找此函数
        return ...
```

奖励尺度非零时，`_prepare_reward_function()` 会通过名字查找 `_reward_<name>`。配置里写了不存在的方法会在环境初始化时失败；零尺度可以移除该项。

## 5. 分阶段实施路线（不要一开始就上 rough terrain）

### 阶段 A：asset 和静态姿态

目标是“模型正确”，不是“机器人会走”。

1. 将 URDF、mesh、纹理放到 `resources/robots/my_robot/`；路径使用 `{LEGGED_GYM_ROOT_DIR}`。
2. `mesh_type="plane"`、`num_envs=1`、关闭噪声、推力、摩擦随机化和 curriculum。
3. `control_type="P"`，先使用保守的 `action_scale`、Kp/Kd；`collapse_fixed_joints=False`。
4. `init_state.pos[2]` 设为 default pose 下脚刚好接近地面且不穿透；不要照抄 A1 的 0.42 或 ANYmal 的 0.6。
5. 用 `play.py` 或小测试脚本观察静止 default pose；确认没有持续翻倒、穿模或关节高速发散。

建议先把 reset 的随机速度改成 0，避免把初始姿态问题误判成控制问题：

```python
# 仅调试期间，在自定义类中覆写
def _reset_root_states(self, env_ids):
    super()._reset_root_states(env_ids)
    self.root_states[env_ids, 7:13] = 0.0
    self.gym.set_actor_root_state_tensor_indexed(...)
```

也可以直接在本地调试分支暂时修改基类，但不要把调试改动混进最终配置。

### 阶段 B：单环境 action/PD 回路

用固定 action 序列验证每个关节：

1. action 全 0：应保持 default pose；
2. 只给 index i 一个正 action：只应有对应关节移动，方向必须符合预期；
3. 观察 `dof_pos`、`dof_vel`、`torques`、关节 limit；
4. 确认扭矩不会超过 URDF effort limit；
5. 用 `sim_dt * decimation` 验证动作每 20 ms 更新一次（若仍采用默认值）。

不要在这一步打开 actuator network；先证明基础 PD 的符号、限幅、单位都对。

### 阶段 C：只训练站立/短距离行走

推荐初始配置：

```text
num_envs             = 64 或 256
mesh_type            = plane
measure_heights      = False
noise.add_noise      = False
randomize_friction   = False
randomize_base_mass  = False
push_robots          = False
commands.lin_vel_x   = [0.0, 0.5]
commands.lin_vel_y   = [0.0, 0.0]
commands.ang_vel_yaw = [0.0, 0.0]
```

先让 reward 中 `tracking_lin_vel`、`orientation`、`lin_vel_z`、`ang_vel_xy`、`torques` 工作，再加入 `feet_air_time`、碰撞和关节限位惩罚。一次只改一组变量，并保存曲线和 checkpoint。

### 阶段 D：扩大命令和训练规模

按以下顺序增加难度：

1. 正向速度 `[0, 0.5]`；
2. 正负正向速度 `[-1, 1]`；
3. 侧向速度；
4. yaw 角速度；
5. 摩擦随机化；
6. 质量、推力和观测噪声；
7. heightfield/trimesh 与 terrain curriculum。

训练已经稳定后再开启 `num_envs=4096`。GPU 显存不足时优先降低 `num_envs`，并同步考虑 PPO rollout 大小：`num_envs * num_steps_per_env` 是一次更新的数据量。

### 阶段 E：rough terrain

只有平地能稳定跟踪多个速度、并能连续运行数个 episode 后，才把 `mesh_type` 改成 `heightfield` 或 `trimesh`。确认：

- `terrain.measure_heights=True` 时观测维度和 noise slice 正确；
- `terrain.num_rows/num_cols` 与 `num_envs` 不造成空/越界 terrain type；
- `projected_gravity`、高度采样坐标和 base frame 方向符合预期；
- 三角网格 GPU contact force 的已知不可靠性不会影响脚接触奖励/终止。

README 已说明 triangle mesh 上 `net_contact_force_tensor` 可能不可靠；若看到脚接触奖励抖动或终止误判，应按 README 的 force sensor 方案为脚端添加传感器，并统一传感器坐标系/是否包含重力。

## 6. 训练命令、日志和验收指标

### 6.1 最小命令

```bash
cd /home/liyong/legged_gym
python legged_gym/scripts/train.py --task=my_robot_flat --headless \
  --num_envs 64 --rl_device=cuda:0 --sim_device=cuda:0 \
  --max_iterations 20
```

先用 20 次迭代验证没有初始化错误，再增加到 1500 或配置值。可用 `--seed` 固定随机性。

播放最近 checkpoint：

```bash
python legged_gym/scripts/play.py --task=my_robot_flat \
  --load_run <run-directory-name> --checkpoint <iteration>
```

`play.py` 会把环境数量压到最多 50、关闭噪声/摩擦随机化/推力，并调用 [helpers.py](legged_gym/utils/helpers.py) 的 `export_policy_as_jit()`，默认输出：

```text
logs/<experiment_name>/exported/policies/policy_1.pt
```

注意仓库 README 中的 `issacgym_anymal` 是历史目录名/拼写；本仓库实际根目录由 `LEGGED_GYM_ROOT_DIR` 决定，日志在 `logs/<experiment_name>/...`。

### 6.2 每个阶段的验收，而不是只看 Mean reward

建议记录以下指标（可以扩展 `Logger` 或导出 CSV）：

| 指标 | 通过标准 |
|---|---|
| episode length | 大量 episode 达到 time limit，而非频繁跌倒 |
| `tracking_lin_vel` | 多个 command 区间都能跟踪，不能只在 0 m/s 得高分 |
| base height/orientation | 无持续俯仰、侧倾和上下振荡 |
| `dof_pos`, `dof_vel` | 不长期贴近 limit，不出现 NaN/Inf |
| torque | 不持续饱和；饱和时知道是模型能力还是奖励设计问题 |
| foot contact | 四脚接触顺序合理，空中时间不是全 0 或异常大 |
| reset cause | 统计脚/腿/机身碰撞和 time-out |
| seed consistency | 至少 3 个 seed 中多数能学会，避免偶然成功 |

`play.py` 已记录单个关节目标、位置、速度、力矩、命令、base 速度和脚接触力，可先利用这些日志；不要只凭 viewer 画面判断策略好坏。

## 7. 观测、奖励和 reset 的设计原则

### 7.1 观测设计

策略输入必须是可观测且在 Isaac/MuJoCo 中都能得到的量。推荐第一版保持基类观测，避免引入仿真器特有量：

- base 线速度/角速度：base frame；
- projected gravity：base frame；
- command：`lin_vel_x, lin_vel_y, ang_vel_yaw`；
- `(q - default_q)`、`qdot`、动作；
- rough terrain 才加入局部高度采样。

不要把 MuJoCo 才有的 constraint force、Isaac 才有的内部 solver 量放进策略观测，除非你能在另一仿真器中构造同义量。

改变观测时同时修改：

1. `cfg.env.num_observations`；
2. `compute_observations()` 的拼接；
3. `_get_noise_scale_vec()` 的每一段；
4. PPO actor/critic 输入维度（由 runner 从环境自动读取，旧 checkpoint 不再兼容）；
5. MuJoCo wrapper 的同样顺序和 scale。

### 7.2 奖励尺度的时间含义

`_prepare_reward_function()` 会把每个非零 scale 乘以 `self.dt`（policy dt）。所以 reward scale 的数值不是简单“每步分数”；改变 `sim.dt` 或 `decimation` 会改变每个 episode 的累计量。调参时固定控制频率，或按 policy dt 重新标定。

默认基类包含 tracking、姿态、速度、力矩、加速度、action rate、feet air time、碰撞和 limit 等项。新机器人要根据质量、腿长、最大力矩和步态重新标定，不要直接把 ANYmal 的 `base_height_target`、`max_contact_force` 和 gains 原样复制。

### 7.3 终止条件和 timeout

基类 `check_termination()` 主要按 `terminate_after_contacts_on` 的接触力和 episode timeout reset；`env.send_timeouts=True` 会把 `time_outs` 传给 PPO，使算法区分“正常到时”与“跌倒失败”。应保留这个行为，并为你的机器人补充必要条件，例如：

- base 高度低于阈值；
- projected gravity 的 z 分量过低（翻倒）；
- 关键 link 发生碰撞；
- 数值异常（NaN/Inf）。

终止阈值必须在 Isaac Gym 和 MuJoCo 中使用同一物理定义，不能一边使用 base 姿态角、一边使用脚端高度却仍称为同一终止条件。

## 8. 调试故障树

### 导入时报 `KeyError`、索引为空或数量不一致

- 打印 `dof_names`/`body_names`；
- 检查 default pose 是否覆盖每个 DOF；
- 检查 `foot_name` 是 body 名的子串；
- 检查 `num_actions == num_dof`；
- 检查 `termination_contact_indices` 不为空；
- 检查 mesh 路径、关节 limit 和 asset 是否被 fixed-joint collapse 改变。

### 一启动就倒、抖动或爆炸

- 初始 z、高度和 default pose 可能不匹配；
- Kp 太大、Kd 太小、action_scale 太大；
- URDF 质量/惯量数量级错误；
- collision mesh 穿透或脚的 collision 未加载；
- DOF 正方向与 default angle/PD 方向反了；
- effort limit 为 0 或过小；
- `sim.dt`/substeps/PhysX 参数不适合模型。

先单环境、无随机、固定 action 诊断，避免直接调 PPO。

### reward 上升但机器人不走

- `tracking_lin_vel` 权重相对姿态/力矩惩罚太小；
- command 采样大部分接近 0；
- `only_positive_rewards=True` 把失败负奖励截断，早期信号不足；
- feet air-time 奖励鼓励跳跃或拖脚；
- 终止接触 body 配错，机器人实际上一直被 reset；
- base velocity 坐标系或 command 轴定义错。

绘制 command、base velocity、每项 episode reward 和 reset 原因，不要只看总 reward。

### `NaN`、PPO KL 爆炸或训练极慢

- 检查 mesh、惯量、contact、action 和 observation 是否出现 NaN/Inf；
- 降低 Kp/action_scale，缩小 command 范围；
- 初期关闭 terrain/randomization/noise；
- 减小学习率、entropy、`num_envs` 或增大 rollout；
- 检查 CUDA/PhysX 日志和显存；
- 固定 seed 重现后再改超参数。

### MuJoCo 立即倒地

优先怀疑接口不一致，而不是 PPO 本身：

1. DOF/action permutation；
2. hip/thigh/calf 正方向或左右腿符号；
3. quaternion 顺序；
4. `q` 是绝对角还是相对 default 角；
5. MuJoCo actuator 重复施加 PD；
6. policy dt 与 Isaac 不同；
7. torque/velocity/position limit、重力、摩擦和接触参数差异。

## 9. Isaac Gym -> MuJoCo sim-to-sim 的实现方案

### 9.1 仓库现状和目标边界

本仓库没有 MuJoCo wrapper、MJCF 转换器或现成 `sim2sim.py`。现有 `play.py` 只在 Isaac Gym 环境中运行 policy，并导出 TorchScript。因此需要单独新建，例如：

```text
sim2sim/
  my_robot_mujoco.py       # 加载 MJCF/URDF，构造观测和控制
  mujoco_model.xml         # 或由 URDF 转换后的模型
  compare_rollout.py       # Isaac/MuJoCo 同一初始状态和 action 对比
```

推荐使用 MuJoCo Python 官方绑定：

```bash
python -m pip install mujoco
```

版本固定在项目 requirements/README 中，避免不同 MuJoCo 版本接触行为变化。

### 9.2 模型一致性

最好从同一套几何、质量、惯量和 joint limit 生成 MuJoCo MJCF；如果从 URDF 转换，逐项核对：

- free joint/root 的位置、旋转和质量；
- joint axis、range、damping、frictionloss、armature；
- actuator gear、ctrlrange、forcerange；
- gravity、地面平面高度、摩擦模型、solver timestep；
- collision geom 的尺寸、接触 margin、过滤规则；
- 四个 foot geom/body 的名称和接触判断方式。

视觉外观不重要，动力学一致性重要。转换后用“静态重力 + 固定姿态”和“单关节小角度”测试，而不是直接跑策略。

### 9.3 MuJoCo wrapper 必须逐字复现的步骤

每个 policy step：

```python
for _ in range(decimation):
    # qpos/qvel 来自 MuJoCo；计算与 Isaac 相同的 torque
    q = data.qpos[dof_qpos_ids]
    qd = data.qvel[dof_qvel_ids]
    q_target = default_q + action_scale * action
    tau = kp * (q_target - q) - kd * qd
    tau = np.clip(tau, -torque_limits, torque_limits)
    data.ctrl[:] = torque_to_actuator_ctrl(tau)  # 取决于 actuator 定义
    mujoco.mj_step(model, data)
```

然后按 Isaac 的同一顺序构造观测：

```python
base_lin_vel = rotate_world_to_base(base_quat, data.qvel[root_lin_vel_ids])
base_ang_vel = rotate_world_to_base(base_quat, data.qvel[root_ang_vel_ids])
projected_gravity = rotate_world_to_base(base_quat, [0, 0, -1])
obs = np.concatenate([
    base_lin_vel * obs_scales.lin_vel,
    base_ang_vel * obs_scales.ang_vel,
    projected_gravity,
    command[:3] * commands_scale,
    (q - default_q) * obs_scales.dof_pos,
    qd * obs_scales.dof_vel,
    action,
    heights_if_enabled,
])
```

推理时：

```python
policy = torch.jit.load("logs/my_robot_flat/exported/policies/policy_1.pt")
with torch.no_grad():
    action = policy(torch.from_numpy(obs).float().unsqueeze(0)).cpu().numpy()[0]
```

如果使用 LSTM actor，必须维持隐藏状态并在 episode reset 调用导出的 `reset_memory()`；本仓库的 `PolicyExporterLSTM` 已提供该接口。不要把 recurrent policy 当成纯 MLP。

### 9.4 torque actuator 的两种正确建模方式

**方式 A：MuJoCo 中使用 torque actuator。** MuJoCo wrapper 直接计算 Isaac 的 PD torque，再通过 `motor`/`general` actuator 输出。这样最容易保证控制律一致；不要再在 MJCF actuator 中配置一套额外的 position gain。

**方式 B：MuJoCo 中使用 position actuator。** 让 policy 输出 `q_target`，MJCF 内置 servo 负责 PD。但这必须复现 Isaac 的 Kp、Kd、ctrlrange、力矩限幅、更新周期和阻尼，否则不是同一控制器。第一版 sim-to-sim 建议方式 A。

若 Isaac 使用 ANYmal 的 actuator LSTM（见 `envs/anymal_c/anymal.py` 和 `resources/actuator_nets/anydrive_v3_lstm.pt`），MuJoCo 端必须实现同样网络输入、隐藏状态、输出和历史；否则先关闭 actuator network 训练一个纯 PD baseline，再单独做执行器模型迁移。

### 9.5 分层验证，不要只比较最终视频

建立固定 seed、固定初始 state、固定 command/action 的回放，依次比较：

1. **模型静态量**：总质量、base 惯量、每个 DOF range/effort；
2. **零 action**：1 个 policy step 后 q/qdot/tau；
3. **单关节 action**：每个 DOF 的符号、幅度和响应；
4. **短 rollout**：10~100 个 policy steps 的 q、qdot、root pose、base velocity；
5. **接触事件**：每只脚 touchdown/liftoff 时间和法向力；
6. **完整 episode**：相同 command 下的速度、姿态、能耗、是否跌倒。

定义数值误差，例如：

```text
max_abs(q_isaac - q_mujoco)       < 1e-2 rad（短 rollout 起点）
max_abs(qdot_isaac - qdot_mujoco) < 1e-1 rad/s
base position/velocity            以模型尺度和接触差异设置阈值
```

阈值不是物理定律；重点是先定位第一处发散的量。若第一步 torque 就不同，查 action/DOF/PD；若无接触时一致、接触后发散，查 geom、摩擦、solver 和脚索引。

### 9.6 观测与 policy 契约测试

在 Isaac 环境中保存一条 `.npz` fixture：

```text
initial_root_state, initial_dof_pos, initial_dof_vel,
command, action, obs_before_step, obs_after_step,
torques, next_root_state, next_dof_pos, next_dof_vel
```

MuJoCo wrapper 读取同一 fixture，逐字段比较。这样可以把“策略问题”和“仿真器差异”分开。建议把 fixture 和模型版本 hash 写入文件名，不要覆盖旧基准。

## 10. 推荐的工程检查清单

### 接入前

- [ ] URDF/MJCF 的单位、坐标、质量、惯量和 collision 已检查
- [ ] base、四个脚、惩罚和终止 body 名称已记录
- [ ] Isaac Gym 打印出的 DOF 顺序已记录
- [ ] action index -> DOF 名称映射已固定
- [ ] default pose 在平地上不穿透且能静态站立

### Isaac Gym 单环境

- [ ] `num_actions == num_dof`
- [ ] `num_observations == 实际拼接长度`
- [ ] 每个 DOF 都有 default angle 和非零/预期 PD gain
- [ ] 正 action 对应正确关节和方向
- [ ] 力矩始终在 effort limit 内
- [ ] reset 后 root、q、qdot 可重复且无 NaN

### Isaac Gym 训练

- [ ] 平地、无随机化可以连续跑满 episode
- [ ] 至少 3 个 seed 做过短训练
- [ ] command、每项 reward、reset cause 有记录
- [ ] 再逐步加入噪声、摩擦、质量、推力和 rough terrain
- [ ] checkpoint 能用 `play.py` 正确加载

### MuJoCo sim-to-sim

- [ ] MJCF/URDF 的 DOF 映射和符号已显式写成表
- [ ] `sim_dt`、decimation、policy dt 完全一致
- [ ] 观测顺序、尺度、裁剪和 action 处理完全一致
- [ ] PD/torque 只实现一次，且限幅一致
- [ ] TorchScript MLP/LSTM 导出和 reset 行为已验证
- [ ] 固定 fixture 的单步和短 rollout 误差在阈值内
- [ ] 接触后发散时已分别检查脚 geom、摩擦和 solver

## 11. 一个务实的里程碑计划

```text
M0（半天~1天）  打印 asset 名称；静态姿态；单关节 action；无 NaN
M1（1~3天）     平地纯 PD，64~256 env，能稳定站立和低速前进
M2（数天）       多速度/转向命令；3 个 seed；checkpoint/play 验收
M3（数天）       打开摩擦、质量、噪声、推力；重新校准奖励和限位
M4（数天）       rough terrain；处理 height obs 和 GPU 接触力问题
M5（1~2周）      导出 TorchScript；MuJoCo 模型/控制器/观测逐层对齐
M6               固定 fixture、短 rollout 和完整 episode 的 sim-to-sim 报告
```

每个里程碑都保留：配置快照、git commit、seed、checkpoint、环境版本、模型 hash 和已知失败案例。这样当 MuJoCo 结果变差时，可以回到最后一个确定正确的契约，而不是重新盲调 PPO。

## 12. 与本仓库相关的文件索引

| 文件 | 迁移时关注点 |
|---|---|
| `legged_gym/envs/base/legged_robot_config.py` | 所有通用配置默认值、观测/控制/奖励字段 |
| `legged_gym/envs/base/legged_robot.py` | asset 加载、DOF/body 索引、PD、reset、观测、奖励、终止 |
| `legged_gym/envs/__init__.py` | import 和 `task_registry.register()` |
| `legged_gym/utils/task_registry.py` | 根据任务名创建环境和 PPO runner、resume 路径 |
| `legged_gym/scripts/train.py` | 训练入口 |
| `legged_gym/scripts/play.py` | checkpoint 推理、TorchScript 导出、状态日志 |
| `legged_gym/utils/helpers.py` | 参数覆盖、seed、配置转 dict、policy 导出 |
| `legged_gym/envs/a1/a1_config.py` | 简单四足配置参考，尤其 default pose/foot/contact |
| `legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py` | 平地观测维度、命令和 reward 覆盖参考 |
| `legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py` | rough terrain、高度观测和 actuator network 参考 |
| `README.md` | 原项目安装、已知 triangle mesh contact force 问题 |

最终验收标准是：在 Isaac Gym 平地和目标 terrain 上，多个 seed 的策略能稳定完成指定命令；同一 TorchScript policy 在 MuJoCo 中经过显式映射和控制/观测对齐后，短 rollout 和完整行走的行为差异可解释、可量化，而不是只看“画面看起来像能走”。
