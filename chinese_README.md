# 用于足式机器人的 Isaac Gym 环境 #

本仓库提供了基于 NVIDIA Isaac Gym 训练 ANYmal（及其他机器人）在粗糙地形上行走的环境。
其中包含实现 sim-to-real 迁移所需的全部组件：执行器网络、摩擦系数与质量随机化、带噪观测，以及训练过程中的随机扰动推力。

**维护者**：Nikita Rudin  
**所属机构**：苏黎世联邦理工学院机器人系统实验室（Robotic Systems Lab, ETH Zurich）  
**联系方式**：rudinn@ethz.ch

---

### :bell: 公告（2024.01.09） ###

随着 NVIDIA 从 Isaac Gym 转向 Isaac Sim，我们已将本项目的全部环境迁移至 [Isaac Lab](https://github.com/isaac-sim/IsaacLab)。迁移完成后，本仓库仅会获得有限的更新与支持。建议所有用户将其应用迁移至新框架。

有关本项目在 Isaac Lab 中的运动控制相关任务的信息，请参见[此处](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html#locomotion)。

---

### 相关链接 ###

项目网站：https://leggedrobotics.github.io/legged_gym/   
论文：https://arxiv.org/abs/2109.11978

### 安装 ###

1. 使用 Python 3.6、3.7 或 3.8 创建新的 Python 虚拟环境（推荐 3.8）。
2. 安装适配 CUDA 11.3 的 PyTorch 1.10：
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. 安装 Isaac Gym。
   - 从 https://developer.nvidia.com/isaac-gym 下载并安装 Isaac Gym Preview 3（Preview 2 无法使用）。
   - `cd isaacgym/python && pip install -e .`
   - 尝试运行示例：`cd examples && python 1080_balls_of_solitude.py`。
   - 如需排障，请查阅文档 `isaacgym/docs/index.html`。
4. 安装 rsl_rl（PPO 实现）。
   - 克隆 https://github.com/leggedrobotics/rsl_rl。
   - `cd rsl_rl && git checkout v1.0.2 && pip install -e .`
5. 安装 legged_gym。
    - 克隆本仓库。
   - `cd legged_gym && pip install -e .`

### 代码结构 ###

1. 每个环境由环境文件（`legged_robot.py`）和配置文件（`legged_robot_config.py`）定义。配置文件包含两个类：一个用于保存全部环境参数（`LeggedRobotCfg`），另一个用于保存训练参数（`LeggedRobotCfgPPo`）。
2. 环境类和配置类均采用继承机制。
3. `cfg` 中每个非零的奖励尺度（reward scale）都会将一个同名函数加入列表；列表中各项求和后得到总奖励。
4. 必须通过 `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` 注册任务。该操作在 `envs/__init__.py` 中完成，也可在本仓库之外执行。

### 使用方法 ###

1. 训练：  
  ```python
  legged_gym/scripts/train.py --task=anymal_c_flat
  ```
    - 如需在 CPU 上运行，请添加参数：`--sim_device=cpu`、`--rl_device=cpu`（可将仿真运行在 CPU 上，同时将 RL 运行在 GPU 上）。
    - 如需无头模式（不渲染），请添加 `--headless`。
    - **重要提示**：为提高性能，训练启动后请按 `v` 停止渲染。之后可重新启用渲染以检查训练进度。
    - 训练好的策略将保存至 `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`。其中，`<experiment_name>` 和 `<run_name>` 在训练配置中定义。
    - 以下命令行参数会覆盖配置文件中设置的值：
     - `--task TASK`：任务名称。
     - `--resume`：从检查点恢复训练。
     - `--experiment_name EXPERIMENT_NAME`：要运行或加载的实验名称。
     - `--run_name RUN_NAME`：本次运行的名称。
     - `--load_run LOAD_RUN`：当 `resume=True` 时要加载的运行名称。若为 `-1`，则加载最近一次运行。
     - `--checkpoint CHECKPOINT`：已保存模型的检查点编号。若为 `-1`，则加载最新检查点。
     - `--num_envs NUM_ENVS`：要创建的环境数量。
     - `--seed SEED`：随机种子。
     - `--max_iterations MAX_ITERATIONS`：最大训练迭代次数。
2. 回放已训练的策略：  
  ```python
  legged_gym/scripts/play.py --task=anymal_c_flat
  ```
    - 默认加载实验目录中最后一次运行的最后一个模型。
    - 可在训练配置中设置 `load_run` 和 `checkpoint`，以选择其他运行或模型迭代版本。

### 添加新环境 ###

基础环境 `legged_robot` 实现了一个粗糙地形运动控制任务。相应的 cfg 未指定机器人资产（URDF/MJCF），也未定义奖励尺度。

1. 在 `envs/` 下添加新目录及 `'<your_env>_config.py`，其应继承自现有环境 cfg。
2. 若添加新机器人：
    - 将对应资产添加至 `resources/`。
    - 在 `cfg` 中设置资产路径，定义刚体名称、默认关节位置及 PD 增益；指定所需的 `train_cfg` 与环境名称（Python 类）。
    - 在 `train_cfg` 中设置 `experiment_name` 和 `run_name`。
3. （如有需要）在 `<your_env>.py` 中实现环境：继承现有环境，重写所需函数和/或添加奖励函数。
4. 在 `isaacgym_anymal/envs/__init__.py` 中注册环境。
5. 按需修改或调优 `cfg`、`cfg_train` 中的其他参数。若要移除某项奖励，请将其尺度设为零。请勿修改其他环境的参数。

### 故障排除 ###

1. 若出现以下错误：`ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`，请执行：`sudo apt install libpython3.8`。也可能需要执行 `export LD_LIBRARY_PATH=/path/to/libpython/directory` 或 `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`（适用于 Conda 用户；请将 `/path/to/` 替换为对应路径）。

### 已知问题 ###

1. 在 GPU 上结合三角网格地形进行仿真时，`net_contact_force_tensor` 报告的接触力不可靠。一种替代方案是使用力传感器，但力会沿相邻刚体的传感器传播，从而导致非预期行为。对于足式机器人，可仅在足部/末端执行器添加传感器，以获得预期结果。使用力传感器时，请通过 `sensor_options.enable_forward_dynamics_forces` 确保将重力等前向动力学力从报告的力中排除。示例：

```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # 例如重力
        sensor_options.enable_constraint_solver_forces = True # 例如接触力
        sensor_options.use_world_frame = True # 在世界坐标系中报告力（更便于获取竖直分量）
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```
