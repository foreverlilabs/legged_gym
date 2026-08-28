# ANYmal C 强化学习笔记：Actor-Critic、PPO 与 legged_gym

> 本笔记根据当前工作空间中的 `legged_gym` 和 `rsl_rl` 源码整理，主要对应任务 `anymal_c_flat`。目标是从强化学习基本原理出发，把观测、Actor、Critic、动作、奖励和 PPO 更新串成一条完整链路。

## 1. 当前项目在做什么

当前训练系统由两个主要仓库组成：

- `legged_gym`：负责机器人模型、Isaac Gym 物理仿真、观测、动作、奖励、终止条件和随机化。
- `rsl_rl`：负责 Actor-Critic 网络、PPO 算法、经验缓存和训练循环。

当前主要任务是 ETH Zurich 的 ANYmal C 平地运动任务：

```text
48 维观测
    ↓
Actor 策略网络
    ↓
12 维动作
    ↓
ANYdrive 执行器网络
    ↓
12 个关节力矩
    ↓
Isaac Gym 物理仿真
    ↓
新观测、奖励、终止标志
    ↓
Critic 估值 + PPO 更新 Actor 和 Critic
```

任务注册位置：

```text
legged_gym/legged_gym/envs/__init__.py
```

当前平地配置：

```text
legged_gym/legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py
```

## 2. 强化学习的基本问题

强化学习不是直接告诉机器人“每个关节应该转多少度”，而是定义：

1. 机器人能观察到什么。
2. 机器人可以输出什么动作。
3. 什么行为能获得奖励。
4. 什么情况下一个回合结束。

每个控制周期发生一次交互：

```text
观测 o_t → 策略 π(a|o) → 动作 a_t → 环境 → 奖励 r_t 和新观测 o_(t+1)
```

常见符号：

| 符号 | 含义 |
|---|---|
| `s_t` | 环境的完整状态，通常只有仿真器完全知道 |
| `o_t` | 策略实际得到的观测 |
| `a_t` | 策略输出的动作 |
| `r_t` | 当前一步奖励 |
| `π(a|o)` | Actor 表示的策略 |
| `V(o)` | Critic 对未来累计奖励的估计 |
| `γ` | 折扣因子，当前为 `0.99` |

机器人的目标是最大化折扣累计奖励：

```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
```

这里的重点是“累计”。一个动作当前可能得到不错的速度奖励，但如果它让机器人下一刻跌倒，那么它的长期回报并不好。

### 状态和观测为什么不同

仿真器知道完整且精确的机器人状态，但真实机器人只能通过 IMU、编码器等传感器获得带噪声的信息。因此策略通常使用观测 `o_t`，而不是理想化的完整状态 `s_t`。

这也意味着四足运动通常属于部分可观测问题：仅凭当前一帧信息，未必能完全判断接触状态、外力和地形属性。当前代码把上一次动作放进观测，能给策略提供少量历史信息；更复杂的任务也可以使用 LSTM/GRU。

## 3. Actor：负责选择动作

Actor 是策略网络。当前结构为：

```text
48 → 128 → 64 → 32 → 12
```

隐藏层使用 ELU 激活函数。最后的 12 维对应 ANYmal C 的 12 个可驱动关节。

Actor 首先根据观测输出动作均值：

```math
\mu_t = \operatorname{Actor}(o_t)
```

训练时不会总是直接使用均值，而是从高斯分布采样：

```math
a_t \sim \mathcal{N}(\mu_t, \sigma)
```

源码位置：

```text
rsl_rl/rsl_rl/modules/actor_critic.py
```

其中：

- `mean` 由 Actor 网络输出。
- `std` 是 12 维可训练参数，初始值为 `1.0`。
- 训练时调用 `distribution.sample()`，用随机动作探索不同步态。
- 推理时调用 `act_inference()`，直接使用均值，动作更加稳定。

所以 Actor 不是输出一个固定动作表，而是在学习“不同观测下，什么动作分布比较好”。

## 4. Critic：负责评价当前处境

当前 Critic 结构为：

```text
48 → 128 → 64 → 32 → 1
```

Critic 最终只输出一个标量：

```math
V_\phi(o_t) \approx \mathbb{E}[G_t \mid o_t]
```

它表示：从当前观测开始，继续按照当前策略运动，预计还能得到多少折扣累计奖励。

例如：

- 状态 A 的 Critic 输出 `20`：姿态稳定、速度跟踪良好，未来比较有希望。
- 状态 B 的 Critic 输出 `2`：机器人可能已经严重倾斜，即将跌倒。

Critic 不直接控制机器人，也不会把动作发送给 Actor。它主要充当 Actor 的学习基准。

### 为什么不能只用奖励训练 Actor

同一个动作的结果可能受到摩擦、初始姿态、随机推力等因素影响，直接使用累计奖励更新 Actor，方差会很大。

Actor-Critic 使用优势函数衡量动作是否“比预期更好”：

```math
A_t = G_t - V_\phi(o_t)
```

- `A_t > 0`：实际结果比 Critic 的预期好，增加该动作的概率。
- `A_t < 0`：实际结果比预期差，降低该动作的概率。
- `A_t ≈ 0`：结果基本符合预期，不需要大幅修改策略。

因此 Critic 的核心作用是降低策略梯度的方差，提高训练稳定性。

## 5. Actor 和 Critic 的关系

当前实现中，它们是两个独立 MLP，没有共享隐藏层，但被放在同一个 `ActorCritic` 模块里，并由同一个 Adam 优化器更新。

```text
                   ┌→ Actor → 动作分布
观测 ──────────────┤
                   └→ Critic → 状态价值
```

训练时二者都需要；部署到机器人时通常只保留 Actor：

```text
训练：Actor + Critic + PPO + 奖励 + 随机探索
部署：观测 → Actor 均值 → 动作 → 底层控制器
```

## 6. 对称与不对称 Actor-Critic

配置中的：

```python
num_privileged_obs = None
```

表示当前没有开启特权观测，Actor 和 Critic 使用相同的 48 维观测。

不对称 Actor-Critic 通常是：

- Actor 只看真实机器人部署时能获得的信息。
- Critic 在训练时额外看到真实摩擦系数、地形高度、接触力、外力等仿真真值。
- 部署时删除 Critic，因此特权信息不需要在真机上获取。

更准确的 Critic 往往能给 Actor 提供更好的优势估计。但仅修改 `num_privileged_obs` 并不能开启它，还必须在环境中正确填充 `privileged_obs_buf`。

## 7. 当前 48 维观测的组成

观测在以下函数中构造：

```text
legged_gym/legged_gym/envs/base/legged_robot.py
compute_observations()
```

| 索引范围 | 维度 | 内容 |
|---|---:|---|
| `0:3` | 3 | 机身坐标系线速度 `vx, vy, vz` |
| `3:6` | 3 | 机身坐标系角速度 |
| `6:9` | 3 | 机身坐标系中的投影重力方向 |
| `9:12` | 3 | 目标 `vx, vy, yaw速度` |
| `12:24` | 12 | 相对默认站姿的关节角 |
| `24:36` | 12 | 关节速度 |
| `36:48` | 12 | 上一次应用的策略动作 |

合计：

```text
3 + 3 + 3 + 3 + 12 + 12 + 12 = 48
```

### 为什么使用投影重力

投影重力表示世界重力方向在机身坐标系中的方向。如果机身完全水平，它大致为 `[0, 0, -1]`；机器人向一侧倾斜时，x/y 分量会出现变化。

这比直接给欧拉角更连续，也避免角度在 `-π` 和 `π` 处跳变。

### 为什么需要观测缩放

不同物理量的量级差异很大。例如关节速度可能远大于姿态方向分量。如果不做缩放，大数值特征可能主导神经网络。

当前主要缩放为：

```text
线速度 × 2.0
角速度 × 0.25
关节位置 × 1.0
关节速度 × 0.05
```

### 为什么添加观测噪声

训练时对速度、关节状态和重力方向加入均匀噪声，可以减少策略对完美仿真数据的依赖，有利于迁移到存在传感器误差的真实机器人。

## 8. 12 维动作怎样变成关节力矩

Actor 输出的 12 个数不是直接力矩，而是相对默认站姿的关节目标偏移：

```math
q^{des} = q^{default} + \text{action_scale} \cdot a_t
```

当前：

```text
action_scale = 0.5
```

例如 Actor 对某关节输出 `0.4`，则目标角度相对默认角度偏移：

```text
0.4 × 0.5 = 0.2 rad
```

对于普通位置控制模式，基础实现使用 PD 控制：

```math
\tau = K_p(q^{des}-q)-K_d\dot q
```

但 ANYmal C 当前启用了：

```python
use_actuator_network = True
```

所以真正使用的是预训练 ANYdrive LSTM 执行器模型：

```math
\tau_t = f_{actuator}(q^{des}-q, \dot q, h_{t-1})
```

源码位置：

```text
legged_gym/legged_gym/envs/anymal_c/anymal.py
```

它能够表达理想 PD 控制器没有包含的执行器动态，有助于缩小仿真与真实电机之间的差距。

## 9. 控制频率和物理频率

当前配置：

```text
物理仿真步长 dt_sim = 0.005 s
物理频率 = 1 / 0.005 = 200 Hz
decimation = 4
策略周期 dt_policy = 4 × 0.005 = 0.02 s
策略频率 = 50 Hz
```

也就是说，Actor 每 20 ms 输出一次动作；在两次 Actor 更新之间，Isaac Gym 使用同一个动作连续仿真4个物理步。

回合长度是20秒，因此一个完整回合最多约为：

```text
20 / 0.02 = 1000 个策略步
```

## 10. 命令如何生成

Actor 不只是学习向前走，而是学习跟踪速度命令。当前命令范围为：

```text
目标 x 速度：[-1.0, 1.0] m/s
目标 y 速度：[-1.0, 1.0] m/s
目标 yaw 角速度：[-1.5, 1.5] rad/s
```

平地配置中：

```text
heading_command = False
resampling_time = 4 s
```

因此每个环境大约每4秒重新随机采样目标线速度和角速度，Actor 必须学会根据命令改变运动方向，而不是只记住一种固定步态。

## 11. 奖励函数怎样塑造步态

总奖励由多个子奖励相加：

```math
r_t = \sum_i w_i r_i
```

源码位置：

```text
legged_gym/legged_gym/envs/base/legged_robot.py
compute_reward()
```

当前主要有效奖励如下：

| 奖励项 | 权重 | 作用 |
|---|---:|---|
| `tracking_lin_vel` | `1.0` | 跟踪目标 x/y 速度 |
| `tracking_ang_vel` | `0.5` | 跟踪目标 yaw 角速度 |
| `lin_vel_z` | `-2.0` | 减少机身上下跳动 |
| `ang_vel_xy` | `-0.05` | 减少 roll/pitch 摇晃 |
| `orientation` | `-5.0` | 保持机身水平 |
| `torques` | `-0.000025` | 减少关节力矩 |
| `dof_acc` | `-2.5e-7` | 减少关节剧烈加速 |
| `feet_air_time` | `2.0` | 鼓励足端形成腾空阶段 |
| `collision` | `-1.0` | 惩罚大腿、小腿碰撞 |
| `action_rate` | `-0.01` | 减少相邻动作突变 |

### 速度跟踪奖励

线速度奖励为：

```math
r_v = \exp\left(-\frac{\|v^{cmd}_{xy}-v_{xy}\|^2}{\sigma}\right)
```

当前 `tracking_sigma=0.25`。误差越小，奖励越接近1；误差越大，奖励指数下降。

角速度跟踪使用相似形式：

```math
r_\omega = \exp\left(-\frac{(\omega_z^{cmd}-\omega_z)^2}{\sigma}\right)
```

### 姿态惩罚

姿态惩罚使用投影重力的 x/y 分量：

```math
r_{orientation} = g_x^2 + g_y^2
```

函数返回非负误差，再乘负权重 `-5.0`，所以倾斜越严重，惩罚越大。

### 足端腾空奖励

脚首次重新接触地面时，根据腾空时间计算：

```math
r_{air} = t_{air} - 0.5
```

只有存在明显移动命令时才启用。它能避免四只脚一直拖地，但权重过大也可能诱导夸张跨步、跳跃或不自然步态。

### 力矩和动作平滑惩罚

力矩惩罚：

```math
r_\tau = \sum_j \tau_j^2
```

动作变化惩罚：

```math
r_{rate} = \sum_j(a_{t,j}-a_{t-1,j})^2
```

前者限制能耗和执行器负担，后者减少高频抖动。二者过强会导致机器人不愿运动，过弱则容易产生剧烈动作。

### 奖励权重还会乘策略周期

初始化奖励函数时，所有非零权重都会乘：

```text
dt = 0.02 s
```

因此配置中的权重更接近“每秒奖励权重”，修改控制频率时，单回合累计奖励不会简单地按步数成倍变化。

### `only_positive_rewards` 的影响

当前：

```python
only_positive_rewards = True
```

普通奖励求和后会被裁剪：

```math
r_t \leftarrow \max(r_t,0)
```

如果某一步所有惩罚超过正奖励，最终奖励通常只是0，而不是很大的负数。

此外当前：

```text
termination = -0.0
```

跌倒没有直接负奖励。跌倒的主要代价是回合提前结束，从而失去未来可能获得的奖励。这个设计可以避免早期训练时过大的负奖励让机器人学会完全不动，但阅读训练结果时必须清楚这一点。

## 12. 随机化与 Sim-to-Real

当前训练并不是在完全固定的环境中进行，包含：

- 地面摩擦随机化。
- 机身质量随机增加或减少最多5 kg。
- 每15秒随机推机器人一次。
- 初始关节角随机化。
- 初始机身线速度和角速度随机化。
- 观测噪声。

这些随机化迫使 Actor 学习一个具有鲁棒性的反馈控制策略，而不是只记住某条理想轨迹。

平地摩擦范围配置为 `[0.0, 1.5]`。代码注释指出平面接触使用平均组合，因此最终脚地组合摩擦并不简单等于这个采样值。

## 13. 一轮 PPO 训练发生了什么

训练循环位于：

```text
rsl_rl/rsl_rl/runners/on_policy_runner.py
```

当前参数：

```text
并行环境数 num_envs = 4096
每个环境采集步数 num_steps_per_env = 24
每轮数据量 = 4096 × 24 = 98304 条 transition
学习 epoch 数 = 5
mini-batch 数 = 4
每轮梯度更新次数 = 5 × 4 = 20
```

一轮训练流程：

1. Actor 根据观测产生动作分布并采样动作。
2. Critic 计算当前观测的价值。
3. 环境执行动作，产生奖励和下一个观测。
4. 保存观测、动作、奖励、价值、动作概率和终止标志。
5. 连续采集24步。
6. 使用 Critic 对第24步之后的未来进行 bootstrap。
7. 使用 GAE 计算每一步的 advantage 和 return。
8. 将98304条数据划分为4个 mini-batch。
9. 使用同一批数据训练5个 epoch。
10. 清空 rollout storage，用更新后的策略重新采集数据。

PPO 是 on-policy 算法。旧数据只能短暂重复使用，策略更新后必须重新采样，不能像某些 off-policy 算法那样长期保留大量历史经验。

## 14. GAE：怎样计算优势

源码位置：

```text
rsl_rl/rsl_rl/storage/rollout_storage.py
compute_returns()
```

首先计算一步时序差分误差：

```math
\delta_t = r_t + \gamma V(o_{t+1}) - V(o_t)
```

然后反向递推 GAE：

```math
A_t = \delta_t + \gamma\lambda A_{t+1}
```

当前参数：

```text
gamma = 0.99
lambda = 0.95
```

`lambda` 越接近1，优势估计考虑得越远，偏差较小但方差较大；越接近0，就越依赖 Critic 的一步估值，方差较小但可能带来更大偏差。

最后计算 Critic 的监督目标：

```math
R_t = A_t + V(o_t)
```

代码还会对整批 advantage 做标准化，使其均值约为0、标准差约为1，从而提高优化稳定性。

### 时间结束与真正失败

20秒自然结束和机器人跌倒具有不同含义：

- 时间结束：状态本身不一定失败，应使用 Critic 对后续价值做 bootstrap。
- 跌倒结束：这是实际 terminal，不继续 bootstrap。

`send_timeouts=True` 就是为了把二者区分开。

## 15. PPO 的核心：限制策略变化幅度

PPO 比较新旧策略对已采样动作的概率：

```math
\rho_t(\theta)=\frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{old}}(a_t|o_t)}
```

当前：

```text
clip_param = 0.2
```

因此 PPO 将比例限制在大致 `[0.8, 1.2]` 范围：

```math
L^{clip}=\mathbb{E}\left[
\min\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
```

直觉上，即使某一批动作表现很好，也不能一次性把它们的概率提高很多。四足策略如果更新太猛，原本稳定的步态可能在一轮更新后突然崩溃。

源码位置：

```text
rsl_rl/rsl_rl/algorithms/ppo.py
```

## 16. Critic 损失与总损失

Critic 使用回报目标做均方误差训练：

```math
L_V = (V_\phi(o_t)-R_t)^2
```

当前还启用了 value clipping，防止 Critic 一次变化过大。

Actor-Critic 总损失可概括为：

```math
L = L_{actor} + c_v L_{critic} - c_e H(\pi)
```

其中：

- `L_actor`：PPO 策略代理损失。
- `L_critic`：价值估计误差。
- `H(π)`：动作分布熵，代表探索程度。
- `value_loss_coef=1.0`。
- `entropy_coef=0.01`。

负的熵损失意味着最小化总损失时会鼓励更大的熵，防止策略过早变成近乎确定的动作。

## 17. KL 距离和自适应学习率

当前配置：

```text
schedule = adaptive
desired_kl = 0.01
learning_rate = 1e-3
```

代码计算新旧高斯策略的 KL 距离：

- KL 太大：说明策略变化过快，学习率除以 `1.5`。
- KL 太小：说明更新过于保守，学习率乘以 `1.5`。
- 学习率限制在 `[1e-5, 1e-2]`。

PPO clipping 和 KL 自适应学习率共同限制策略更新速度。

## 18. 为什么不需要对物理仿真求导

Actor 输出动作后，Isaac Gym 中的接触、碰撞和奖励计算并不参与反向传播。PPO 使用的是策略梯度思想：

```math
\nabla_\theta J \approx
\mathbb{E}[\nabla_\theta\log\pi_\theta(a_t|o_t)A_t]
```

只需要知道：

1. 当时的动作在策略下概率是多少。
2. 这个动作最后比预期好还是差。

因此环境可以包含不可导的碰撞、重置和离散接触判断。

## 19. 当前配置的训练规模

当前平地配置：

```text
max_iterations = 300
```

预计总采样量：

```text
300 × 4096 × 24 = 29,491,200 个环境步
```

每个环境每轮只推进：

```text
24 × 0.02 = 0.48 秒
```

但由于有4096个并行环境，每轮可以同时获得大量不同初始状态、命令和随机物理参数下的数据。

当前网络比基础配置的 `512 → 256 → 128` 更小：

```text
当前：128 → 64 → 32
```

优点是训练和推理更快、部署计算量更低；缺点是表达能力较弱。是否需要扩大网络应根据学习曲线和最终行为决定，而不是只看网络大小。

## 20. 训练日志应该怎样看

训练日志位于：

```text
legged_gym/logs/flat_anymal_c/
```

不要只看总奖励，至少同时关注：

| 指标 | 观察重点 |
|---|---|
| `Mean reward` | 是否整体上升并逐渐稳定 |
| `Mean episode length` | 是否从频繁跌倒逐渐接近完整回合 |
| `tracking_lin_vel` | 线速度跟踪能力 |
| `tracking_ang_vel` | 转向能力 |
| `orientation` | 机身稳定性 |
| `torques` | 是否依赖过大力矩 |
| `action_rate` | 动作是否抖动 |
| `feet_air_time` | 是否形成步态，是否出现过度腾空 |
| `Value function loss` | Critic 是否能拟合回报 |
| `Surrogate loss` | Actor 更新情况，绝对值不直接等于性能 |
| `Mean action noise std` | 探索程度是否合理 |
| `Learning rate` | KL 自适应是否频繁缩小学习率 |

奖励上升不一定等于动作自然。策略可能找到奖励函数漏洞，所以必须同时查看仿真视频、速度误差、接触力、关节动作和跌倒情况。

## 21. 当前配置中值得特别注意的地方

### 21.1 `feet_air_time=2.0` 较强

它可能促进清晰步态，但也可能鼓励过大步幅或跳跃。若出现弹跳、腾空时间异常，可以考虑结合以下信息判断，而不是立即只改一个权重：

- `lin_vel_z` 惩罚。
- `orientation` 惩罚。
- 足端接触力惩罚。
- 动作变化惩罚。
- 仿真中的真实步态视频。

### 21.2 足端接触力惩罚当前没有启用

配置中的：

```python
# feet_contact_forces = -0.01
```

仍被注释，因此 `max_contact_force=350` 暂时不会产生对应奖励惩罚。只有非零奖励项才会被注册到奖励函数列表。

### 21.3 `clip_actions=100` 很宽

Actor 的高斯动作理论上无界。虽然策略通常会学到合理范围，但 `100` 并没有真正把动作限制到常见的 `[-1,1]`。分析异常动作时，应查看 Actor 均值、标准差和实际关节目标，而不能假设输出天然在 `[-1,1]`。

### 21.4 Critic 当前没有特权信息

Actor 与 Critic 输入维度相同并不是错误，只是普通对称 Actor-Critic。后续若研究 sim-to-real，可以再学习 asymmetric Actor-Critic，但初学阶段先把当前链路理解透更重要。

### 21.5 代码里的两种 `actor` 不要混淆

Isaac Gym 中也把一个仿真实体称为 `actor`，例如机器人模型实例；强化学习中的 Actor 则是策略神经网络。它们只是英文名称相同，并不是同一个对象。

## 22. 一条 transition 中保存了什么

Rollout storage 会保存：

```text
observation
critic_observation
action
reward
done
old value
old action log probability
old action mean
old action standard deviation
```

保存旧动作概率、均值和标准差，是因为 PPO 更新时必须比较新策略与采样数据时的旧策略。

## 23. 从启动训练到保存模型的代码链路

```text
train.py
  ↓
task_registry.make_env()
  ↓
创建 ANYmal 环境与 4096 个并行机器人
  ↓
task_registry.make_alg_runner()
  ↓
创建 ActorCritic、PPO、RolloutStorage
  ↓
OnPolicyRunner.learn()
  ↓
采集24步 → GAE → PPO更新 → 保存 checkpoint
```

关键文件阅读顺序：

1. `legged_gym/legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py`
2. `legged_gym/legged_gym/envs/base/legged_robot_config.py`
3. `legged_gym/legged_gym/envs/base/legged_robot.py`
4. `legged_gym/legged_gym/envs/anymal_c/anymal.py`
5. `rsl_rl/rsl_rl/modules/actor_critic.py`
6. `rsl_rl/rsl_rl/runners/on_policy_runner.py`
7. `rsl_rl/rsl_rl/storage/rollout_storage.py`
8. `rsl_rl/rsl_rl/algorithms/ppo.py`

## 24. 建议的学习顺序

### 第一阶段：能够描述一次环境交互

应能回答：

- 48维观测分别是什么？
- 12维动作表示关节位置、速度还是力矩？
- Actor 多久执行一次？
- 环境什么时候返回奖励和 `done`？

### 第二阶段：理解奖励塑造行为

尝试预测：

- 增大 `orientation` 惩罚会发生什么？
- 删除 `action_rate` 会不会抖动？
- `feet_air_time` 太大会出现什么行为？
- 为什么力矩惩罚权重很小但仍可能影响明显？

### 第三阶段：理解 Actor-Critic

重点掌握：

- Actor 输出动作分布。
- Critic 输出状态价值。
- Advantage 表示结果比预期好多少。
- Critic 只辅助训练，不负责部署控制。

### 第四阶段：理解 PPO

重点掌握：

- on-policy 数据为什么很快过期。
- old log probability 为什么必须保存。
- ratio 与 clipping 如何限制策略变化。
- GAE 如何从奖励和 Critic value 得到 advantage。

### 第五阶段：理解 Sim-to-Real

再学习：

- 观测噪声。
- 物理参数随机化。
- 执行器网络。
- 控制延迟。
- 特权 Critic。
- 真机安全限幅和状态估计。

## 25. 最重要的整体认识

Actor-Critic 并不是两个网络互相比赛，而是明确分工：

```text
Actor：我现在应该怎么动？
Critic：当前处境下，这次结果比原本预期好还是差？
PPO：利用 Critic 的判断，小幅、稳定地修改 Actor。
Reward：定义什么叫“走得好”。
Environment：执行动作并产生后果。
```

在四足强化学习中，神经网络结构通常不是最困难的部分。真正决定策略行为的是整个闭环：

```text
观测是否充分
    + 动作定义是否合理
    + 奖励是否存在漏洞
    + 物理和执行器是否可信
    + PPO 更新是否稳定
    + 随机化是否覆盖真机差异
```

理解这条闭环后，再调整某个奖励权重、网络大小或 PPO 参数，才会知道改动为什么有效，而不只是反复试数值。

