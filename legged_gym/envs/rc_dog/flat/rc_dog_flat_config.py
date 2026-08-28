# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs import RC_DogRoughCfg, RC_DogRoughCfgPPO


class RC_DogFlatCfg(RC_DogRoughCfg):
    class env(RC_DogRoughCfg.env):
        num_observations = 48
    class terrain( RC_DogRoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(RC_DogRoughCfg.init_state):
        pos = [0.0, 0.0, 0.36]  # x, y, z [m]
        # The URDF joint origins already encode the nominal bent-leg pose.
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FL_calf_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "FR_calf_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.0,
            "RL_calf_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.0,
            "RR_calf_joint": 0.0,
        }
    class rewards( RC_DogRoughCfg.rewards ):
            max_contact_force = 350.
            class scales ( RC_DogRoughCfg.rewards.scales ):
                orientation = -5.0
                torques = -0.000025
                feet_air_time = 2.#足端腾空时间
                # feet_contact_forces = -0.01
                #   - orientation = -5.0：强烈惩罚机身倾斜。机器人要保持躯干接近水平。
                #   - torques = -0.000025：惩罚关节力矩，避免用蛮力运动。数值较小，是因为力矩平方量级可能很大。
                #   - feet_air_time = 2.0：奖励合理的足端腾空时间，鼓励形成步态，而不是四只脚一直拖地。
                #   - max_contact_force = 350：接触力超过 350 N 才触发相关惩罚阈值。
    class commands( RC_DogRoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( RC_DogRoughCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand( RC_DogRoughCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

    class asset(RC_DogRoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/rc_dog/urdf/rc_dog.urdf"
        name = "rc_dog"
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1


class RC_DogFlatCfgPPO(RC_DogRoughCfgPPO):
    class policy( RC_DogRoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid，ELU激活函数，这里的三层MLP通常指的是三层隐藏层

    class algorithm( RC_DogRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner ( RC_DogRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_rc_dog'
        load_run = -1
        max_iterations = 300
