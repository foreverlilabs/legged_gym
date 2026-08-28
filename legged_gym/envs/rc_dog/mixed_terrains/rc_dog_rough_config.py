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

from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO


class RC_DogRoughCfg(A1RoughCfg):
    class init_state(A1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.33]  # x, y, z [m]
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

    class asset(A1RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/rc_dog/urdf/rc_dog.urdf"
        name = "rc_dog"
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False


class RC_DogRoughCfgPPO(A1RoughCfgPPO):
    class runner(A1RoughCfgPPO.runner):
        experiment_name = "rough_rc_dog"
