"""官方 FR3 Duo + 可变形线缆的第一阶段 ManiSkill 环境。"""

from typing import Any

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import (
    DefaultMaterialsConfig,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)

from ..actors.rope import RopeSpec, build_rope
from ..robot import FR3DualArm


@register_env("FR3Cable-v1", max_episode_steps=400)
class FR3CableEnv(BaseEnv):
    """先验证线缆生成、落到桌面、稳定和可重复重置，不定义抓取成功。"""

    SUPPORTED_ROBOTS = [FR3DualArm.uid]
    SUPPORTED_REWARD_MODES = ["none"]

    table_center = np.array([0.65, 0.0, 0.18])
    table_half_size = np.array([0.35, 0.45, 0.025])
    rope_spawn_height = 0.34

    def __init__(
        self,
        *args,
        robot_uids=FR3DualArm.uid,
        rope_length=0.5,
        rope_radius=0.006,
        rope_segments=20,
        **kwargs,
    ):
        self.rope_spec = RopeSpec(
            length=rope_length,
            radius=rope_radius,
            segments=rope_segments,
        )
        self.rope_spec.validate()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def table_top_z(self):
        return float(self.table_center[2] + self.table_half_size[2])

    @property
    def _default_sim_config(self):
        # 线缆比普通刚体需要更小的物理步长和更多约束求解迭代。
        return SimConfig(
            sim_freq=200,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.005,
                solver_position_iterations=30,
                solver_velocity_iterations=4,
                enable_ccd=True,
            ),
            default_materials_config=DefaultMaterialsConfig(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21,
                max_rigid_patch_count=2**19,
                found_lost_pairs_capacity=2**25,
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.15, -1.1, 0.85], [0.5, 0.0, 0.28])
        return [
            CameraConfig("base_camera", pose, 256, 256, 1.0, 0.01, 100.0)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 0.9], [0.48, 0.0, 0.3])
        return CameraConfig(
            "render_camera", pose, 960, 720, 1.0, 0.01, 100.0
        )

    def _load_agent(self, options: dict):
        # 初始构建位置与 Agent rest_pose 一致；每次 reset 仍会再次设置完整状态。
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.05]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)

        table_builder = self.scene.create_actor_builder()
        table_builder.add_box_collision(half_size=self.table_half_size)
        table_builder.add_box_visual(
            half_size=self.table_half_size,
            material=[0.55, 0.58, 0.62, 1.0],
        )
        table_builder.set_initial_pose(sapien.Pose(p=self.table_center))
        self.table = table_builder.build_static(name="cable_table")

        material = physx.PhysxMaterial(
            static_friction=0.9, dynamic_friction=0.7, restitution=0.0
        )
        self._rope_initial_pose = self._make_rope_pose()
        self.rope = build_rope(
            self.scene,
            name="cable",
            initial_pose=self._rope_initial_pose,
            spec=self.rope_spec,
            material=material,
        )
        self.rope_segments = [
            self.rope.links_map[f"rope_{index}"]
            for index in range(self.rope_spec.segments)
        ]

    def _make_rope_pose(self):
        # 第一段中心位于负 y 端，整根直线线缆沿 +y 穿过双臂公共工作区。
        first_center_y = -(
            (self.rope_spec.segments - 1) * self.rope_spec.segment_length / 2
        )
        return sapien.Pose(
            p=[0.55, first_center_y, self.rope_spawn_height],
            q=[np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # Agent.reset() 自己不会选择 keyframe，所以在任务中显式恢复官方 ready。
            keyframe = self.agent.keyframes["rest"]
            self.agent.robot.set_pose(keyframe.pose)
            self.agent.reset(torch.as_tensor(keyframe.qpos, device=self.device))

            b = len(env_idx)
            initial = self._make_rope_pose()
            rope_pose = Pose.create_from_pq(
                p=torch.as_tensor(
                    initial.p, dtype=torch.float32, device=self.device
                ).repeat(b, 1),
                q=torch.as_tensor(
                    initial.q, dtype=torch.float32, device=self.device
                ).repeat(b, 1),
            )
            self.rope.set_pose(rope_pose)
            self.rope.set_qpos(
                torch.zeros((b, self.rope.max_dof), device=self.device)
            )
            self.rope.set_qvel(
                torch.zeros((b, self.rope.max_dof), device=self.device)
            )
            self.rope.set_root_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.rope.set_root_angular_velocity(torch.zeros((b, 3), device=self.device))

    @property
    def rope_segment_positions(self):
        return torch.stack([link.pose.p for link in self.rope_segments], dim=1)

    def evaluate(self):
        positions = self.rope_segment_positions
        qpos = self.rope.qpos
        qvel = self.rope.qvel
        root_speed = torch.linalg.norm(self.rope.root_linear_velocity, dim=-1)
        finite = (
            torch.isfinite(positions).all(dim=(1, 2))
            & torch.isfinite(qpos).all(dim=1)
            & torch.isfinite(qvel).all(dim=1)
        )
        return {
            # 这些是体检指标，不叫 success，避免“线缆落稳”被误认为抓取任务完成。
            "rope_state_finite": finite,
            "rope_min_z": positions[..., 2].amin(dim=1),
            "rope_max_z": positions[..., 2].amax(dim=1),
            "rope_max_joint_speed": qvel.abs().amax(dim=1),
            "rope_root_speed": root_speed,
        }

    def _get_obs_extra(self, info: dict):
        positions = self.rope_segment_positions
        obs = {
            "left_tcp_pose": self.agent.left_tcp.pose.raw_pose,
            "right_tcp_pose": self.agent.right_tcp.pose.raw_pose,
            # 每个 state 字段保持二维 (num_envs, feature)，这样 ManiSkill 的
            # 默认扁平 state 模式和保留字典结构的 state_dict 模式都能使用。
            "rope_head_position": positions[:, 0],
            "rope_tail_position": positions[:, -1],
            "rope_center_position": positions.mean(dim=1),
        }
        if self.obs_mode_struct.use_state:
            obs.update(rope_qpos=self.rope.qpos, rope_qvel=self.rope.qvel)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        raise RuntimeError("FR3Cable-v1 第一阶段不定义奖励；请使用 reward_mode='none'")
