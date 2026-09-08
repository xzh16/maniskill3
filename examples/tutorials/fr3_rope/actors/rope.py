"""把多段胶囊刚体连接成可弯曲线缆。

设计参考 ``reference/Rope-Actor/create_actors.py``：相邻两段之间依次放置
x/y/z 三个旋转关节，从而近似一小段可扭转、可弯曲的线缆。这里修正了原接口中
可能出现负胶囊长度的问题，并去掉与建模无关的示例代码。
"""

from dataclasses import dataclass

import numpy as np
import sapien
import sapien.physx as physx

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.articulation import Articulation


@dataclass(frozen=True)
class RopeSpec:
    """线缆的几何和物理参数，所有长度单位均为米。"""

    length: float = 0.5
    radius: float = 0.006
    segments: int = 20
    twist_limit_deg: float = 85.0
    bend_limit_deg: float = 85.0
    joint_friction: float = 0.05
    joint_damping: float = 0.2
    density: float = 800.0
    color: tuple[float, float, float, float] = (0.08, 0.25, 0.8, 1.0)

    @property
    def segment_length(self):
        return self.length / self.segments

    @property
    def collision_half_length(self):
        # SAPIEN capsule 的 half_length 不包含两端半球。略微缩短可避免相邻段
        # 初始时互相穿入，同时仍让整根线缆在视觉上保持连续。
        return self.segment_length / 2 - 1.05 * self.radius

    def validate(self):
        values = (
            self.length,
            self.radius,
            self.twist_limit_deg,
            self.bend_limit_deg,
            self.joint_damping,
            self.density,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("线缆长度、半径、角度、阻尼和密度必须是有限正数")
        if not isinstance(self.segments, int) or not 2 <= self.segments <= 85:
            raise ValueError("segments 必须是 2 到 85 之间的整数")
        if not np.isfinite(self.joint_friction) or self.joint_friction < 0:
            raise ValueError("joint_friction 必须是有限非负数")
        if self.collision_half_length <= 0:
            raise ValueError(
                "每段太短或半径太大：需要 length/segments > 2.1*radius"
            )
        color = np.asarray(self.color, dtype=float)
        if color.shape != (4,) or not np.isfinite(color).all():
            raise ValueError("color 必须是四个有限 RGBA 数值")


def build_rope(
    scene: ManiSkillScene,
    *,
    name: str,
    initial_pose: sapien.Pose,
    spec: RopeSpec = RopeSpec(),
    material: physx.PhysxMaterial | None = None,
    fix_root_link: bool = False,
) -> Articulation:
    """构建线缆 Articulation；N 段会产生 ``3*(N-1)`` 个自由度。"""

    spec.validate()
    builder = scene.create_articulation_builder()
    builder.set_initial_pose(initial_pose)
    segment_builders = []
    half_segment = spec.segment_length / 2
    twist_limit = np.deg2rad(spec.twist_limit_deg)
    bend_limit = np.deg2rad(spec.bend_limit_deg)

    # SAPIEN 四元数顺序为 wxyz。关节默认绕局部 x 轴旋转；下面两个姿态
    # 分别把局部 x 轴对齐到 y、z，从而得到三个串联旋转自由度。
    sqrt_half = np.sqrt(0.5)
    joint_axis_y = [sqrt_half, 0, 0, sqrt_half]
    joint_axis_z = [sqrt_half, 0, -sqrt_half, 0]

    for index in range(spec.segments):
        if index == 0:
            segment = builder.create_link_builder()
        else:
            parent = segment_builders[-1]
            helper_x = builder.create_link_builder(parent)
            helper_x.set_name(f"rope_helper_x_{index}")
            helper_x.set_joint_name(f"rope_joint_x_{index}")
            helper_x.set_mass_and_inertia(
                mass=1e-5,
                cmass_local_pose=sapien.Pose(),
                inertia=[1e-8, 1e-8, 1e-8],
            )
            helper_x.set_joint_properties(
                "revolute",
                limits=[[-twist_limit, twist_limit]],
                pose_in_parent=sapien.Pose(p=[half_segment, 0, 0]),
                pose_in_child=sapien.Pose(),
                friction=spec.joint_friction,
                damping=spec.joint_damping,
            )

            helper_y = builder.create_link_builder(helper_x)
            helper_y.set_name(f"rope_helper_y_{index}")
            helper_y.set_joint_name(f"rope_joint_y_{index}")
            helper_y.set_mass_and_inertia(
                mass=1e-5,
                cmass_local_pose=sapien.Pose(),
                inertia=[1e-8, 1e-8, 1e-8],
            )
            helper_y.set_joint_properties(
                "revolute",
                limits=[[-bend_limit, bend_limit]],
                pose_in_parent=sapien.Pose(q=joint_axis_y),
                pose_in_child=sapien.Pose(q=joint_axis_y),
                friction=spec.joint_friction,
                damping=spec.joint_damping,
            )

            segment = builder.create_link_builder(helper_y)
            segment.set_joint_name(f"rope_joint_z_{index}")
            segment.set_joint_properties(
                "revolute",
                limits=[[-bend_limit, bend_limit]],
                pose_in_parent=sapien.Pose(q=joint_axis_z),
                pose_in_child=sapien.Pose(
                    p=[-half_segment, 0, 0], q=joint_axis_z
                ),
                friction=spec.joint_friction,
                damping=spec.joint_damping,
            )

        segment.set_name(f"rope_{index}")
        segment.add_capsule_collision(
            radius=spec.radius,
            half_length=spec.collision_half_length,
            density=spec.density,
            material=material,
        )
        segment.add_capsule_visual(
            radius=spec.radius,
            # 视觉胶囊略微重叠，避免画面中出现段间空隙。
            half_length=half_segment,
            material=list(spec.color),
        )
        segment_builders.append(segment)

    return builder.build(name=name, fix_root_link=fix_root_link)
