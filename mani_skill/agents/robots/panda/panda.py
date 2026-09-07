"""Franka Emika Panda 在 ManiSkill 中的 Agent 实现。

这个文件不定义具体任务，只定义“Panda 机器人是什么、怎么控制”：

1. 从哪个 URDF 加载机器人；
2. 手臂和夹爪分别包含哪些关节；
3. 支持哪些关节空间/末端空间控制模式；
4. 哪些 link 是左右手指和 TCP；
5. 如何判断物体被夹住。

以后实现 FR3 Agent 时，可以沿用这个结构，但 URDF、关节名、link 名和
控制参数都必须以 FR3 的真实模型为准，不能只把 ``panda`` 字符串改名。
"""

from copy import deepcopy
from typing import cast

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


# 将该类注册到 ManiSkill 的 Agent 注册表。注册后，环境才能通过
# ``robot_uids="panda"`` 这样的字符串找到并实例化这个类。
@register_agent()
class Panda(BaseAgent):
    """Panda 机器人的模型、控制器和便利接口。"""

    # uid 是 Agent 在 ManiSkill 中的唯一注册名称。
    uid = "panda"

    # URDF 描述了 link、joint、质量/惯量、视觉网格和碰撞网格。
    # PACKAGE_ASSET_DIR 指向 mani_skill/assets。
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf"

    # 在加载 URDF 时覆盖部分物理材质参数。默认摩擦往往不足以稳定抓取，
    # 因此这里给左右手指分配摩擦系数为 2.0 的 gripper 材质。
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            panda_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    # keyframe 是可复用的机器人预设状态。rest 中的 qpos 共有 9 项：
    # 前 7 项是七个手臂关节角，最后 2 项是左右手指的位置（单位为米）。
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    # 两根手指各离中心 0.04 m，表示夹爪打开。
                    0.04,
                    0.04,
                ]
            ),
            # 单位 Pose 表示机器人根节点位于世界原点，且没有额外旋转。
            pose=sapien.Pose(),
        )
    )

    # 这些名称必须与 URDF 里的活动关节名完全一致。
    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    # Panda 夹爪在 URDF 中有两个平移关节，但对外只需要一个开合命令。
    gripper_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    # TCP（Tool Center Point）是做末端位置/姿态控制时使用的参考 link，
    # 它通常位于两根手指之间。
    ee_link_name = "panda_hand_tcp"

    # PD 控制器的主要参数：
    # stiffness 类似“追赶目标的力度”，damping 用来抑制震荡，force_limit 限制输出力。
    # 这些数值是仿真参数，不能不加验证地照搬到另一台机器人。
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        """构建 Panda 支持的所有控制模式。

        每个最终控制模式都由手臂控制器和夹爪控制器组成。例如：
        ``pd_joint_pos = {arm: 关节位置控制, gripper: 夹爪位置控制}``。

        对初学者，先关注 ``pd_joint_pos`` 和 ``pd_joint_delta_pos`` 即可。
        """

        # -------------------------------------------------------------------------- #
        # Arm：机械臂的各种控制方式
        # -------------------------------------------------------------------------- #

        # 绝对关节位置控制：action 直接表示 7 个手臂关节的目标角度。
        # lower/upper=None 表示使用 URDF 中的关节限位；normalize_action=False
        # 表示输入值就是实际弧度，不是归一化后的 [-1, 1]。
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        # 增量关节位置控制：新目标 = 当前关节位置 + action。
        # 归一化 action 的 [-1, 1] 会被缩放到每步 [-0.1, 0.1] rad。
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        # target_delta 版本不是相对“当前真实 qpos”累加，而是相对“上一次目标”累加。
        # deepcopy 用来避免修改新配置时意外改到原配置。
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # 末端位置增量控制：action 只指定 TCP 的 x/y/z 位移，内部再用 IK
        # （逆运动学）把 TCP 目标转换成 7 个关节的目标角。
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=cast(str, self.urdf_path),
        )

        # 末端位置+姿态增量控制：action 同时控制 TCP 的 3 维平移和 3 维旋转。
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=cast(str, self.urdf_path),
        )

        # 绝对末端姿态控制：action 是世界/参考坐标系中的目标 TCP pose，而非增量。
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-2.0,
            pos_upper=2.0,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=cast(str, self.urdf_path),
            use_delta=False,
            normalize_action=False,
        )

        # 与关节 target_delta 同理，这两个模式在“上一次末端目标”上累加新动作。
        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # 关节速度控制：action 表示每个关节的目标角速度。
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # 同时控制关节位置和速度。这类模式的 action 会包含两组目标。
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper：夹爪控制
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior

        # 夹爪在 URDF 中有两个关节，但希望用一个动作同步开合，因此使用 MimicController。
        # action=0.04 接近完全打开；下限故意给到 -0.01，使手指遇到细物体后
        # 仍然会产生夹持力，而不是刚好停在几何接触点。
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            # finger2 跟随 finger1，所以外部只需要提供一个夹爪动作。
            mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}},
        )

        # 最终暴露给环境/用户的控制模式字典。每个模式都是一个组合控制器：
        # arm 负责七轴机械臂，gripper 负责夹爪。例如 pd_joint_pos 的动作维度是
        # 7 个手臂目标 + 1 个夹爪目标 = 8，而机器人 qpos 仍有 9 项。
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        """URDF 加载完成后，缓存任务中经常用到的 link。

        ``links_map`` 是 ``{link_name: Link}`` 字典。提前保存这三个引用后，
        环境就可以直接读取 ``self.agent.tcp.pose`` 或检查手指接触。
        """

        self.finger1_link = self.robot.links_map["panda_leftfinger"]
        self.finger2_link = self.robot.links_map["panda_rightfinger"]
        self.tcp = self.robot.links_map[self.ee_link_name]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """判断左右手指是否从合理方向同时夹住了物体。

        Args:
            object (Actor): 需要检查的刚体物件。
            min_force: 单根手指需要达到的最小接触力，单位为 N。
            max_angle: 接触力与手指合拢方向的最大夹角，单位为度。

        Returns:
            形状为 ``(num_envs,)`` 的布尔 Tensor，每个并行环境对应一个结果。
        """

        # 分别查询左/右手指与目标物体之间的三维接触力向量。
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )

        # 向量范数就是接触力大小。
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # 从手指的世界变换矩阵中取出局部 y 轴，构造两根手指的夹持方向。
        # 右手指的局部轴方向与左手指相反，所以需要取负号。
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]

        # 计算夹持方向与实际接触力的夹角。只有力的方向也合理时，
        # 才把接触认为“夹住”，而不是手指侧面偶然碰到物体。
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)

        # 左右手指都必须同时满足“力够大”和“方向正确”。
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        """检查七个手臂关节是否基本静止。

        ``[..., :-2]`` 排除最后两个夹爪关节；所有手臂关节速度的绝对值
        都不超过 threshold 时，才返回 True。
        """

        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        """返回 TCP 的三维位置，形状通常是 ``(num_envs, 3)``。"""

        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        """返回 TCP 的完整位置和旋转姿态。"""

        return self.tcp.pose

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """根据接近方向、夹爪合拢方向和抓取中心构造 TCP 抓取姿态。

        ``approaching`` 和 ``closing`` 必须是互相垂直的单位向量。函数使用它们
        构造一组正交坐标轴，再把 ``center`` 作为位移，最终返回 sapien.Pose。
        """

        # 两个方向向量的长度必须约等于 1。
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3

        # 点积约等于 0 表示两个方向互相垂直。
        assert np.abs(approaching @ closing) <= 1e-3

        # 叉积生成第三根与前两根都垂直的坐标轴。
        ortho = np.cross(closing, approaching)

        # T 是 4x4 齐次变换矩阵：左上角 3x3 是旋转，最后一列前 3 项是位置。
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
