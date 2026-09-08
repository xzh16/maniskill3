"""FR3 双臂 Agent：默认使用 Franka Robotics 官方模型。

``FR3DualArm`` 是现在实际使用的官方 FR3 Duo；随示例保留的旧版 CAD 导出模型以
``FR3DualArmCAD`` 保留，便于对照和回退。两者使用不同 uid，不会互相覆盖。
"""

from pathlib import Path

import numpy as np
import sapien

from mani_skill.agents.registration import register_agent

from .agent import RobotAgentTemplate


ASSET_DIR = Path(__file__).resolve().parent / "assets"
OFFICIAL_MODEL_DIR = ASSET_DIR / "franka_description"
CAD_MODEL_DIR = ASSET_DIR / "FR3-dual-arm_description_0820"


@register_agent()
class FR3DualArm(RobotAgentTemplate):
    """官方 FR3 Duo（两台 FR3 v2 + 两个 Franka Hand）。"""

    uid = "fr3_dual_arm"
    urdf_path = str(OFFICIAL_MODEL_DIR / "urdfs" / "fr3_duo_franka_hand.urdf")
    fix_root_link = True

    # 官方 SRDF 使用 Adjacent/Never/UserDisable 等 reason；当前 SAPIEN 3.0.3
    # 只读取 reason="Default"，会漏掉手掌与 link7 等必要过滤，产生巨大假碰撞。
    # 暂时关闭机器人内部碰撞；机器人与线缆/场景之间的碰撞仍然保留。
    disable_self_collisions = True

    left_arm_joint_names = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
    right_arm_joint_names = [f"right_fr3v2_joint{i}" for i in range(1, 8)]
    arm_joint_names = left_arm_joint_names + right_arm_joint_names

    left_gripper_joint_names = [
        "left_fr3v2_finger_joint1",
        "left_fr3v2_finger_joint2",
    ]
    right_gripper_joint_names = [
        "right_fr3v2_finger_joint1",
        "right_fr3v2_finger_joint2",
    ]
    gripper_joint_names = left_gripper_joint_names + right_gripper_joint_names
    # 每只手的 finger_joint2 跟随 finger_joint1。因此机器人有 18 个 qpos，
    # 但动作只有 14 个臂关节 + 2 个夹爪开合量，共 16 维。
    gripper_mimic = {
        "left_fr3v2_finger_joint2": {"joint": "left_fr3v2_finger_joint1"},
        "right_fr3v2_finger_joint2": {"joint": "right_fr3v2_finger_joint1"},
    }

    # 官方模型直接提供左右 hand_tcp，不再把手掌原点临时当作抓取中心。
    ee_link_name = "left_fr3v2_hand_tcp"
    finger_link_names = [
        "left_fr3v2_leftfinger",
        "left_fr3v2_rightfinger",
        "right_fr3v2_leftfinger",
        "right_fr3v2_rightfinger",
    ]

    # 采用官方 SRDF 中的 ready 关节状态和 open=0.035 m 夹爪状态。
    _ready = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
    rest_qpos = {
        **dict(zip(left_arm_joint_names, _ready)),
        **dict(zip(right_arm_joint_names, _ready)),
        **dict.fromkeys(gripper_joint_names, 0.035),
    }
    # 官方 mount 的碰撞几何会略低于根坐标原点；抬高 5 cm，避免穿入地面。
    rest_pose = sapien.Pose(p=[0, 0, 0.05])

    # PD 增益沿用 ManiSkill Panda 的仿真起始值；这不是实机控制参数。
    arm_stiffness = 1000.0
    arm_damping = 100.0
    # 每组 7 个值来自官方 URDF 的 effort 上限，两组按控制器顺序拼接。
    arm_force_limit = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0] * 2
    arm_delta_limit = 0.05
    gripper_stiffness = 1000.0
    gripper_damping = 100.0
    gripper_force_limit = 100.0
    gripper_lower = 0.0
    gripper_upper = 0.04

    # 提高手指摩擦，参数与 ManiSkill 内置 Panda 一致；抓线缆时仍需调参。
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            name: dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1)
            for name in finger_link_names
        },
    )

    def _after_init(self):
        super()._after_init()
        self.left_hand = self.robot.links_map["left_fr3v2_hand"]
        self.right_hand = self.robot.links_map["right_fr3v2_hand"]
        self.left_tcp = self.robot.links_map["left_fr3v2_hand_tcp"]
        self.right_tcp = self.robot.links_map["right_fr3v2_hand_tcp"]


@register_agent()
class FR3DualArmCAD(RobotAgentTemplate):
    """随示例保留的旧版 CAD 导出模型，仅用于回退和问题对照。"""

    uid = "fr3_dual_arm_cad"
    urdf_path = str(CAD_MODEL_DIR / "urdf" / "FR3-dual-arm_description_0820.urdf")
    fix_root_link = True
    disable_self_collisions = False

    # 名称必须原样匹配 URDF，包括原文件中的 Lilnk7Joint 拼写。
    left_arm_joint_names = [
        "Llink1joint",
        "Llink2Joint",
        "Llink3Joint",
        "Llink4Joint",
        "Llink5Joint",
        "Llink6Joint",
        "Lilnk7Joint",
    ]
    right_arm_joint_names = [
        "RLink1",
        "RLink2Joint",
        "RLink3Joint",
        "RLink4Joint",
        "RLink5Joint",
        "RLink6Joint",
        "RLink7Joint",
    ]
    arm_joint_names = left_arm_joint_names + right_arm_joint_names
    left_gripper_joint_names = ["LhandFig1", "LhandFig2"]
    right_gripper_joint_names = ["RFig1Joint", "Rfig2Joint"]
    gripper_joint_names = left_gripper_joint_names + right_gripper_joint_names
    gripper_mimic = {}
    ee_link_name = "Lhand"
    finger_link_names = ["Lfig1", "Lfig2", "Rfig1", "Rfig2"]
    rest_qpos = dict.fromkeys(arm_joint_names + gripper_joint_names, 0.0)
    rest_pose = sapien.Pose(p=[0, 0, 0.8])

    arm_stiffness = 1000.0
    arm_damping = 100.0
    arm_force_limit = 100.0
    arm_delta_limit = 0.05
    gripper_stiffness = 100.0
    gripper_damping = 10.0
    gripper_force_limit = 10.0

    def _after_init(self):
        super()._after_init()
        self.left_hand = self.robot.links_map["Lhand"]
        self.right_hand = self.robot.links_map["Rhand"]
        # 旧模型没有 TCP；别名只为了让同一演示脚本可以显示两端位置。
        self.left_tcp = self.left_hand
        self.right_tcp = self.right_hand
