"""参考内置 Panda，实现一个最小的自定义 Agent。

模型仍使用仓库已有的 Panda URDF；这是导入流程示例，还不是 FR3。
阅读顺序：注册名称 -> 模型 -> 关节与初始姿态 -> 控制器 -> TCP。
"""

import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.registration import register_agent


# Python 导入本模块时执行装饰器，建立 "custom_panda" -> CustomPanda 的映射。
# 这里只继承 BaseAgent，机器人配置由本示例显式填写。
@register_agent()
class CustomPanda(BaseAgent):
    uid = "custom_panda"

    # 使用已经随仓库提供的模型和 mesh，不需要另外下载或复制资产。
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf"
    fix_root_link = True  # 桌面机械臂的底座固定在场景中。

    # 名称必须与 URDF 中一致；以后换模型时首先重新核对这些名字。
    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    ee_link_name = "panda_hand_tcp"

    # qpos 按模型的活动关节顺序填写：7 个关节角（rad）+ 2 个手指位置（m）。
    # keyframe 只是保存的配置；启动程序需要显式把它应用到机器人。
    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array(
                [
                    0,
                    np.pi / 8,
                    0,
                    -5 * np.pi / 8,
                    0,
                    3 * np.pi / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ],
                dtype=np.float32,
            ),
        )
    )

    @property
    def _controller_configs(self):
        """只提供一个模式：7 维手臂目标 + 1 维夹爪目标。"""

        # 绝对关节位置控制：输入是目标角度，限位从 URDF 读取。
        # stiffness/damping/force_limit 参考内置 Panda，换模型后需要重新验证。
        arm = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )

        # 两个手指共享一个位置目标，第二个手指跟随第一个。
        # 本示例显式关闭归一化：0.04 表示每根手指移动到 0.04 m，夹爪打开；
        # 0 表示闭合目标。这里的动作单位与内置 Panda 的归一化夹爪动作不同。
        gripper = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=0.0,
            upper=0.04,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
            mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}},
        )

        # BaseAgent 会将 arm 和 gripper 组合成 CombinedController。
        # 每次访问都创建新配置，因此没有共享配置需要额外 deepcopy。
        return dict(pd_joint_pos=dict(arm=arm, gripper=gripper))

    def _after_init(self):
        """模型和控制器加载完成后，保存任务经常使用的 link。"""

        self.tcp = self.robot.links_map[self.ee_link_name]
        self.finger1_link = self.robot.links_map["panda_leftfinger"]
        self.finger2_link = self.robot.links_map["panda_rightfinger"]
