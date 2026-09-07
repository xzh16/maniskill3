"""用之前已验证的配置验收模板；Panda 依赖仅存在于这个可选示例中。"""

from mani_skill.agents.registration import register_agent

from ..custom_agent.custom_panda import CustomPanda
from .agent import RobotAgentTemplate


@register_agent()
class TemplatePanda(RobotAgentTemplate):
    """已知可运行的验收夹具，不是新机器人必须继承的父类。"""

    # 配置取自旧示例，实际加载和控制逻辑全部走通用模板。
    # 这里继承 RobotAgentTemplate，而不是继承 Panda/CustomPanda；CustomPanda
    # 仅被当作“已验证参数来源”，证明通用实现可以承载一套真实机器人配置。
    uid = "template_panda"
    urdf_path = CustomPanda.urdf_path
    arm_joint_names = CustomPanda.arm_joint_names.copy()
    gripper_joint_names = CustomPanda.gripper_joint_names.copy()
    gripper_mimic = {"panda_finger_joint2": {"joint": "panda_finger_joint1"}}
    ee_link_name = CustomPanda.ee_link_name
    finger_link_names = ["panda_leftfinger", "panda_rightfinger"]
    rest_qpos = dict(
        zip(arm_joint_names + gripper_joint_names, CustomPanda.keyframes["rest"].qpos)
    )
    rest_pose = CustomPanda.keyframes["rest"].pose
    arm_stiffness = gripper_stiffness = 1e3
    arm_damping = gripper_damping = 1e2
    gripper_lower = 0.0
    gripper_upper = 0.04
