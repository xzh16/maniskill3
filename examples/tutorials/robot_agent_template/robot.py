"""具体机器人的配置入口。

复制模板后主要修改这个文件。它只描述“机器人是什么”，加载和控制的公共实现
留在 agent.py。空配置会被 check.py 明确拒绝，避免错误进入 PhysX 后才暴露。
"""

from pathlib import Path

import sapien

from mani_skill.agents.registration import register_agent

from .agent import RobotAgentTemplate


@register_agent()
class MyRobot(RobotAgentTemplate):
    """把类名改成具体机器人名称，例如 FR3；随后逐项填写以下字段。"""

    # ---------- 1. 注册名称与模型 ----------
    # 装饰器在本模块被 import 时执行，并把 uid -> MyRobot 写入注册表。
    uid = "my_robot"  # 每个机器人配置使用唯一名字。
    # Path(__file__).parent 使路径以本文件为基准，不受启动命令所在目录影响。
    urdf_path = str(Path(__file__).parent / "assets" / "robot.urdf")

    # ---------- 2. 活动关节如何分组 ----------
    # 填写实际关节名，允许不同数量的手臂关节；不带夹爪时保留空列表/空字典。
    # 所有非 fixed 关节都必须出现且只能出现一次；check.py 会验证完整覆盖。
    arm_joint_names = []
    gripper_joint_names = []
    # 跟随夹爪示意：{"follower": {"joint": "source", "multiplier": 1, "offset": 0}}
    # 独立夹爪关节则保持 {}，由普通位置控制器分别控制。
    gripper_mimic = {}
    gripper_lower = None  # None 使用 URDF 限位；也可填写标量或每关节数组。
    gripper_upper = None

    # ---------- 3. 任务常用 link 与安全初始状态 ----------
    ee_link_name = ""  # TCP link 的名称。
    finger_link_names = []  # 可选：所有需要缓存的手指 link。
    # 使用字典是为了通过关节名绑定数值；模板会按 SAPIEN 的实际 qpos 顺序重排。
    rest_qpos = {}  # 必须覆盖所有活动关节，例如 {"joint1": 0.0, "finger": 0.02}。
    # rest_pose 移动的是整台机器人的根节点，rest_qpos 改变的是内部关节。
    rest_pose = sapien.Pose()  # 底座在世界中的位姿；应避免与地面相交。

    # ---------- 4. PD 控制参数 ----------
    # 示例起始值，按实际模型修改，或分别为每个关节指定数值。
    # stiffness 决定追赶目标的强度，damping 抑制振荡，force_limit 限制输出。
    arm_stiffness = 100.0
    arm_damping = 10.0
    arm_force_limit = 100.0
    arm_delta_limit = 0.05  # rad 或 m，取决于关节类型。
    gripper_stiffness = 100.0
    gripper_damping = 10.0
    gripper_force_limit = 100.0

    # ---------- 5. 后续任务能力 ----------
    # 需要抓取时可继续配置 urdf_config（摩擦）、_sensor_configs（相机），
    # 并实现 is_grasping/is_static；第一版通用模板不猜测这些任务相关参数。
