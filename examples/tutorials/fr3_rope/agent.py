"""与机器人型号无关的 Agent 实现。

调用顺序：子类实例化 -> __init__ 静态验证 -> BaseAgent 加载 URDF
-> _after_loading_articulation 生成 keyframe -> BaseAgent 创建控制器
-> _after_init 缓存 TCP/手指。具体字段集中填写在 robot.py 中。
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill import format_path
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
    PDJointPosMimicController,
    PDJointPosMimicControllerConfig,
)
from mani_skill.utils import common

from .validation import require, validate_definition


class RobotAgentTemplate(BaseAgent):
    """固定底座机械臂，支持任意数量的单自由度手臂关节和可选夹爪。

    本基类不注册。继承本类，填写配置并用 @register_agent 注册具体机器人。
    两种模式的动作都使用实际单位：旋转关节 rad，平移关节 m。
    """

    # 以下空值构成子类必须填写的“配置协议”。基类本身没有 @register_agent，
    # 因而不会作为一个不完整机器人出现在 ManiSkill 注册表中。
    uid = ""
    urdf_path = ""
    fix_root_link = True
    arm_joint_names = []
    gripper_joint_names = []
    gripper_mimic = {}
    ee_link_name = ""
    finger_link_names = []
    rest_qpos = {}
    rest_pose = sapien.Pose()

    # 起始调试参数，并非所有模型的最优值；可填写标量或每关节一个数值。
    arm_stiffness = 100.0
    arm_damping = 10.0
    arm_force_limit = 100.0
    arm_delta_limit = 0.05
    gripper_stiffness = 100.0
    gripper_damping = 10.0
    gripper_force_limit = 100.0
    gripper_lower = None
    gripper_upper = None

    def __init__(self, *args, **kwargs):
        """先检查类配置，再把场景、频率和控制模式交给 BaseAgent。"""

        # 在 BaseAgent 尝试加载/下载资产前报告缺失配置。
        self.joint_definitions = validate_definition(type(self))
        # BaseAgent 接受字符串路径；转换为绝对路径可使运行位置不影响模型加载。
        self.urdf_path = str(
            Path(format_path(str(self.urdf_path))).expanduser().resolve()
        )
        super().__init__(*args, **kwargs)

    def _after_loading_articulation(self):
        """模型已经进入场景、控制器尚未创建时，生成正确顺序的 rest keyframe。"""

        # URDF XML 顺序、配置列表顺序、SAPIEN qpos 顺序不一定相同。
        names = [joint.name for joint in self.robot.active_joints]
        require(set(names) == set(self.rest_qpos), "加载后的活动关节与 rest_qpos 不一致")
        require(len(names) == self.robot.max_dof, "加载模型不是每活动关节一个自由度")
        # self.robot.active_joints 的顺序就是 get_qpos/set_qpos 使用的顺序。
        qpos = np.array([self.rest_qpos[name] for name in names], dtype=np.float32)
        limits = self.robot.get_qlimits()[0].cpu().numpy()
        require(
            np.all(qpos >= limits[:, 0] - 1e-6) and np.all(qpos <= limits[:, 1] + 1e-6),
            "rest_qpos 超过 SAPIEN 实际加载的限位",
        )
        # 这里创建实例自己的字典，避免多个机器人子类共享 BaseAgent.keyframes。
        self.keyframes = dict(rest=Keyframe(pose=self.rest_pose, qpos=qpos))

    @property
    def _controller_configs(self):
        """为手臂及可选夹爪组合两种关节位置控制模式。"""

        # 绝对模式：action 表示目标关节位置，范围直接采用 URDF 关节限位。
        arm = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        # 增量手臂配置从绝对配置复制，随后只修改动作范围和 use_delta。
        # deepcopy 防止修改 delta_arm 时连带改变上面的 arm 配置。
        delta_arm = deepcopy(arm)
        delta_arm.lower = -np.asarray(self.arm_delta_limit)
        delta_arm.upper = np.asarray(self.arm_delta_limit)
        delta_arm.use_delta = True
        # 字典表示组合控制器。BaseAgent 会把各部分动作拼成一个扁平 Box。
        absolute = dict(arm=arm)
        delta = dict(arm=delta_arm)
        if self.gripper_joint_names:
            # 有 mimic 关系时，一个 source 动作驱动多个手指；否则每个夹爪关节
            # 都有自己的动作维度。
            gripper_cls = (
                PDJointPosMimicControllerConfig
                if self.gripper_mimic
                else PDJointPosControllerConfig
            )
            options = (
                dict(mimic=deepcopy(self.gripper_mimic)) if self.gripper_mimic else {}
            )
            gripper = gripper_cls(
                joint_names=self.gripper_joint_names,
                lower=self.gripper_lower,
                upper=self.gripper_upper,
                stiffness=self.gripper_stiffness,
                damping=self.gripper_damping,
                force_limit=self.gripper_force_limit,
                normalize_action=False,
                **options,
            )
            # 两个模式都让夹爪使用绝对位置，因此“手臂增量”不会改变夹爪语义。
            absolute["gripper"] = gripper
            delta["gripper"] = deepcopy(gripper)
        return dict(pd_joint_pos=absolute, pd_joint_delta_pos=delta)

    def _after_init(self):
        """模型和控制器均就绪后，把名字转换为可直接访问的 link 对象。"""

        require(self.ee_link_name in self.robot.links_map, "加载后的模型缺少 TCP link")
        self.tcp = self.robot.links_map[self.ee_link_name]
        self.finger_links = {}
        for name in self.finger_link_names:
            require(name in self.robot.links_map, f"加载后的模型缺少手指 link: {name}")
            self.finger_links[name] = self.robot.links_map[name]

    def action_from_qpos(self, target_qpos):
        """将完整 qpos 转为当前模式的批量动作，正确处理乱序关节和 mimic。

        绝对模式给目标位置；增量模式给目标减当前位置，夹爪仍为绝对位置。
        调用者应确保增量不超过 arm_delta_limit（检查脚本会分步裁剪）。
        """
        # 允许传入单个环境的一维 qpos；broadcast_to 自动补成 (num_envs, dof)。
        target = common.to_tensor(target_qpos, device=self.device)
        target = torch.broadcast_to(target, self.robot.qpos.shape)
        parts = []
        # controller.active_joint_indices 把完整 qpos 映射到 arm/gripper 各自关节，
        # 所以无需假设这两组关节在 URDF 中连续排列。
        for controller in self.controller.controllers.values():
            values = target[..., controller.active_joint_indices]
            if controller.config.use_delta:
                # PD 增量控制器收到“目标与当前值的差”，而不是绝对目标。
                values = values - controller.qpos
            if isinstance(controller, PDJointPosMimicController):
                # follower 由控制器内部计算，外部 action 只保留 source 关节。
                values = values[..., controller.control_joint_indices]
            parts.append(values)
        return torch.cat(parts, dim=-1)
