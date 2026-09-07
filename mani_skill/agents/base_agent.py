"""ManiSkill 中所有机器人 Agent 的基类。

初学者可以把 Agent 理解成环境与机器人模型之间的“适配层”：

* SAPIEN/PhysX 的 ``Articulation`` 负责保存机器人本体和物理状态；
* Controller 负责把动作转换成关节的位置、速度或力目标；
* BaseAgent 把机器人本体、控制器、传感器和状态接口组织到一起；
* 具体机器人（例如 Panda）继承 BaseAgent，并填写 URDF、控制器和专用接口。

一次动作的大致流向是：
``BaseEnv.step(action) -> BaseAgent.set_action(action) -> Controller -> PhysX``。

阅读本文件时建议先看 ``__init__``、``_load_articulation``、
``set_control_mode``、``set_action``、``get_proprioception`` 和 ``reset``。
资产下载、多机器人分别构建和 GPU 同步属于进阶内容，可以稍后再看。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import sapien
import torch
from gymnasium import spaces

from mani_skill import format_path
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosControllerConfig
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.utils import assets, download_asset, sapien_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs import Actor, Array, Articulation
from mani_skill.utils.structs.pose import Pose

from .controllers.base_controller import (
    BaseController,
    CombinedController,
    ControllerConfig,
)

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.building.mjcf_loader import MJCFLoader
    from mani_skill.utils.building.urdf_loader import URDFLoader
DictControllerConfig = dict[str, ControllerConfig]


@dataclass
class Keyframe:
    """机器人的一个预设状态，例如 Panda 的 ``rest`` 姿态。

    ``pose`` 控制整台机器人根节点在世界坐标系中的位姿；``qpos`` 控制机器人
    内部各关节的位置；``qvel`` 控制各关节速度。这三者不要混为一谈。
    """

    pose: sapien.Pose
    """机器人根节点在世界坐标系中的位姿。"""
    qpos: Optional[Array] = None
    """该预设状态的关节位置；None 表示不指定。"""
    qvel: Optional[Array] = None
    """该预设状态的关节速度；None 表示不指定。"""


class BaseAgent:
    """Base class for agents/robots, forming an interface of an articulated robot (SAPIEN's physx.PhysxArticulation).
    Users implementing their own agents/robots should inherit from this class.
    A tutorial on how to build your own agent can be found in :doc:`its tutorial </user_guide/tutorials/custom_robots>`

    Args:
        scene (ManiSkillScene): simulation scene instance.
        control_freq (int): control frequency (Hz).
        control_mode (str | None): uid of controller to use
        fix_root_link (bool): whether to fix the robot root link
        agent_idx (str | None): an index for this agent in a multi-agent task setup If None, the task should be single-agent
        initial_pose (sapien.Pose | Pose | None): the initial pose of the robot. Important to set for GPU simulation to ensure robot
        does not collide with other objects in the scene during GPU initialization which occurs before `env._initialize_episode` is called

    对自定义机器人的作者来说，最少需要在子类中提供：

    1. ``uid``，机器人在注册表中的名字；
    2. ``urdf_path`` 或 ``mjcf_path``，机器人模型文件；
    3. 通常还会覆盖 ``_controller_configs``，给出适合该机器人的控制器。

    Panda/FR3 这类夹爪机器人还经常覆盖 ``_after_init``、``is_grasping`` 和
    ``is_static``，以缓存 TCP/手指 link 并提供任务常用判断。
    """

    uid: str
    """unique identifier string of this"""
    urdf_path: Union[str, None] = None
    """path to the .urdf file describe the agent's geometry and visuals. One of urdf_path or mjcf_path must be provided."""
    urdf_config: Union[dict, None] = None
    """Optional provide a urdf_config to further modify the created articulation"""
    mjcf_path: Union[str, None] = None
    """path to a MJCF .xml file defining a robot. This will only load the articulation defined in the XML and nothing else.
    One of urdf_path or mjcf_path must be provided."""

    fix_root_link: bool = True
    """Whether to fix the root link of the robot in place."""
    load_multiple_collisions: bool = False
    """Whether the referenced collision meshes of a robot definition should be loaded as multiple convex collisions"""
    disable_self_collisions: bool = False
    """Whether to disable self collisions. This is generally not recommended as you should be defining a SRDF file to exclude specific collisions.
    However for some robots/tasks it may be easier to disable all self collisions between links in the robot to increase simulation speed
    """

    keyframes: dict[str, Keyframe] = dict()
    """a dict of predefined keyframes similar to what Mujoco does that you can use to reset the agent to that may be of interest"""

    robot: Articulation
    """The robot object, which is an Articulation. Data like pose, qpos etc. can be accessed from this object."""

    def __init__(
        self,
        scene: ManiSkillScene,
        control_freq: int,
        control_mode: Optional[str] = None,
        agent_idx: Optional[int] = None,
        initial_pose: Optional[Union[sapien.Pose, Pose]] = None,
        build_separate: bool = False,
    ):
        # 环境先创建 ManiSkillScene，再把它交给 Agent。Agent 本身不拥有独立场景。
        self.scene = scene
        # 控制器需要知道控制频率，才能把每次 action 换算成合适的控制目标。
        self._control_freq = control_freq
        # 多机器人任务用 agent_idx 区分同一种机器人；单机器人通常为 None。
        self._agent_idx = agent_idx
        self.build_separate = build_separate

        self.controllers: dict[str, BaseController] = dict()
        """The controllers of the robot."""
        self.sensors: dict[str, BaseSensor] = dict()
        """The sensors that come with the robot."""

        # 1. 解析 URDF/MJCF，并创建 SAPIEN Articulation（真正的机器人本体）。
        self._load_articulation(initial_pose)
        # 2. 给子类一个“模型已加载、控制器尚未创建”的扩展时机。
        self._after_loading_articulation()

        # 3. 读取子类定义的控制器配置。字典的 key 就是 control_mode 名称。
        self.supported_control_modes = list(self._controller_configs.keys())
        """List of all possible control modes for this robot."""
        if control_mode is None:
            # 没有显式指定时，使用配置字典中的第一个控制模式。
            control_mode = self.supported_control_modes[0]
        # The control mode after reset for consistency
        self._default_control_mode = control_mode
        # 4. 根据配置真正创建当前控制器。
        self.set_control_mode()

        # 5. 所有基础组件就绪后调用子类钩子。Panda 在这里缓存 TCP 和手指 link。
        self._after_init()

    @property
    def _sensor_configs(self) -> list[BaseSensorConfig]:
        """返回安装在机器人上的传感器配置；默认没有传感器。

        例如腕部相机属于 Agent 传感器，而场景角落的固定相机通常属于 Env。
        """
        return []

    @property
    def _controller_configs(
        self,
    ) -> dict[str, Union[ControllerConfig, DictControllerConfig]]:
        """返回 ``{控制模式名称: 控制器配置}``。

        基类提供绝对关节位置和增量关节位置两种最小默认配置，并作用于所有
        active joints。Panda 等完整机器人会覆盖它，把手臂与夹爪组合起来。
        此处返回的只是配置对象，``set_control_mode`` 才会创建控制器实例。
        """
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                [x.name for x in self.robot.active_joints],
                lower=None,
                upper=None,
                stiffness=100,
                damping=10,
                normalize_action=False,
            ),
            pd_joint_delta_pos=PDJointPosControllerConfig(
                [x.name for x in self.robot.active_joints],
                lower=-0.1,
                upper=0.1,
                stiffness=100,
                damping=10,
                normalize_action=True,
                use_delta=True,
            ),
        )

    @property
    def device(self):
        """仿真数据所在的 torch 设备，例如 ``cpu`` 或 ``cuda:0``。"""

        return self.scene.device

    def _load_articulation(
        self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None
    ):
        """从 URDF/MJCF 加载机器人，并保存为 ``self.robot``。

        Articulation 是由多个 link 和 joint 连接起来的可动刚体系统。对外部代码
        来说，``self.robot`` 是读取 qpos/qvel、link pose 和设置关节状态的入口。
        """

        def build_articulation(scene_idxs: Optional[list[int]] = None):
            """为指定子场景解析并构建一个 Articulation。"""

            loader: Union[URDFLoader, MJCFLoader, None] = None
            if self.urdf_path is not None:
                # 大多数机械臂走这个分支，例如 Panda 使用 URDF。
                loader = self.scene.create_urdf_loader()
                asset_path = format_path(str(self.urdf_path))
            elif self.mjcf_path is not None:
                # 一些来自 MuJoCo 的机器人会使用 MJCF/XML。
                loader = self.scene.create_mjcf_loader()
                asset_path = format_path(str(self.mjcf_path))
            assert (
                loader is not None
            ), "No loader found. Provide either path in either urdf_path or mjcf_path"
            loader.name = self.uid
            if self._agent_idx is not None:
                loader.name = f"{self.uid}-agent-{self._agent_idx}"
            # 将 Agent 类上的加载选项传给 SAPIEN loader。
            loader.fix_root_link = self.fix_root_link
            loader.load_multiple_collisions_from_file = self.load_multiple_collisions
            loader.disable_self_collisions = self.disable_self_collisions

            if self.urdf_config is not None:
                # urdf_config 可覆盖摩擦、碰撞体、材质等 URDF 加载属性。
                urdf_config = sapien_utils.parse_urdf_config(self.urdf_config)
                sapien_utils.check_urdf_config(urdf_config)
                sapien_utils.apply_urdf_config(loader, urdf_config)

            if not os.path.exists(asset_path):
                # 内置机器人资产缺失时，ManiSkill 会尝试引导用户下载对应资产。
                print(f"Robot {self.uid} definition file not found at {asset_path}")
                if (
                    self.uid in assets.DATA_GROUPS
                    or len(assets.DATA_GROUPS[self.uid]) > 0
                ):
                    response = download_asset.prompt_yes_no(
                        f"Robot {self.uid} has assets available for download. Would you like to download them now?"
                    )
                    if response:
                        for (
                            asset_id
                        ) in assets.expand_data_group_into_individual_data_source_ids(
                            self.uid
                        ):
                            download_asset.download(assets.DATA_SOURCES[asset_id])
                    else:
                        print(
                            f"Exiting as assets for robot {self.uid} are not downloaded"
                        )
                        exit()
                else:
                    print(
                        f"Exiting as assets for robot {self.uid} are not found. Check that this agent is properly registered with the appropriate download asset ids"
                    )
                    exit()
            # parse 先得到 builder，build 才把机器人实体加入物理场景。
            builder = loader.parse(asset_path)["articulation_builders"][0]
            builder.initial_pose = initial_pose
            if scene_idxs is not None:
                # GPU 并行环境由多个子场景组成，可限制该机器人属于哪些子场景。
                builder.set_scene_idxs(scene_idxs)
                builder.set_name(f"{self.uid}-agent-{self._agent_idx}-{scene_idxs}")
            robot = builder.build()
            assert robot is not None, f"Fail to load URDF/MJCF from {asset_path}"
            return robot

        if self.build_separate:
            # 进阶用法：每个并行环境分别构建机器人，便于逐环境随机化外观/物理参数，
            # 最后再合并成一个批量 Articulation 接口。
            arts = []
            for scene_idx in range(self.scene.num_envs):
                robot = build_articulation([scene_idx])
                self.scene.remove_from_state_dict_registry(robot)
                arts.append(robot)
            self.robot = Articulation.merge(
                arts, name=f"{self.uid}-agent-{self._agent_idx}", merge_links=True
            )
            self.scene.add_to_state_dict_registry(self.robot)
        else:
            # 常见路径：一次构建一个可批量操作的机器人。
            self.robot = build_articulation()
        # 缓存 link 名称，分割图观测会用它识别哪些像素属于机器人。
        self.robot_link_names = [link.name for link in self.robot.get_links()]

    def _after_loading_articulation(self):
        """模型加载后、控制器创建前调用的子类扩展钩子；默认不做任何事。"""

    def _after_init(self):
        """Agent 全部初始化完成后的子类扩展钩子。

        常用于通过 ``self.robot.links_map`` 缓存 TCP、左右手指等特殊 link。
        """

    # -------------------------------------------------------------------------- #
    # Controllers
    # -------------------------------------------------------------------------- #

    @property
    def control_mode(self):
        """Get the currently activated controller uid."""
        return self._control_mode

    def set_control_mode(self, control_mode: Optional[str] = None):
        """切换并按需创建控制器。

        ``None`` 表示恢复默认模式。控制器采用懒加载：第一次使用某个模式时才
        创建实例，后续切回该模式会复用已有实例。本方法不会自动 reset 控制器。
        """
        if control_mode is None:
            control_mode = self._default_control_mode
        assert (
            control_mode in self.supported_control_modes
        ), "{} not in supported modes: {}".format(
            control_mode, self.supported_control_modes
        )
        self._control_mode = control_mode
        # 第一次使用该模式时，根据配置现场创建控制器实例。
        if control_mode not in self.controllers:
            config = self._controller_configs[self._control_mode]
            balance_passive_force = True
            if isinstance(config, dict):
                # 配置是字典时，说明它由多个部分组成，例如 Panda 的 arm + gripper。
                if "balance_passive_force" in config:
                    balance_passive_force = config.pop("balance_passive_force")
                self.controllers[control_mode] = CombinedController(
                    configs=config,
                    articulation=self.robot,
                    scene=self.scene,
                    control_freq=self._control_freq,
                )
            else:
                # 单一配置则直接创建对应 controller_cls。
                self.controllers[control_mode] = config.controller_cls(
                    config=config,
                    articulation=self.robot,
                    scene=self.scene,
                    control_freq=self._control_freq,
                )
            self.controllers[control_mode].set_drive_property()
            if balance_passive_force:
                # 当前实现通过关闭机器人 link 的重力，避免控制器还要额外抵消重力。
                # NOTE (stao): Balancing passive force is currently not supported in PhysX, so we work around by disabling gravity
                if not self.scene._gpu_sim_initialized:
                    for link in self.robot.links:
                        link.disable_gravity = True
                else:
                    for link in self.robot.links:
                        if link.disable_gravity.all() != True:
                            logger.warning(
                                f"Attemped to set control mode and disable gravity for the links of {self.robot}. However the GPU sim has already initialized with the links having gravity enabled so this will not work."
                            )

    @property
    def controller(self) -> BaseController:
        """Get currently activated controller."""
        if self._control_mode is None:
            raise RuntimeError("Please specify a control mode first")
        else:
            return self.controllers[self._control_mode]

    @property
    def action_space(self) -> spaces.Space:
        """当前控制模式的批量动作空间，用于 ``env.step``。"""

        if self._control_mode is None:
            return spaces.Dict(
                {
                    uid: controller.action_space
                    for uid, controller in self.controllers.items()
                }
            )
        else:
            return self.controller.action_space

    @property
    def single_action_space(self) -> spaces.Space:
        """单个环境的动作空间，不包含 ``num_envs`` 这一批量维度。"""

        if self._control_mode is None:
            return spaces.Dict(
                {
                    uid: controller.single_action_space
                    for uid, controller in self.controllers.items()
                }
            )
        else:
            return self.controller.single_action_space

    def set_action(self, action):
        """把动作交给当前控制器，供接下来的物理步执行。

        这里并不会直接让时间前进；真正的 ``scene.step()`` 位于 BaseEnv。
        """
        if not self.scene.gpu_sim_enabled:
            if np.isnan(action).any():
                raise ValueError("Action cannot be NaN. Environment received:", action)
        self.controller.set_action(action)

    def before_simulation_step(self):
        """每个物理小步前更新一次控制器输出。

        一个 control step 往往包含多个 simulation step，所以该方法可能在一次
        ``env.step(action)`` 中被调用多次。
        """
        self.controller.before_simulation_step()

    # -------------------------------------------------------------------------- #
    # Observations and State
    # -------------------------------------------------------------------------- #
    def get_proprioception(self):
        """返回机器人的本体感知观测。

        默认包含关节位置 ``qpos``、关节速度 ``qvel``，以及非空的控制器内部
        状态。任务环境可通过 ``BaseEnv._get_obs_extra`` 再添加线缆、目标点等信息。
        """
        obs = dict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        controller_state = self.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def get_controller_state(self):
        """
        Get the state of the controller.
        """
        return self.controller.get_state()

    def set_controller_state(self, state: dict):
        """
        Set the state of the controller.
        """
        self.controller.set_state(state)

    def get_state(self) -> dict:
        """导出机器人和控制器的完整状态，便于保存、回放或恢复。"""
        state = dict()

        # 根节点位姿/速度描述整台机器人；qpos/qvel 描述内部关节。
        root_link = self.robot.get_links()[0]
        state["robot_root_pose"] = root_link.pose
        state["robot_root_vel"] = root_link.get_linear_velocity()
        state["robot_root_qvel"] = root_link.get_angular_velocity()
        state["robot_qpos"] = self.robot.get_qpos()
        state["robot_qvel"] = self.robot.get_qvel()

        # 某些增量控制器会保存上一次目标，因此控制器状态也必须一起保存。
        state["controller"] = self.get_controller_state()

        return state

    def set_state(self, state: dict, ignore_controller=False):
        """恢复 ``get_state`` 导出的状态。

        ``ignore_controller=True`` 时只恢复机器人，不恢复控制器内部目标。
        GPU 仿真还需要显式同步缓存，CPU 仿真不需要这一步。
        """
        # robot state
        self.robot.set_root_pose(state["robot_root_pose"])
        self.robot.set_root_linear_velocity(state["robot_root_vel"])
        self.robot.set_root_angular_velocity(state["robot_root_qvel"])
        self.robot.set_qpos(state["robot_qpos"])
        self.robot.set_qvel(state["robot_qvel"])

        if not ignore_controller and "controller" in state:
            self.set_controller_state(state["controller"])
        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()  # pyright: ignore[reportAttributeAccessIssue]
            self.scene._gpu_fetch_all()

    # -------------------------------------------------------------------------- #
    # Other
    # -------------------------------------------------------------------------- #
    def reset(self, init_qpos: Optional[torch.Tensor] = None):
        """清空机器人速度和关节力，并可选择设置初始关节位置。

        Args:
            init_qpos: 初始关节位置。为 None 时保留当前 qpos。
        """
        if init_qpos is not None:
            self.robot.set_qpos(init_qpos)
        self.robot.set_qvel(torch.zeros(self.robot.max_dof, device=self.device))
        self.robot.set_qf(torch.zeros(self.robot.max_dof, device=self.device))

    # -------------------------------------------------------------------------- #
    # Optional per-agent APIs, implemented depending on agent affordances
    # -------------------------------------------------------------------------- #
    def is_grasping(self, object: Union[Actor, None] = None):
        """判断机器人是否抓住指定物体；基类只规定接口。

        Args:
            object (Actor | None):
                给出 Actor 时检查该物体；None 可表示检查是否抓住任意物体。

        Returns:
            是否抓取成功。具体判据由机器人子类实现，例如 Panda 会检查左右
            手指的接触力大小与方向。
        """
        raise NotImplementedError()

    def is_static(self, threshold: float):
        """根据关节速度判断机器人是否基本静止；基类只规定接口。

        Args:
            threshold: 允许的最大关节速度阈值。

        Returns:
            所有相关关节速度是否都在阈值内。
        """
        raise NotImplementedError()
