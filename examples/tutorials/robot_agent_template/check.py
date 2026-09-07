"""自动检查入口：默认 CPU、无窗口；失败返回非零退出码。

推荐按风险从低到高运行：
1. --static-only：只查配置和 URDF，不创建物理场景；
2. 默认：实际加载机器人，并在两种控制模式下保持 rest 姿态；
3. --motion：再给每个可控关节一个很小的目标，确认它确实能运动。
"""

import argparse
import importlib

import gymnasium as gym
import numpy as np
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import REGISTERED_AGENTS

from .validation import require, validate_definition


def load_agent(spec):
    """显式导入“模块路径:类名”，检查该类是否真的通过装饰器注册。"""

    require(":" in spec, "--agent 格式应为 模块路径:类名")
    module_name, class_name = spec.rsplit(":", 1)
    # import 模块不仅用于找到类，也会执行类上方的 @register_agent()。
    cls = getattr(importlib.import_module(module_name), class_name)
    # 复制到其他包后，模板基类的 Python 类型身份会改变；按 BaseAgent 和约定接口
    # 检查，允许统一入口继续验证复制后的模板。
    require(
        isinstance(cls, type)
        and issubclass(cls, BaseAgent)
        and callable(getattr(cls, "action_from_qpos", None)),
        "指定类必须继承本模板（提供 action_from_qpos 接口）",
    )
    registered = REGISTERED_AGENTS.get(cls.uid)
    require(
        registered is not None and registered.agent_cls is cls,
        f"{cls.uid} 未注册或 uid 已被其他类占用，请检查 @register_agent 和 uid",
    )
    return cls


def initialize_rest(env):
    """Empty-v1 只创建实体，rest 姿态需要由调用方应用。此函数用于 CPU 检查。"""

    env.reset(seed=0)
    agent = env.unwrapped.agent
    keyframe = agent.keyframes["rest"]
    # set_pose 放置底座；agent.reset(qpos) 设置内部关节和速度。
    agent.robot.set_pose(keyframe.pose)
    agent.reset(keyframe.qpos)
    # 改完状态后清空控制器内部目标，避免沿用 reset 前的目标造成突然跳动。
    agent.controller.reset()
    return keyframe.qpos.copy()


def tolerances(agent, angle_tolerance, position_tolerance):
    """按实际关节类型生成容差：旋转关节用 rad，平移关节用 m。"""

    return np.array(
        [
            position_tolerance
            if agent.joint_definitions[joint.name]["kind"] == "prismatic"
            else angle_tolerance
            for joint in agent.robot.active_joints
        ]
    )


def check_state(agent, tolerance):
    """每一步都检查有限状态、关节限位和 TCP，而不只检查最后一帧。"""
    qpos = agent.robot.qpos.cpu().numpy()[0]
    qvel = agent.robot.qvel.cpu().numpy()[0]
    tcp = agent.tcp.pose.raw_pose.cpu().numpy()
    require(
        np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(tcp).all(),
        "仿真产生 NaN/Inf，请检查模型、初始碰撞和控制参数",
    )
    limits = agent.robot.get_qlimits()[0].cpu().numpy()
    require(
        np.all(qpos >= limits[:, 0] - tolerance)
        and np.all(qpos <= limits[:, 1] + tolerance),
        "仿真中的关节超出限位容差",
    )
    return qpos


def advance(env, target, steps, tolerance, *, hold=False):
    """连续推进若干控制步；可选择同时断言机器人应保持在目标附近。"""

    agent = env.unwrapped.agent
    low = torch.as_tensor(env.action_space.low, device=agent.device)
    high = torch.as_tensor(env.action_space.high, device=agent.device)
    for _ in range(steps):
        # 根据关节的实际索引构造动作。delta 模式裁剪成每步允许的变化量。
        action = agent.action_from_qpos(target)
        require(tuple(action.shape) == (1, *env.action_space.shape), "动作维度不正确")
        require(torch.isfinite(action).all().item(), "控制动作含 NaN/Inf")
        if agent.control_mode == "pd_joint_delta_pos":
            action = torch.maximum(torch.minimum(action, high), low)
        require(env.action_space.contains(action[0].cpu().numpy()), "目标动作超出动作空间")
        # env.step 接收的是“控制频率”动作；环境内部会执行若干个物理仿真步。
        env.step(action)
        qpos = check_state(agent, tolerance)
        if hold:
            require(np.all(np.abs(qpos - target) <= tolerance), "保持 rest 姿态时发生明显漂移")
    return qpos


def motion_target(agent, rest):
    """为每个独立关节生成限位内的小幅目标，再计算 mimic 跟随关节位置。"""
    target = rest.copy()
    names = [joint.name for joint in agent.robot.active_joints]
    index = {name: i for i, name in enumerate(names)}
    limits = agent.robot.get_qlimits()[0].cpu().numpy()
    # 夹爪范围可能比 URDF 更小，以实际夹爪控制器的范围为准。
    for controller in agent.controller.controllers.values():
        if controller.config.joint_names == agent.gripper_joint_names:
            indices = controller.active_joint_indices
            if agent.gripper_mimic:
                indices = indices[controller.control_joint_indices]
            limits[indices.cpu().numpy(), 0] = controller.single_action_space.low
            limits[indices.cpu().numpy(), 1] = controller.single_action_space.high
    # 对每个 source/独立关节，选择离 rest 空间更充足的一侧作为运动方向。
    for name, i in index.items():
        if name in agent.gripper_mimic:
            continue
        lower, upper = limits[i]
        room_up, room_down = upper - rest[i], rest[i] - lower
        direction = 1 if room_up >= room_down else -1
        room = max(room_up, room_down)
        increment = (
            0.01 if agent.joint_definitions[name]["kind"] == "prismatic" else 0.05
        )
        target[i] += direction * min(increment, room / 2)
    # follower 不独立选择目标，而是严格根据 source、比例和偏移算出。
    for follower, config in agent.gripper_mimic.items():
        target[index[follower]] = target[index[config["joint"]]] * config.get(
            "multiplier", 1.0
        ) + config.get("offset", 0.0)
    return target


def run_simulation_checks(
    cls, *, steps=100, motion=False, angle_tolerance=0.02, position_tolerance=0.002
):
    """两个模式分别创建场景，避免切换控制器时使用旧的 action_space。"""
    # 每个模式创建独立环境，使 action_space、控制器目标和仿真状态互不污染。
    for mode in ("pd_joint_pos", "pd_joint_delta_pos"):
        env = gym.make(
            "Empty-v1",
            robot_uids=cls.uid,
            control_mode=mode,
            obs_mode="state_dict",
            reward_mode="none",
            num_envs=1,
            sim_backend="physx_cpu",
            render_backend="none",
            sim_config=dict(sim_freq=100, control_freq=20),
        )
        try:
            agent = env.unwrapped.agent
            require(type(agent) is cls, "环境实例化了错误的 Agent 类")
            rest = initialize_rest(env)
            # 同时查覆盖和重复；不假设控制器列表顺序等于 qpos 顺序。
            indices = agent.controller.active_joint_indices.cpu().numpy()
            require(
                sorted(indices.tolist()) == list(range(agent.robot.max_dof)),
                "加载后的控制器存在漏配或重复分配的活动关节",
            )
            # 每个普通关节占一个动作维度；mimic follower 由 source 推导，不占维度。
            expected_dim = (
                len(cls.arm_joint_names)
                + len(cls.gripper_joint_names)
                - len(cls.gripper_mimic)
            )
            require(env.action_space.shape == (expected_dim,), "实际动作空间维度与配置不符")
            require(len(rest) == agent.robot.max_dof, "keyframe 长度与实际 qpos 不符")
            tolerance = tolerances(agent, angle_tolerance, position_tolerance)
            check_state(agent, tolerance)
            # 第一阶段不要求移动，只验证重力、碰撞和 PD 参数下能否稳定保持。
            advance(env, rest, steps, tolerance, hold=True)
            print(f"[PASS] {mode}: {len(rest)} 个关节，{expected_dim} 维动作，保持姿态 {steps} 步")

            if motion:
                # 第二阶段验证“动作能发出并产生预期位移”，不是性能或抓取测试。
                target = motion_target(agent, rest)
                qpos = advance(env, target, steps, tolerance)
                errors = np.abs(qpos - target)
                require(np.all(errors <= tolerance), f"{mode}: 未跟踪到运动目标，各关节误差 {errors}")
                changed = np.abs(target - rest) > 1e-6
                require(changed.any(), "没有可用的小幅运动目标，请检查关节/夹爪限位")
                require(
                    np.all(
                        np.abs(qpos[changed] - rest[changed])
                        >= 0.2 * np.abs(target[changed] - rest[changed])
                    ),
                    "部分关节没有产生可测量运动，请检查控制参数或碰撞",
                )
                print(f"[PASS] {mode}: 小幅运动及夹爪（若有）跟踪正常")
        finally:
            env.close()


def main():
    """解析命令行、选择具体 Agent，然后串联静态检查和可选仿真检查。"""

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--agent", help="机器人配置的 模块路径:类名，默认使用 robot.py:MyRobot")
    selection.add_argument(
        "--example", action="store_true", help="使用旧 Panda 示例配置验收通用模板"
    )
    parser.add_argument("--static-only", action="store_true", help="只检查配置和文件，不启动物理仿真")
    parser.add_argument("--motion", action="store_true", help="额外验证限位内的小幅关节运动")
    parser.add_argument("--steps", type=int, default=100, help="每个控制模式、每个阶段的控制步数")
    parser.add_argument(
        "--angle-tolerance", type=float, default=0.02, help="角度误差容差（rad）"
    )
    parser.add_argument(
        "--position-tolerance", type=float, default=0.002, help="平移关节误差容差（m）"
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps 必须大于 0")
    if any(
        not np.isfinite(value) or value <= 0
        for value in (args.angle_tolerance, args.position_tolerance)
    ):
        parser.error("容差必须是有限正数")
    if args.static_only and args.motion:
        parser.error("--static-only 与 --motion 不能同时使用")
    # 默认检查等待用户填写的 robot.py；--example 才会明确改用 Panda 示例。
    spec = args.agent or f"{__package__}.robot:MyRobot"
    if args.example:
        spec = f"{__package__}.example_panda:TemplatePanda"
    try:
        # 先确认注册，再做纯文件检查，最后才启动成本更高的物理仿真。
        cls = load_agent(spec)
        joints = validate_definition(cls)
        print(f"[PASS] 注册: {cls.uid} -> {cls.__name__}")
        print(f"[PASS] URDF 与显式资产路径；{len(joints)} 个活动关节；rest/TCP/夹爪配置")
        if args.static_only:
            print("静态检查通过；尚未检查实际模型加载、视觉效果或运动。")
        else:
            run_simulation_checks(
                cls,
                steps=args.steps,
                motion=args.motion,
                angle_tolerance=args.angle_tolerance,
                position_tolerance=args.position_tolerance,
            )
            print("CPU 无窗口检查通过。碰撞形状、抓取效果和 GPU 仍需单独验证。")
        return 0
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
