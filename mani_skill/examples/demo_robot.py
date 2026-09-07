"""
Instantiates a empty environment with a floor, and attempts to place any given robot in there

这是一个“机器人单机测试器”：它会创建一个只有地面的 Empty-v1 环境，
加载指定的 Agent（机器人），并打开 SAPIEN 可视化窗口。
它不包含方块、线缆或任务成功条件，非常适合检查新机器人是否能正确加载和控制。
"""

from dataclasses import dataclass
from typing import Annotated, Optional

import gymnasium as gym
import tyro

# 导入 mani_skill 时，内置环境和机器人会被注册，之后才能用字符串 ID 创建它们。
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv


@dataclass
class Args:
    """命令行参数。

    tyro 会根据这个 dataclass 自动生成命令行界面。例如：
    ``-r panda`` 会把 robot_uid 设为 ``"panda"``。
    """

    # 机器人的注册名称；-r 是 --robot-uid 的简写。
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"

    # 物理计算后端。auto 会自动选择，也可显式使用 physx_cpu/physx_cuda。
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"

    # 机器人的控制模式。pd_joint_pos 表示输入的是目标关节位置。
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"

    # 要显示的预设姿态（keyframe）名称；None 表示自动使用第一个。
    keyframe: Annotated[Optional[str], tyro.conf.arg(aliases=["-k"])] = None

    # SAPIEN 渲染器的 shader 配置。
    shader: str = "default"

    # 以下四个开关决定主循环中给机器人发送什么动作。
    keyframe_actions: bool = False
    random_actions: bool = False
    none_actions: bool = False
    zero_actions: bool = False

    # 物理仿真每秒运行 100 次，控制器每秒接收 20 次动作。
    # 因此每个控制步中包含 100 / 20 = 5 个物理步。
    sim_freq: int = 100
    control_freq: int = 20

    # 随机种子。注意：当前 main() 中仍固定使用 seed=0，这个参数尚未被使用。
    seed: Annotated[Optional[int], tyro.conf.arg(aliases=["-s"])] = None


def main(args: Args):
    # gym.make 根据注册 ID 创建环境。Empty-v1 只提供基础场景，
    # robot_uids 决定在这个场景中加载哪一个 Agent。
    env = gym.make(
        "Empty-v1",
        # 这个演示只用来观察机器人，不需要任务观测和奖励。
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        # 分别配置传感器、录像相机和交互式 Viewer 的渲染方式。
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        # human 表示打开人可交互的 SAPIEN 窗口。
        render_mode="human",
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
    )

    # reset 会建立/重置场景、机器人和控制器状态。
    # 固定 seed=0 使每次启动尽量得到相同的初始结果。
    env.reset(seed=0)

    # gym.make 会在环境外包装 TimeLimit 等 wrapper。unwrapped 取出真正的 BaseEnv，
    # 这样就能直接访问 env.agent、env.scene 等 ManiSkill 属性。
    env: BaseEnv = env.unwrapped

    print(f"Selected robot {args.robot_uid}. Control mode: {args.control_mode}")
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())

    # qpos 是机器人所有活动关节的当前位置。乘以 0 可得到形状相同的全零数组。
    # 这里先清零，如果机器人有 keyframe，下面会再用 keyframe 覆盖它。
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)

    # kf 最终会指向选中的 Keyframe 对象。
    kf = None
    if len(env.agent.keyframes) > 0:
        kf_name = None
        if args.keyframe is not None:
            # 用户通过 -k 指定了 keyframe，按名称取出它。
            kf_name = args.keyframe
            kf = env.agent.keyframes[kf_name]
        else:
            # 没有指定时，使用字典中的第一个 keyframe。
            for kf_name, kf in env.agent.keyframes.items():
                # keep the first keyframe we find
                break

        if kf.qpos is not None:
            # 设置机器人内部关节角，然后重置控制器，
            # 避免控制器仍然追赶之前的旧目标而导致突然跳动。
            env.agent.robot.set_qpos(kf.qpos)
            env.agent.controller.reset()
        if kf.qvel is not None:
            # qvel 是关节速度；多数静态 keyframe 不需要单独指定它。
            env.agent.robot.set_qvel(kf.qvel)

        # pose 是整台机器人根节点在世界中的位置和姿态；
        # 它与表示内部关节位置的 qpos 是两个不同概念。
        env.agent.robot.set_pose(kf.pose)
        if kf_name is not None:
            print(f"Viewing keyframe {kf_name}")

    # GPU 仿真的状态保存在 GPU 缓冲区中，手动修改 qpos/pose 后需要同步。
    # physx_cpu 后端不会进入这个分支。
    if env.gpu_sim_enabled:
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()

    # 第一次 render() 创建/获取 Viewer。之后将它设为暂停，
    # 所以打开窗口后需要在 Control 面板中取消勾选 Pause。
    viewer = env.render()
    viewer.paused = True
    viewer = env.render()

    # 主循环：生成一个动作 -> 让物理仿真前进 -> 重新绘制窗口。
    while True:
        if args.random_actions:
            # 从合法动作空间中随机采样，只用于快速检查机器人能否运动。
            env.step(env.action_space.sample())
        elif args.none_actions:
            # 不发送机器人命令，仅让重力、碰撞等物理过程继续。
            env.step(None)
        elif args.zero_actions:
            # 创建形状正确的全零动作。注意：pd_joint_pos 下的 0 是“目标关节角为 0”，
            # 而 pd_joint_delta_pos 下的 0 才通常表示“保持当前关节位置”。
            env.step(env.action_space.sample() * 0)
        elif args.keyframe_actions:
            assert kf is not None, "this robot has no keyframes, cannot use it to set actions"
            if isinstance(env.agent.controller, DictController):
                # Panda 的“机械臂 + 夹爪”是一个组合控制器。from_qpos 会把完整 qpos
                # 转换成该组合控制器所需的动作格式。
                env.step(env.agent.controller.from_qpos(kf.qpos))
            else:
                env.step(kf.qpos)

        # 将最新的仿真状态绘制到 SAPIEN 窗口。
        viewer = env.render()


if __name__ == "__main__":
    # tyro.cli(Args) 解析命令行并创建 Args 实例，然后交给 main()。
    main(tyro.cli(Args))
