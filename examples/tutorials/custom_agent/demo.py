"""在 Empty-v1 中加载自定义 Agent，可显示窗口或运行有限步无窗口检查。"""

import argparse

import gymnasium as gym
import numpy as np

import mani_skill.envs  # 注册 ManiSkill 环境，包括 Empty-v1。

# 关键一步：创建环境之前导入模块，让 @register_agent() 执行。
# 只在磁盘上保存 custom_panda.py，并不会让 ManiSkill 自动发现它。
from .custom_panda import CustomPanda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="关闭渲染，检查后退出")
    parser.add_argument("--steps", type=int, default=100, help="无窗口模式的控制步数")
    parser.add_argument("--move", action="store_true", help="小幅摆动第一个关节并开合夹爪")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps 必须大于 0")

    # 环境只知道注册名称，BaseEnv 内部会找到 CustomPanda 并加载它的 URDF。
    # 本示例固定使用单环境 CPU 仿真，便于先理解最基本的导入过程。
    env = gym.make(
        "Empty-v1",
        robot_uids=CustomPanda.uid,
        control_mode="pd_joint_pos",
        obs_mode="state_dict",
        reward_mode="none",
        sim_backend="physx_cpu",
        render_backend="none" if args.headless else "gpu",
        render_mode=None if args.headless else "human",
        sim_config=dict(sim_freq=100, control_freq=20),
    )
    try:
        env.reset(seed=0)
        base_env = env.unwrapped
        agent = base_env.agent

        # Empty-v1 不会自动应用 Agent.keyframes，所以这里手动设置 rest 姿态。
        # 先修改机器人状态，再 reset 控制器，使其目标与新姿态一致。
        keyframe = agent.keyframes["rest"]
        agent.robot.set_pose(keyframe.pose)
        agent.reset(keyframe.qpos)
        agent.controller.reset()

        # qpos 有 9 项，而组合控制器只需要 8 项；from_qpos 合并夹爪目标。
        # 这个转换适用于本示例的非归一化绝对位置控制，不能通用于所有模式。
        rest_action = agent.controller.from_qpos(keyframe.qpos).cpu().numpy()
        print(f"Agent: {agent.uid} ({type(agent).__name__})")
        print(f"URDF: {agent.urdf_path}")
        print(f"活动关节: {[joint.name for joint in agent.robot.active_joints]}")
        print(f"qpos 形状: {tuple(agent.robot.qpos.shape)}；动作空间: {env.action_space}")
        print(f"TCP link: {agent.tcp.name}")

        viewer = None
        if not args.headless:
            viewer = env.render()
            viewer.paused = False
            print("窗口已开始运行；关闭窗口或在终端按 Ctrl+C 退出。")

        step = 0
        while not args.headless or step < args.steps:
            if viewer is not None and viewer.closed:
                break
            action = rest_action.copy()
            if args.move:
                # 用仿真时间生成连续目标：手臂小幅摆动，两个手指同步开合。
                t = step / base_env.control_freq
                action[0] += 0.15 * np.sin(t)
                action[-1] = 0.02 * (1 + np.cos(t))
            env.step(action)
            step += 1
            if viewer is not None:
                env.render()

        qpos = agent.robot.qpos.cpu().numpy()
        if not np.isfinite(qpos).all():
            raise RuntimeError("仿真产生了非有限关节位置，请检查模型和控制参数")
        print(f"完成 {step} 个控制步，关节状态有效。")
        print(f"最终 qpos: {qpos[0].round(4)}")
        print(f"TCP 位置: {agent.tcp.pose.p.cpu().numpy()[0].round(4)}")
    except KeyboardInterrupt:
        pass
    finally:
        # 无论正常退出还是中途报错，都释放场景和 Viewer。
        env.close()


if __name__ == "__main__":
    main()
