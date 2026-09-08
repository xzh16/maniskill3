"""实例化官方 FR3 Duo，显示窗口或进行有限步 CPU 检查；此处不加载线缆。"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # 注册 Empty-v1。
from mani_skill.utils import sapien_utils

from ..check import (
    check_state,
    initialize_rest,
    motion_target,
    tolerances,
)
from ..robot import FR3DualArm, FR3DualArmCAD
from ..validation import require


EYE = np.array([1.25, -1.25, 0.95])
TARGET = np.array([0.0, 0.0, 0.35])


def make_env(
    *,
    robot_cls=FR3DualArm,
    control_mode="pd_joint_pos",
    render_backend="none",
    render_mode=None,
):
    """固定单环境 CPU 仿真；渲染是否使用 GPU 是另一个独立选项。"""
    return gym.make(
        "Empty-v1",
        robot_uids=robot_cls.uid,
        control_mode=control_mode,
        obs_mode="state_dict",
        reward_mode="none",
        num_envs=1,
        sim_backend="physx_cpu",
        render_backend=render_backend,
        render_mode=render_mode,
        sim_config=dict(sim_freq=100, control_freq=20),
        human_render_camera_configs=dict(
            pose=sapien_utils.look_at(EYE, TARGET),
            width=960,
            height=720,
        ),
    )


def print_contacts(scene):
    """只列出穿入或有接触冲量的接触对；正间距的候选接触不等于碰撞。"""
    for contact in scene.get_contacts():
        points = list(contact.points)
        if not points:
            continue
        separation = min(point.separation for point in points)
        impulse = max(float(np.linalg.norm(point.impulse)) for point in points)
        if separation < -1e-6 or impulse > 1e-8:
            names = [body.entity.name for body in contact.bodies]
            print(
                f"  接触 {names[0]} <-> {names[1]}: "
                f"最小间距 {separation:.6g} m，最大点冲量 {impulse:.6g} N·s"
            )


def run(args):
    robot_cls = FR3DualArm if args.model == "official" else FR3DualArmCAD
    render = not args.headless or args.screenshot is not None
    env = make_env(
        robot_cls=robot_cls,
        control_mode=args.control_mode,
        render_backend=args.render_backend if render else "none",
        render_mode="rgb_array"
        if args.headless and render
        else ("human" if render else None),
    )
    try:
        rest = initialize_rest(env)
        agent = env.unwrapped.agent
        names = [joint.name for joint in agent.robot.active_joints]
        tolerance = tolerances(agent, 0.02, 0.002)
        print(f"Agent: {agent.uid}; links={len(agent.robot.links)}; DOF={len(rest)}")
        print(f"控制模式: {agent.control_mode}; 动作空间: {env.action_space.shape}")
        print(f"实际 qpos 顺序: {names}")
        gripper_action_count = len(agent.gripper_joint_names) - len(
            agent.gripper_mimic
        )
        if args.model == "official":
            print(
                "动作顺序: 左臂 7 项、右臂 7 项、"
                f"夹爪 {gripper_action_count} 项（每只手一个开合量）。"
            )
            print("模型: Franka 官方 fr3_duo；左右抓取参考为各自 hand_tcp。")
            print(
                "为兼容当前 SAPIEN 的 SRDF 过滤规则，暂时关闭机器人内部碰撞。"
            )
        else:
            print("动作顺序: 左臂 7 项、右臂 7 项、四根手指各 1 项。")
            print(
                "模型: 随示例保留的旧版 CAD 导出，仅用于问题对照；没有专用 TCP。"
            )
        print(f"机器人内部碰撞启用: {not agent.disable_self_collisions}")

        target = rest.copy()
        if args.motion != "hold":
            target = motion_target(agent, rest)
            if args.motion == "arms":
                # arms 模式只验证两条机械臂，让所有手指保持原位。
                for i, name in enumerate(names):
                    if name in agent.gripper_joint_names:
                        target[i] = rest[i]

        viewer = None
        if not args.headless:
            viewer = env.render()
            viewer.paused = False
            viewer.set_camera_xyz(*EYE)
            direction = TARGET - EYE
            viewer.set_camera_rpy(
                0,
                -np.arctan2(direction[2], np.linalg.norm(direction[:2])),
                np.arctan2(direction[1], direction[0]),
            )
            print("关闭窗口或按 Ctrl+C 退出。窗口运行时保持所选目标姿态。")

        low = torch.as_tensor(env.action_space.low, device=agent.device)
        high = torch.as_tensor(env.action_space.high, device=agent.device)
        step = 0
        # 先保持 100 步，再发送运动目标，便于从窗口观察前后变化。
        warmup = 100 if args.motion != "hold" else 0
        maximum_hold_error = np.zeros_like(rest)
        qpos = check_state(agent, tolerance)
        while (args.steps == 0 or step < args.steps + warmup) and (
            viewer is None or not viewer.closed
        ):
            start_time = time.monotonic()
            goal = rest if step < warmup else target
            action = agent.action_from_qpos(goal)
            if args.control_mode == "pd_joint_delta_pos":
                action = torch.maximum(torch.minimum(action, high), low)
            require(env.action_space.contains(action[0].cpu().numpy()), "动作超出动作空间")
            env.step(action)
            qpos = check_state(agent, tolerance)
            if step < warmup or args.motion == "hold":
                maximum_hold_error = np.maximum(maximum_hold_error, np.abs(qpos - rest))
            # 通用模板只缓存一个主 TCP；双臂演示还要单独检查右 TCP。
            require(
                torch.isfinite(agent.right_tcp.pose.raw_pose).all().item(),
                "右 TCP 位姿含 NaN/Inf",
            )
            step += 1
            if viewer is not None:
                env.render()
                time.sleep(
                    max(
                        0.0,
                        1 / env.unwrapped.control_freq
                        - (time.monotonic() - start_time),
                    )
                )

        print(f"完成 {step} 个控制步。逐关节目标 / 实际 / 误差：")
        errors = np.abs(qpos - target)
        for name, desired, actual, error in zip(names, target, qpos, errors):
            print(f"  {name:16s} {desired: .6f} / {actual: .6f} / {error:.6g}")
        print("左 TCP 位置:", agent.left_tcp.pose.p.cpu().numpy()[0].round(5))
        print("右 TCP 位置:", agent.right_tcp.pose.p.cpu().numpy()[0].round(5))
        print_contacts(env.unwrapped.scene)

        if args.screenshot is not None:
            from PIL import Image

            rgb = env.unwrapped.render_rgb_array()[0].cpu().numpy()
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(args.screenshot)
            print(f"截图: {args.screenshot.resolve()}")

        if viewer is not None and viewer.closed:
            print(
                "[STOP] 窗口已关闭；如需完整验收，请使用 --headless "
                "运行有限步检查。"
            )
            return

        # 先打印具名误差与接触证据，再用统一断言返回非零退出码。
        require(np.all(maximum_hold_error <= tolerance), "保持初始姿态时发生明显漂移")
        require(
            np.all(errors <= tolerance),
            "部分关节未跟踪目标；请查看上方具名误差和接触对",
        )
        if args.motion != "hold":
            changed = np.abs(target - rest) > 1e-6
            require(
                np.all(
                    np.abs(qpos[changed] - rest[changed])
                    >= 0.2 * np.abs(target[changed] - rest[changed])
                ),
                "部分目标关节没有产生可测量运动",
            )
        print(
            "[PASS] 当前模式的加载、状态和目标检查通过；尚未进行抓取测试。"
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["official", "cad"],
        default="official",
        help=(
            "默认加载 Franka 官方模型；cad 仅用于与随示例保留的旧版模型对照"
        ),
    )
    parser.add_argument(
        "--headless", action="store_true", help="有限步无窗口检查，默认不启用渲染"
    )
    parser.add_argument(
        "--steps", type=int, help="每阶段控制步数；无窗口默认 100，窗口默认持续运行"
    )
    parser.add_argument(
        "--motion",
        choices=["hold", "arms", "all"],
        default="hold",
        help="保持 / 双臂小幅运动 / 同时包含两只夹爪的运动",
    )
    parser.add_argument(
        "--control-mode",
        choices=["pd_joint_pos", "pd_joint_delta_pos"],
        default="pd_joint_pos",
    )
    parser.add_argument("--render-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="输出 PNG；即使无窗口，也需要可用的 Vulkan 渲染器",
    )
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 100 if args.headless else 0
    if args.steps < 0 or (args.headless and args.steps == 0):
        parser.error("无窗口模式 steps 必须为正数；窗口模式 0 表示持续运行")
    if args.screenshot is not None and args.screenshot.suffix.lower() != ".png":
        parser.error("--screenshot 必须使用 .png 后缀")
    try:
        run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
