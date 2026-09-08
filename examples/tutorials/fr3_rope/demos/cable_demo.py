"""运行 FR3Cable-v1：保持双臂，让线缆自由落到桌面并检查稳定性。"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ..envs.cable_env import FR3CableEnv  # noqa: F401：导入时注册环境。


def print_relevant_contacts(scene):
    """打印线缆与桌面/机器人之间真正产生冲量的接触。"""

    pairs = {}
    for contact in scene.get_contacts():
        points = list(contact.points)
        if not points:
            continue
        names = tuple(body.entity.name for body in contact.bodies)
        if not any("rope_" in name for name in names):
            continue
        impulse = max(float(np.linalg.norm(point.impulse)) for point in points)
        if impulse <= 1e-8:
            continue
        pairs[names] = max(pairs.get(names, 0.0), impulse)
    for names, impulse in sorted(pairs.items()):
        print(f"  {names[0]} <-> {names[1]}: 最大点冲量 {impulse:.6g} N·s")
    return pairs


def run(args):
    render = not args.headless or args.screenshot is not None
    env = gym.make(
        "FR3Cable-v1",
        obs_mode="state_dict",
        reward_mode="none",
        control_mode=args.control_mode,
        num_envs=1,
        sim_backend="physx_cpu",
        render_backend=args.render_backend if render else "none",
        render_mode=(
            "rgb_array" if args.headless and render else ("human" if render else None)
        ),
        rope_segments=args.rope_segments,
    )
    try:
        _, info = env.reset(seed=0)
        base = env.unwrapped
        agent = base.agent
        target = agent.keyframes["rest"].qpos
        initial_positions = base.rope_segment_positions.cpu().numpy()[0]
        viewer = None
        if not args.headless:
            viewer = env.render()
            viewer.paused = False
            print("关闭窗口或按 Ctrl+C 退出。机器人保持 ready，线缆在重力下落到桌面。")

        step = 0
        last_info = info
        while (args.steps == 0 or step < args.steps) and (
            viewer is None or not viewer.closed
        ):
            start = time.monotonic()
            action = agent.action_from_qpos(target)
            if args.control_mode == "pd_joint_delta_pos":
                low = torch.as_tensor(env.action_space.low, device=agent.device)
                high = torch.as_tensor(env.action_space.high, device=agent.device)
                action = torch.maximum(torch.minimum(action, high), low)
            _, _, _, _, last_info = env.step(action)
            step += 1
            if viewer is not None:
                env.render()
                time.sleep(max(0.0, 1 / base.control_freq - (time.monotonic() - start)))

        final_positions = base.rope_segment_positions.cpu().numpy()[0]
        print(f"线缆: {args.rope_segments} 段，DOF={base.rope.max_dof}；运行 {step} 个控制步")
        print("初始两端:", initial_positions[[0, -1]].round(5))
        print("最终两端:", final_positions[[0, -1]].round(5))
        print("最终中心:", final_positions.mean(axis=0).round(5))
        for key in (
            "rope_state_finite",
            "rope_min_z",
            "rope_max_z",
            "rope_max_joint_speed",
            "rope_root_speed",
        ):
            value = last_info[key].cpu().numpy()[0]
            print(f"{key}: {value}")
        print("线缆相关接触:")
        pairs = print_relevant_contacts(base.scene)

        if args.screenshot is not None:
            from PIL import Image

            rgb = base.render_rgb_array()[0].cpu().numpy()
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(args.screenshot)
            print(f"截图: {args.screenshot.resolve()}")

        if not bool(last_info["rope_state_finite"][0]):
            raise RuntimeError("线缆状态出现 NaN/Inf")
        if final_positions[:, 2].min() < base.table_top_z - 0.02:
            raise RuntimeError("线缆穿过或掉下桌面")
        if args.steps and not pairs:
            raise RuntimeError("没有检测到线缆与桌面/机器人接触")
        print("[PASS] 线缆生成、自由落体、桌面接触和有限状态检查通过。")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无窗口有限步检查")
    parser.add_argument(
        "--steps", type=int, help="控制步数；无窗口默认 200，窗口默认持续运行"
    )
    parser.add_argument("--rope-segments", type=int, default=20)
    parser.add_argument(
        "--control-mode",
        choices=["pd_joint_pos", "pd_joint_delta_pos"],
        default="pd_joint_pos",
    )
    parser.add_argument("--render-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 200 if args.headless else 0
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
