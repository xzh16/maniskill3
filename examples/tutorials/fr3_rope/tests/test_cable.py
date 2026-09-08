"""FR3Cable-v1 的线缆几何、重置和 CPU 仿真回归测试。"""

import unittest

import gymnasium as gym
import numpy as np

from ..actors.rope import RopeSpec
from ..envs.cable_env import FR3CableEnv  # noqa: F401：导入模块以注册环境。


class CableEnvironmentTests(unittest.TestCase):
    def test_default_flat_state_observation(self):
        """直接 gym.make 时，默认扁平 state 观测也必须可用。"""

        env = gym.make(
            "FR3Cable-v1",
            num_envs=1,
            sim_backend="physx_cpu",
            render_backend="none",
            rope_segments=8,
        )
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.ndim, 2)
            self.assertEqual(obs.shape[0], 1)
            self.assertTrue(np.isfinite(obs.cpu().numpy()).all())
        finally:
            env.close()

    def test_rope_spec_rejects_invalid_geometry(self):
        with self.assertRaisesRegex(ValueError, "segments"):
            RopeSpec(segments=1).validate()
        with self.assertRaisesRegex(ValueError, "每段太短"):
            RopeSpec(length=0.1, radius=0.01, segments=20).validate()

    def test_reset_and_fall_on_cpu_in_both_control_modes(self):
        segments = 12
        for mode in ("pd_joint_pos", "pd_joint_delta_pos"):
            with self.subTest(control_mode=mode):
                env = gym.make(
                    "FR3Cable-v1",
                    obs_mode="state_dict",
                    reward_mode="none",
                    control_mode=mode,
                    num_envs=1,
                    sim_backend="physx_cpu",
                    render_backend="none",
                    rope_segments=segments,
                )
                try:
                    _, reset_info = env.reset(seed=0)
                    base = env.unwrapped
                    initial_positions = (
                        base.rope_segment_positions.cpu().numpy().copy()
                    )

                    # N 个实体线段之间有 N-1 处连接，每处使用 x/y/z 三个关节。
                    self.assertEqual(len(base.rope.links), 3 * segments - 2)
                    self.assertEqual(base.rope.max_dof, 3 * (segments - 1))
                    self.assertEqual(env.action_space.shape, (16,))
                    self.assertTrue(bool(reset_info["rope_state_finite"][0]))
                    np.testing.assert_allclose(base.rope.qpos.cpu(), 0, atol=1e-7)

                    target = base.agent.keyframes["rest"].qpos
                    for _ in range(80):
                        action = base.agent.action_from_qpos(target)
                        _, _, _, _, info = env.step(action)

                    final_positions = base.rope_segment_positions.cpu().numpy()
                    self.assertTrue(bool(info["rope_state_finite"][0]))
                    self.assertLess(final_positions[..., 2].mean(), 0.25)
                    self.assertGreaterEqual(
                        final_positions[..., 2].min(), base.table_top_z - 0.02
                    )

                    contact_names = [
                        tuple(body.entity.name for body in contact.bodies)
                        for contact in base.scene.get_contacts()
                        if contact.points
                    ]
                    self.assertTrue(
                        any(
                            "cable_table" in " ".join(names)
                            and "rope_" in " ".join(names)
                            for names in contact_names
                        )
                    )

                    # 走过仿真后再 reset，线缆应准确回到同一个直线初始状态。
                    env.reset(seed=0)
                    np.testing.assert_allclose(
                        base.rope_segment_positions.cpu().numpy(),
                        initial_positions,
                        atol=1e-6,
                    )
                    np.testing.assert_allclose(base.rope.qpos.cpu(), 0, atol=1e-7)
                    np.testing.assert_allclose(base.rope.qvel.cpu(), 0, atol=1e-7)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
