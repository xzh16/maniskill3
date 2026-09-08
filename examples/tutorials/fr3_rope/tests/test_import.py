"""验证官方 FR3 Duo 的配置、qpos/action 映射和 CPU 运动。"""

import unittest

import numpy as np

from ..check import (
    advance,
    initialize_rest,
    motion_target,
    tolerances,
)
from ..demos.robot_demo import make_env
from ..robot import FR3DualArm, FR3DualArmCAD
from ..validation import validate_definition


class FR3ImportTests(unittest.TestCase):
    def test_official_assets_and_configuration(self):
        joints = validate_definition(FR3DualArm)
        self.assertEqual(len(joints), 18)
        self.assertEqual(sum(j["kind"] == "revolute" for j in joints.values()), 14)
        self.assertEqual(sum(j["kind"] == "prismatic" for j in joints.values()), 4)
        self.assertTrue(FR3DualArm.disable_self_collisions)
        self.assertEqual(len(FR3DualArm.gripper_mimic), 2)
        self.assertTrue(FR3DualArm.urdf_path.endswith("fr3_duo_franka_hand.urdf"))

        # 旧模型没有删除，但已经换用不同 uid，只作为问题对照。
        legacy_joints = validate_definition(FR3DualArmCAD)
        self.assertEqual(len(legacy_joints), 18)
        self.assertNotEqual(FR3DualArm.uid, FR3DualArmCAD.uid)

    def test_cpu_both_modes_and_named_action_mapping(self):
        for mode in ("pd_joint_pos", "pd_joint_delta_pos"):
            with self.subTest(mode=mode):
                env = make_env(robot_cls=FR3DualArm, control_mode=mode)
                try:
                    rest = initialize_rest(env)
                    agent = env.unwrapped.agent
                    self.assertIs(type(agent), FR3DualArm)
                    self.assertEqual(len(agent.robot.links), 59)
                    self.assertEqual(env.action_space.shape, (16,))
                    self.assertEqual(agent.left_tcp.name, "left_fr3v2_hand_tcp")
                    self.assertEqual(agent.right_tcp.name, "right_fr3v2_hand_tcp")

                    names = [joint.name for joint in agent.robot.active_joints]
                    index = {name: i for i, name in enumerate(names)}
                    tolerance = tolerances(agent, 0.02, 0.002)
                    advance(env, rest, 100, tolerance, hold=True)
                    target = motion_target(agent, rest)

                    # controller 动作按“左臂、右臂、左夹爪、右夹爪”排列；
                    # 两个 mimic follower 留在 qpos 中，但不单独占 action 维度。
                    current = agent.robot.qpos.cpu().numpy()[0]
                    expected = []
                    for name in agent.arm_joint_names:
                        value = target[index[name]]
                        if mode == "pd_joint_delta_pos":
                            value -= current[index[name]]
                        expected.append(value)
                    expected.extend(
                        target[index[name]]
                        for name in (
                            "left_fr3v2_finger_joint1",
                            "right_fr3v2_finger_joint1",
                        )
                    )
                    np.testing.assert_allclose(
                        agent.action_from_qpos(target).cpu().numpy()[0],
                        expected,
                        atol=1e-7,
                    )

                    qpos = advance(env, target, 100, tolerance)
                    self.assertTrue(np.all(np.abs(qpos - target) <= tolerance))
                    self.assertAlmostEqual(
                        qpos[index["left_fr3v2_finger_joint1"]],
                        qpos[index["left_fr3v2_finger_joint2"]],
                        places=6,
                    )
                    self.assertAlmostEqual(
                        qpos[index["right_fr3v2_finger_joint1"]],
                        qpos[index["right_fr3v2_finger_joint2"]],
                        places=6,
                    )
                    initialize_rest(env)
                    np.testing.assert_allclose(
                        agent.robot.qpos.cpu().numpy()[0], rest, atol=1e-7
                    )
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
