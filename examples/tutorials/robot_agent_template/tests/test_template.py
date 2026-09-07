"""使用标准库 unittest；无需新安装测试依赖。"""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from xml.etree import ElementTree

import gymnasium as gym
import numpy as np
import sapien

from mani_skill.agents.registration import register_agent

from ..agent import RobotAgentTemplate
from ..check import initialize_rest, load_agent, main, run_simulation_checks
from ..validation import ConfigurationError, resolve_asset, validate_definition


@register_agent()
class TwoJointRobot(RobotAgentTemplate):
    """最小无夹爪机器人：同时包含一个平移关节和一个旋转关节。"""

    uid = "template_test_two_joint"
    urdf_path = str(Path(__file__).parent / "assets" / "two_joint.urdf")
    arm_joint_names = ["slide", "hinge"]  # 故意与 SAPIEN 关节顺序不同。
    rest_qpos = {"slide": 0.04, "hinge": 0.1}
    rest_pose = sapien.Pose([0, 0, 0.3])
    ee_link_name = "tcp"


@register_agent()
class IndependentGripperRobot(TwoJointRobot):
    """把 slide 重新分到夹爪组，覆盖“非 mimic 独立夹爪”的分支。"""

    uid = "template_test_independent_gripper"
    arm_joint_names = ["hinge"]
    gripper_joint_names = ["slide"]
    finger_link_names = ["tool"]


class ConfigurationTests(unittest.TestCase):
    """不启动物理引擎，验证错误配置能否尽早给出明确原因。"""

    def variant(self, **changes):
        # 临时派生类只覆盖本测试关心的字段，避免重复整套正确配置。
        return type("ChangedRobot", (TwoJointRobot,), changes)

    def test_valid_non_panda_definition(self):
        joints = validate_definition(TwoJointRobot)
        self.assertEqual(set(joints), {"hinge", "slide"})

    def test_invalid_configuration_reports_reason(self):
        cases = [
            ({"urdf_path": "/not-a-real-robot.urdf"}, "URDF 不存在"),
            ({"urdf_path": "robot.xacro"}, "先展开 Xacro"),
            ({"arm_joint_names": []}, "arm_joint_names"),
            ({"arm_joint_names": ["hinge", "hinge", "slide"]}, "重复分配"),
            ({"arm_joint_names": ["hinge"]}, "未分配控制器"),
            ({"arm_joint_names": ["hinge", "typo"]}, "不存在的关节"),
            ({"rest_qpos": {"hinge": 0}}, "rest_qpos 名称不匹配"),
            ({"rest_qpos": {"hinge": 4, "slide": 0.04}}, "超过关节限位"),
            ({"rest_qpos": {"hinge": np.nan, "slide": 0.04}}, "NaN/Inf"),
            ({"ee_link_name": "missing_tcp"}, "找不到 TCP"),
            ({"finger_link_names": ["missing_finger"]}, "手指 link"),
            ({"arm_stiffness": [1, 2, 3]}, "长度为 2"),
            ({"arm_force_limit": 0}, "大于 0"),
            ({"arm_delta_limit": np.inf}, "NaN/Inf"),
            ({"fix_root_link": False}, "固定底座"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ConfigurationError, message
            ):
                validate_definition(self.variant(**changes))

    def test_static_checks_do_not_create_scene(self):
        from unittest.mock import patch

        with patch("gymnasium.make", side_effect=AssertionError("不应启动仿真")), patch(
            "sys.argv",
            ["check", "--agent", f"{__name__}:TwoJointRobot", "--static-only"],
        ):
            self.assertEqual(main(), 0)

    def test_unregistered_class_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "未注册"):
            load_agent(f"{RobotAgentTemplate.__module__}:RobotAgentTemplate")

    def test_mesh_missing_and_package_path(self):
        tree = ElementTree.parse(TwoJointRobot.urdf_path)
        geometry = tree.find("link/visual/geometry")
        ElementTree.SubElement(geometry, "mesh", filename="missing.glb")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            tree.write(path)
            with self.assertRaisesRegex(ConfigurationError, "资产不存在"):
                validate_definition(self.variant(urdf_path=str(path)))
            package = Path(directory) / "robot_description"
            package.mkdir()
            mesh = package / "part.glb"
            mesh.touch()
            self.assertEqual(
                resolve_asset("package://robot_description/part.glb", package),
                mesh.resolve(),
            )

    def test_mimic_validation(self):
        # 给两关节测试模型添加第三个关节，使用负比例和非零偏移验证通用关系。
        tree = ElementTree.parse(TwoJointRobot.urdf_path)
        root = tree.getroot()
        source = root.find("joint[@name='slide']")
        follower = deepcopy(source)
        follower.set("name", "follower")
        follower.find("child").set("link", "finger")
        follower.find("limit").set("lower", "-0.18")
        follower.find("limit").set("upper", "0.02")
        ElementTree.SubElement(
            follower, "mimic", joint="slide", multiplier="-1", offset="0.02"
        )
        root.append(follower)
        ElementTree.SubElement(root, "link", name="finger")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            tree.write(path)
            cls = self.variant(
                urdf_path=str(path),
                arm_joint_names=["hinge"],
                gripper_joint_names=["follower", "slide"],
                rest_qpos={"hinge": 0.1, "slide": 0.04, "follower": -0.02},
                gripper_mimic={
                    "follower": {"joint": "slide", "multiplier": -1, "offset": 0.02}
                },
            )
            validate_definition(cls)
            cls.gripper_mimic = {"follower": {"joint": "follower"}}
            with self.assertRaisesRegex(ConfigurationError, "自跟随"):
                validate_definition(cls)
            cls.gripper_mimic = {}
            with self.assertRaisesRegex(ConfigurationError, "声明了 mimic"):
                validate_definition(cls)

    def test_failure_exit_code(self):
        from unittest.mock import patch

        output = StringIO()
        with redirect_stdout(output), patch(
            "sys.argv", ["check", "--agent", "no_such_robot_module:Missing"]
        ):
            self.assertEqual(main(), 1)
        self.assertIn("[FAIL]", output.getvalue())

    def test_copied_template_can_use_original_checker(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "copied_robot_template"
            shutil.copytree(
                Path(__file__).parents[1],
                package,
                ignore=shutil.ignore_patterns("tests", "__pycache__"),
            )
            # 测试中写入一份用户配置，验证复制后的相对导入和类型身份不会阻断检查。
            (package / "configured.py").write_text(
                "from .agent import RobotAgentTemplate\n"
                "from mani_skill.agents.registration import register_agent\n"
                "@register_agent()\n"
                "class CopiedRobot(RobotAgentTemplate):\n"
                "    uid = 'template_test_copied'\n"
                f"    urdf_path = {TwoJointRobot.urdf_path!r}\n"
                "    arm_joint_names = ['hinge', 'slide']\n"
                "    rest_qpos = {'hinge': 0.1, 'slide': 0.04}\n"
                "    ee_link_name = 'tcp'\n",
                encoding="utf-8",
            )
            with patch.object(sys, "path", [directory, *sys.path]):
                cls = load_agent("copied_robot_template.configured:CopiedRobot")
                self.assertFalse(issubclass(cls, RobotAgentTemplate))
                validate_definition(cls)


class SimulationTests(unittest.TestCase):
    """真正创建 CPU PhysX 环境，验证加载、动作映射、保持和小幅运动。"""

    def test_no_gripper_and_independent_gripper_motion(self):
        for cls in (TwoJointRobot, IndependentGripperRobot):
            with self.subTest(robot=cls.uid):
                run_simulation_checks(cls, steps=60, motion=True)

    def test_qpos_to_action_uses_actual_joint_indices(self):
        env = gym.make(
            "Empty-v1",
            robot_uids=TwoJointRobot.uid,
            control_mode="pd_joint_pos",
            obs_mode="state_dict",
            reward_mode="none",
            sim_backend="physx_cpu",
            render_backend="none",
        )
        try:
            rest = initialize_rest(env)
            agent = env.unwrapped.agent
            names = [joint.name for joint in agent.robot.active_joints]
            np.testing.assert_allclose(
                rest, [TwoJointRobot.rest_qpos[name] for name in names]
            )
            np.testing.assert_allclose(
                agent.action_from_qpos(rest).cpu().numpy()[0], [0.04, 0.1]
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
