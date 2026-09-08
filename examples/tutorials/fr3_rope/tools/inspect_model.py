"""只读检查官方 FR3 Duo；可用 ``--model cad`` 对照旧 CAD 导出模型。"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from defusedxml import ElementTree
from scipy.spatial.transform import Rotation

from ..robot import FR3DualArm, FR3DualArmCAD
from ..validation import require, resolve_asset, validate_definition


def _joint_axis_in_parent(joint):
    """把 joint 局部轴旋转到 parent link 坐标系，便于比较两根手指。"""

    origin = joint.find("origin")
    rpy = np.fromstring(
        "0 0 0" if origin is None else origin.attrib.get("rpy", "0 0 0"), sep=" "
    )
    axis_node = joint.find("axis")
    axis = np.fromstring(
        "1 0 0" if axis_node is None else axis_node.attrib.get("xyz", "1 0 0"),
        sep=" ",
    )
    require(axis.shape == (3,) and np.linalg.norm(axis) > 1e-6, "手指关节轴无效")
    return Rotation.from_euler("xyz", rpy).apply(axis / np.linalg.norm(axis))


def inspect_model(cls):
    joints = validate_definition(cls)
    path = Path(cls.urdf_path)
    root = ElementTree.parse(path).getroot()
    links = root.findall("link")
    definitions = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    mesh_paths = {
        resolve_asset(mesh.attrib["filename"], path.parent) for mesh in root.iter("mesh")
    }
    counts = Counter(joint.attrib["type"] for joint in root.findall("joint"))
    print(
        f"[PASS] {cls.__name__}: {len(links)} links；{len(joints)} 活动关节；"
        f"{len(mesh_paths)} 个 mesh 均存在"
    )
    print(f"关节类型: {dict(counts)}")

    # 纯坐标 frame 可以没有 inertial；只检查真正携带惯量的刚体 link。
    inertial_count = 0
    for link in links:
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertial_count += 1
        name = link.attrib["name"]
        mass = float(inertial.find("mass").attrib["value"])
        values = inertial.find("inertia").attrib
        matrix = np.array(
            [
                [float(values["ixx"]), float(values["ixy"]), float(values["ixz"])],
                [float(values["ixy"]), float(values["iyy"]), float(values["iyz"])],
                [float(values["ixz"]), float(values["iyz"]), float(values["izz"])],
            ]
        )
        require(np.isfinite(mass) and mass > 0, f"{name}: 质量无效")
        eigenvalues = np.linalg.eigvalsh(matrix)
        require(np.isfinite(matrix).all() and (eigenvalues > 0).all(), f"{name}: 惯量无效")
        require(
            eigenvalues[-1] <= sum(eigenvalues[:2]) + 1e-8,
            f"{name}: 主惯量不满足三角不等式",
        )
    print(f"[PASS] {inertial_count} 个物理 link 的质量/惯量基本数学检查通过")

    zero_effort, zero_velocity = [], []
    for name in cls.arm_joint_names + cls.gripper_joint_names:
        limit = definitions[name].find("limit")
        if float(limit.attrib.get("effort", "0")) <= 0:
            zero_effort.append(name)
        if float(limit.attrib.get("velocity", "0")) <= 0:
            zero_velocity.append(name)
    if zero_effort or zero_velocity:
        print(
            f"[WARN] effort<=0: {len(zero_effort)}；velocity<=0: {len(zero_velocity)}"
        )
    else:
        print("[PASS] 所有活动关节均提供正的 effort 和 velocity 上限")

    for side, pair in (
        ("左手", cls.left_gripper_joint_names),
        ("右手", cls.right_gripper_joint_names),
    ):
        axes = [_joint_axis_in_parent(definitions[name]) for name in pair]
        cosine = float(np.dot(*axes))
        print(f"[INFO] {side}两手指在共同手掌坐标系中的轴点积: {cosine:.6f}")

    tcp_names = sorted(
        link.attrib["name"] for link in links if "tcp" in link.attrib["name"].lower()
    )
    print(f"TCP links: {tcp_names or '无'}")
    print(f"mimic: {cls.gripper_mimic or '无；每个手指独立控制'}")

    srdf_path = path.with_suffix(".srdf")
    if srdf_path.is_file():
        srdf = ElementTree.parse(srdf_path).getroot()
        reasons = Counter(
            node.attrib.get("reason", "") for node in srdf.findall("disable_collisions")
        )
        print(f"SRDF 禁用碰撞对: {sum(reasons.values())}；reason 分布: {dict(reasons)}")
        print(
            "[WARN] SAPIEN 3.0.3 只识别 reason=Default，"
            "当前 Agent 因此暂时关闭内部碰撞。"
        )
    return joints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["official", "cad"], default="official")
    args = parser.parse_args()
    cls = FR3DualArm if args.model == "official" else FR3DualArmCAD
    inspect_model(cls)


if __name__ == "__main__":
    main()
