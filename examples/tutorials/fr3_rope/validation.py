"""创建物理场景之前检查模型与配置，尽早给出可读的错误。

这里做的是“静态检查”：只读取 Python 配置和 URDF/XML，不启动 SAPIEN。
它能快速发现拼错的名称、漏配关节、错误限位和丢失资产；真正加载后的关节顺序、
控制效果和数值稳定性则由 check.py 的仿真检查负责。
"""

from collections import Counter
from pathlib import Path

import numpy as np
from defusedxml import ElementTree

from mani_skill import format_path


class ConfigurationError(ValueError):
    """机器人配置或模型不符合本模板的约定。"""


def require(condition, message):
    """统一所有配置断言的异常类型，让命令行只需处理一种可读错误。"""

    if not condition:
        raise ConfigurationError(message)


def numeric_array(value, count, label, *, positive=False):
    """接受一个标量或按关节顺序填写的数组，检查维度和数值。"""

    # np.broadcast_to 让 stiffness=100 与 stiffness=[100, 100, ...] 都合法。
    try:
        array = np.broadcast_to(np.asarray(value, dtype=float), (count,)).copy()
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} 必须是标量或长度为 {count} 的数组") from exc
    require(np.isfinite(array).all(), f"{label} 含 NaN/Inf")
    if positive:
        require((array > 0).all(), f"{label} 必须大于 0")
    return array


def resolve_asset(filename, urdf_dir):
    """按当前 SAPIEN loader 规则检查相对路径、绝对路径和 package:// 引用。"""

    require(bool(filename), "mesh/texture 缺少 filename")
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
        candidates = [parent / filename for parent in (urdf_dir, *urdf_dir.parents)]
    else:
        candidates = [urdf_dir / filename]
    # SAPIEN 找不到以上路径时会回退到当前工作目录。
    candidates.append(Path(filename))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise ConfigurationError(f"资产不存在: {filename}（URDF 目录: {urdf_dir}）")


def validate_definition(cls):
    """检查一个 RobotAgentTemplate 子类，返回活动关节的类型/限位资料。

    第一版支持固定底座、URDF、单自由度关节，以及可选的独立/mimic 夹爪。
    不根据特定型号的自由度或名称作任何假设。
    """
    # ---------- 1. Agent 身份和主模型文件 ----------
    # cls 是 robot.py 中的具体类。这里只读取类属性，不需要实例化仿真对象。
    require(isinstance(cls.uid, str) and bool(cls.uid.strip()), "请填写非空 uid")
    require(cls.fix_root_link is True, "第一版模板只支持固定底座机械臂")
    path = Path(format_path(str(cls.urdf_path))).expanduser().resolve()
    require(path.suffix.lower() == ".urdf", "urdf_path 必须指向 .urdf；请先展开 Xacro")
    require(path.is_file(), f"URDF 不存在: {path}；请填写 robot.py 中的 urdf_path")
    try:
        root = ElementTree.parse(path).getroot()
    except Exception as exc:
        raise ConfigurationError(f"无法解析 URDF: {exc}") from exc
    require(root.tag == "robot", "URDF 根元素应为 <robot>")
    require(not any("xacro" in node.tag for node in root.iter()), "请先将 Xacro 展开为 URDF")

    # ---------- 2. URDF 显式引用的外部资产 ----------
    # 视觉和碰撞 mesh 都检查；纹理仅检查 URDF 显式引用，不解析 mesh 内部依赖。
    for element in [*root.iter("mesh"), *root.iter("texture")]:
        resolve_asset(element.get("filename", ""), path.parent)

    # ---------- 3. link、TCP 和手指名称 ----------
    link_names = [node.get("name") for node in root.findall("link")]
    require(all(link_names) and len(set(link_names)) == len(link_names), "link 名称缺失或重复")
    required_links = [cls.ee_link_name, *cls.finger_link_names]
    require(bool(cls.ee_link_name), "请填写 ee_link_name（TCP link）")
    missing_links = set(required_links) - set(link_names)
    require(not missing_links, f"找不到 TCP/手指 link: {sorted(missing_links)}")
    require(
        len(set(cls.finger_link_names)) == len(cls.finger_link_names), "手指 link 名称重复"
    )

    # ---------- 4. 解析关节，并检查 URDF 是否是一棵连通树 ----------
    # joints 只保存会进入 qpos 的活动关节；fixed 关节只参与连接关系检查。
    joints = {}
    all_joint_names = set()
    children = {}
    for joint in root.findall("joint"):
        name, kind = joint.get("name"), joint.get("type")
        require(name and name not in all_joint_names, f"joint 名称缺失或重复: {name}")
        all_joint_names.add(name)
        parent, child = joint.find("parent"), joint.find("child")
        require(parent is not None and child is not None, f"{name} 缺少 parent/child")
        parent_name, child_name = parent.get("link"), child.get("link")
        require(
            parent_name in link_names and child_name in link_names,
            f"{name} 引用不存在的 link",
        )
        require(child_name not in children, f"link {child_name} 存在多个父关节")
        children[child_name] = parent_name
        require(
            kind in ("fixed", "revolute", "continuous", "prismatic"),
            f"{name}: 不支持 {kind} 类型；模板只处理单自由度活动关节",
        )
        if kind == "fixed":
            continue
        if kind == "continuous":
            # continuous 旋转关节没有位置上下限，但后续仍会检查有限的 rest 值。
            low, high = -np.inf, np.inf
        else:
            limit = joint.find("limit")
            require(limit is not None, f"{name} 缺少 limit")
            try:
                low, high = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            except (KeyError, ValueError) as exc:
                raise ConfigurationError(f"{name} 的 lower/upper 无效") from exc
            require(np.isfinite([low, high]).all() and low < high, f"{name} 限位无效")
        joints[name] = dict(kind=kind, lower=low, upper=high, mimic=joint.find("mimic"))

    # BaseAgent 默认只加载第一个 articulation，因此提前拒绝不连通/成环模型。
    roots = set(link_names) - set(children)
    require(len(roots) == 1, "URDF 应是一棵连通树，且只有一个根 link")
    for link in link_names:
        seen = set()
        while link in children:
            require(link not in seen, "URDF 关节连接成环")
            seen.add(link)
            link = children[link]
        require(link in roots, "URDF 存在不连通的 link")

    # ---------- 5. 控制器覆盖与初始状态 ----------
    # 每个活动关节必须恰好属于 arm 或 gripper；被漏掉的活动关节无人控制，
    # 被重复填写的关节则会收到两个互相冲突的控制目标。
    names = [*cls.arm_joint_names, *cls.gripper_joint_names]
    require(bool(cls.arm_joint_names), "请填写 arm_joint_names")
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    require(not duplicates, f"关节被重复分配给控制器: {duplicates}")
    require(
        not set(names) - set(joints),
        f"配置中存在非活动/不存在的关节: {sorted(set(names) - set(joints))}",
    )
    require(
        not set(joints) - set(names), f"有活动关节未分配控制器: {sorted(set(joints) - set(names))}"
    )
    require(isinstance(cls.rest_qpos, dict), "rest_qpos 应按 {关节名称: 初始位置} 填写")
    require(
        set(cls.rest_qpos) == set(joints),
        f"rest_qpos 名称不匹配：缺少 {sorted(set(joints) - set(cls.rest_qpos))}，"
        f"多余 {sorted(set(cls.rest_qpos) - set(joints))}",
    )
    for name, joint in joints.items():
        # 初始位置先按 XML 限位检查；模型加载后还会按 SAPIEN 的实际限位复查。
        value = numeric_array(cls.rest_qpos[name], 1, f"rest_qpos[{name}]")[0]
        require(
            joint["lower"] - 1e-6 <= value <= joint["upper"] + 1e-6,
            f"rest_qpos[{name}]={value} 超过关节限位",
        )
    pose = cls.rest_pose
    require(
        np.isfinite(pose.p).all() and np.isfinite(pose.q).all(), "rest_pose 含 NaN/Inf"
    )
    require(np.isclose(np.linalg.norm(pose.q), 1.0, atol=1e-4), "rest_pose 四元数应为单位四元数")

    # ---------- 6. PD 参数、夹爪 mimic 与控制范围 ----------
    # 标量参数会广播到整组关节；数组则必须严格等于该组的关节数量。
    for group, group_names in (
        ("arm", cls.arm_joint_names),
        ("gripper", cls.gripper_joint_names),
    ):
        if not group_names:
            continue
        for field in ("stiffness", "damping", "force_limit"):
            numeric_array(
                getattr(cls, f"{group}_{field}"),
                len(group_names),
                f"{group}_{field}",
                positive=True,
            )
    numeric_array(
        cls.arm_delta_limit, len(cls.arm_joint_names), "arm_delta_limit", positive=True
    )

    # 只支持一层跟随：q_follower = multiplier * q_source + offset。
    # 例如双指对称运动可能是 q_right = -q_left + offset。
    mimic = cls.gripper_mimic
    require(isinstance(mimic, dict), "gripper_mimic 应是字典")
    followers, sources = set(mimic), set()
    for name, config in mimic.items():
        require(
            isinstance(config, dict) and "joint" in config, f"{name} mimic 缺少 joint"
        )
        source = config["joint"]
        sources.add(source)
        require(
            name in cls.gripper_joint_names and source in cls.gripper_joint_names,
            f"mimic {name} -> {source} 必须都是夹爪关节",
        )
        require(source not in followers, "不支持 mimic 自跟随、链式跟随或循环跟随")
        multiplier = numeric_array(
            config.get("multiplier", 1.0), 1, "mimic multiplier"
        )[0]
        offset = numeric_array(config.get("offset", 0.0), 1, "mimic offset")[0]
        expected = cls.rest_qpos[source] * multiplier + offset
        require(
            np.isclose(cls.rest_qpos[name], expected, atol=1e-6),
            f"rest_qpos[{name}] 不满足 mimic 关系",
        )
    if mimic:
        # 一个同步夹爪可以有多个 follower，但不能混入完全无关的独立关节。
        require(
            followers | sources == set(cls.gripper_joint_names), "mimic 模式有夹爪关节未参与跟随关系"
        )

    # 防止漏掉模型中原有的 mimic，或在 Agent 中填写相矛盾的比例/偏移。
    for name, joint in joints.items():
        declaration = joint["mimic"]
        if declaration is not None:
            require(name in mimic, f"URDF 中 {name} 声明了 mimic，请在 gripper_mimic 中配置")
            config = mimic[name]
            require(
                config["joint"] == declaration.get("joint"), f"{name} mimic 来源与 URDF 不同"
            )
            for field, default in (("multiplier", 1.0), ("offset", 0.0)):
                require(
                    np.isclose(
                        config.get(field, default),
                        float(declaration.get(field, default)),
                    ),
                    f"{name} mimic {field} 与 URDF 不同",
                )

    # lower/upper 可缩小夹爪控制范围，但必须连同 follower 一起落在 URDF 限位内。
    count = len(cls.gripper_joint_names)
    if count:
        low = np.array([joints[n]["lower"] for n in cls.gripper_joint_names])
        high = np.array([joints[n]["upper"] for n in cls.gripper_joint_names])
        if cls.gripper_lower is not None:
            low = numeric_array(cls.gripper_lower, count, "gripper_lower")
        if cls.gripper_upper is not None:
            high = numeric_array(cls.gripper_upper, count, "gripper_upper")
        bounds = dict(zip(cls.gripper_joint_names, zip(low, high)))
        for name in cls.gripper_joint_names:
            lower, upper = bounds[name]
            if name in mimic:
                config = mimic[name]
                ends = np.array(bounds[config["joint"]]) * config.get(
                    "multiplier", 1.0
                ) + config.get("offset", 0.0)
                lower, upper = ends.min(), ends.max()
            require(
                np.isfinite([lower, upper]).all() and lower <= upper, f"{name} 夹爪范围无效"
            )
            require(
                lower >= joints[name]["lower"] - 1e-6
                and upper <= joints[name]["upper"] + 1e-6,
                f"{name} 夹爪控制范围或 mimic 目标超过 URDF 限位",
            )
            require(
                lower - 1e-6 <= cls.rest_qpos[name] <= upper + 1e-6,
                f"{name} 初始位置超出夹爪控制范围",
            )
    # Agent 初始化后还会使用这些资料区分旋转/平移关节，并设置检查容差。
    return joints
