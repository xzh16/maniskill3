# 自定义机械臂 Agent 标准模板

适用于固定底座机械臂，支持任意数量的单自由度手臂关节、无夹爪、独立控制夹爪，
或一层 mimic 同步夹爪。机器人模型使用 URDF。无需修改 ManiSkill 核心源码。

## 文件与职责

| 文件 | 职责 | 接入机器人时 |
| --- | --- | --- |
| `robot.py` | 填写模型、关节、初始姿态、TCP、控制参数，并注册 Agent | 首先修改 |
| `agent.py` | 通用加载、控制器、按名称生成 keyframe、动作转换 | 一般复用 |
| `validation.py` | 检查文件、关节覆盖、限位、mimic 等 | 一般复用 |
| `check.py` | 命令行检查、CPU 加载、姿态保持和可选运动 | 统一验收入口 |
| `assets/` | 放 URDF 和 mesh，也可使用其他模型目录 | 填入实际资产 |
| `example_panda.py` | 引用之前 `custom_agent` 的已验证配置验收模板 | 可选示例 |
| `tests/` | 错误配置回归检查，以及非 Panda 模型验证 | 模板维护时运行 |

`agent.py`、`robot.py`、`validation.py` 没有 Panda 名称或固定 7 关节/8 维动作假设。
`example_panda.py` 是可选验收案例，只有指定 `--example` 才会导入。

## 先验证模板

在仓库根目录执行：

```bash
python -m examples.tutorials.robot_agent_template.check --example --motion
```

这会创建名为 `template_panda` 的 Agent，分别检查 `pd_joint_pos` 和
`pd_joint_delta_pos`。每种模式先保持 rest 100 个控制步，再执行 100 步小幅运动。
成功输出 `[PASS]`；任何检查失败输出 `[FAIL]` 并返回非零退出码。

## 填写自己的机器人

1. 复制整个模板目录，或先直接填写 `robot.py`，放入 URDF 与完整 mesh。
2. 设置唯一 `uid` 和 URDF 路径。相对资产路径建议以 `robot.py` 所在目录为基准。
3. 填 `arm_joint_names`、`gripper_joint_names`、`ee_link_name` 和可选手指 link。
4. 填 `rest_qpos`，用 **关节名到位置的字典**，覆盖模型的所有活动关节。
   加载时模板按 SAPIEN 实际顺序自动生成 1 维 `keyframes["rest"].qpos`。
5. 填 `rest_pose`、PD 参数和夹爪范围。参数是仿真调试起点，需要按模型验证。
6. 有同步夹爪时填写 `gripper_mimic`；无夹爪时保持夹爪列表与字典为空。

不要把夹爪开口宽度直接当作每根手指的关节位置；以 URDF 定义的单位和坐标为准。

先做静态检查，再加载，再测试运动：

```bash
python -m examples.tutorials.robot_agent_template.check --static-only
python -m examples.tutorials.robot_agent_template.check
python -m examples.tutorials.robot_agent_template.check --motion
```

未填写配置时，这些命令应报告 URDF 不存在等错误，不会自动拿 Panda 代替目标机器人。
如果复制到其他目录或更改类名，检查入口支持显式指定配置：

```bash
python -m examples.tutorials.robot_agent_template.check --agent your_package.robot:YourRobot --motion
```

运行时 Python 必须能导入该模块。复制后核心相对导入仍可使用；可选
`example_panda.py` 依赖旧示例位置，若移动它需调整示例导入或不再使用 `--example`。

## 控制约定

两种模式均关闭归一化。旋转关节使用 rad，平移关节使用 m。

| 模式 | 手臂动作 | 夹爪动作 |
| --- | --- | --- |
| `pd_joint_pos` | 目标关节位置 | 目标关节位置 |
| `pd_joint_delta_pos` | 相对当前关节位置的增量，受 `arm_delta_limit` 限制 | 目标关节位置 |

动作顺序为手臂在前、夹爪在后。mimic follower 不独占动作维度。
`agent.action_from_qpos(target_qpos)` 通过实际关节索引生成当前模式的批量动作，
可处理配置顺序与模型顺序不同的情况。增量过大时调用方需分步或裁剪；检查脚本已处理。

没有夹爪也可运行。多个夹爪关节不使用 mimic 时各占一维；使用 mimic 时每个夹爪
关节必须是 follower 或直接 source，不支持循环/链式 mimic。

## 自动检查的范围

静态检查：URDF 存在且可解析、XML 中的 mesh/texture 路径、关节树连接、活动关节
无遗漏/重复、关节参数维度、rest 完整性与限位、TCP/手指 link、mimic 一致性。
加载检查：注册对象正确、SAPIEN 实际关节与配置匹配、keyframe 和动作维度。
运行检查：两种控制模式保持姿态；每步状态有限且关节不明显越界；可选目标跟踪。

默认角度误差容差是 0.02 rad，平移关节是 0.002 m，可通过
`--angle-tolerance` 和 `--position-tolerance` 修改；`--steps` 控制每阶段长度。
保持姿态检查会监测全部步骤，运动检查关注每步状态及最终目标误差。

CPU 无窗口检查通过不代表 GPU、视觉效果、自碰撞或抓取已经验证。静态资产检查
只检查 URDF 的显式引用，不解析 mesh 内部的附属文件；实际加载可补充发现部分错误。
可视化碰撞形状、接触稳定性和真实 FR3 参数仍需要接入模型后验证。

第一版不包含 MJCF、浮动底座、被动活动关节、末端 IK、传感器或线缆任务。
`urdf_config` 等 BaseAgent 扩展接口仍可在子类补充。`is_grasping/is_static`
尚未实现，调用基类接口会报 `NotImplementedError`。

## 回归检查

```bash
python -m unittest examples.tutorials.robot_agent_template.tests.test_template -v
```

包含一个不使用外部 mesh 的小型两关节模型，用于确认模板不依赖 Panda。
