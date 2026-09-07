# 最小自定义 Agent 导入示例

本示例直接继承 `BaseAgent`，参照内置 Panda 配置一个独立的 `CustomPanda`，
注册名称是 `custom_panda`。使用仓库已有的 Panda URDF 和 mesh，因此窗口中的
机器人仍是 Panda；本示例验证自定义 Agent 导入流程，还没有接入 FR3 模型。

## 文件与阅读顺序

1. [custom_panda.py](custom_panda.py)：注册名称、URDF、关节名称、rest 姿态、控制器和 TCP。
2. [demo.py](demo.py)：导入 Agent 模块，通过 `gym.make` 加载，设置初始姿态并发送动作。

调用顺序是：导入 `custom_panda` 模块 → 执行 `@register_agent()` → 注册
`custom_panda` → 环境通过 `robot_uids` 查找类 → `BaseAgent` 加载模型与控制器。
只把 Python 文件放进某个目录，不会自动注册。这里选择在启动程序显式导入，
因此不需要修改 ManiSkill 内置机器人的 `__init__.py`。

## 运行

以下命令在仓库根目录执行，使用已经安装 ManiSkill 的 conda 环境：

```bash
conda activate maniskill3
cd /home/xzh/ManiSkill/ManiSkill
```

先运行无窗口检查：

```bash
python -m examples.tutorials.custom_agent.demo --headless
```

默认执行 100 个控制步，打印注册名称、URDF 路径、活动关节、动作空间和 TCP。
无窗口模式关闭渲染，且物理计算在 CPU 上进行。

打开窗口，保持 rest 姿态：

```bash
python -m examples.tutorials.custom_agent.demo
```

让第一个手臂关节小幅摆动，同时让夹爪开合：

```bash
python -m examples.tutorials.custom_agent.demo --move
```

窗口模式持续运行到关闭窗口或 Ctrl+C。本入口会取消暂停；如果手动暂停过，
可在 Viewer 的 Control 面板中取消 Pause。GUI 需要可用的渲染设备/驱动，
没有窗口或渲染支持时使用 `--headless`。

也可以无窗口执行运动：

```bash
python -m examples.tutorials.custom_agent.demo --headless --move --steps 100
```

使用上面的 `python -m ...` 命令，以便 Python 正确解析示例的包内导入。
单独运行原来的 `mani_skill.examples.demo_robot -r custom_panda` 不会导入这个
示例模块，因此不会自动注册 `custom_panda`；请使用这里的启动入口。

## 动作和关节位置

本例只有 `pd_joint_pos` 一种模式，手臂与夹爪均关闭动作归一化：

| 数据 | 维度 | 含义 |
| --- | --- | --- |
| `qpos` | 9 | 7 个手臂关节角 + 2 个手指位置 |
| `action[:7]` | 7 | 手臂目标关节角，单位 rad |
| `action[7]` | 1 | 每根手指的目标位置，单位 m，范围 0～0.04 |

第二个手指由 Mimic 控制器跟随第一个，所以动作总共是 8 维。`action[7]=0.04`
表示打开目标，`0` 表示闭合目标；这与内置 Panda 的归一化夹爪动作不同。
不要直接把全零动作理解为“保持姿态”，它表示所有目标位置为零。

## 换成自己的机器人时

需要一起核对和替换 `uid`、`urdf_path`、关节名称与顺序、`keyframes`、控制器
参数及范围、mimic 关系、TCP 和手指 link 名称。URDF 引用的 mesh 路径也必须有效。
先让新模型在空环境中稳定运动，再接入任务环境。

第一版仅用于展示导入和位置控制；未配置抓取材质、`is_grasping`、`is_static`、
末端 IK 控制或腕部相机。基类的两个判断接口尚未实现，调用会报
`NotImplementedError`；这些功能可在具体抓取任务需要时添加。
