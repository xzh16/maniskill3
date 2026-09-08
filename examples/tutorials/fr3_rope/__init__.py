"""基于本地 RobotAgentTemplate 的 FR3 双臂与线缆实验包。"""

from .envs import FR3CableEnv
from .robot import FR3DualArm, FR3DualArmCAD

__all__ = ["FR3CableEnv", "FR3DualArm", "FR3DualArmCAD"]
