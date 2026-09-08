# Official Franka model provenance

- Upstream: <https://github.com/frankarobotics/franka_description>
- Release tag: `2.8.1`
- Commit: `02afaae282d4a8e10d7d2f781b23b3515c303ce5`
- License: Apache-2.0; see `LICENSE` and `NOTICE` in this directory.

`urdfs/fr3_duo_franka_hand.urdf` and the matching SRDF were expanded with
Xacro 2.1.1 from the upstream `fr3_duo` entry point, using its default two
`fr3v2` arms and `franka_hand` end effectors. The generated-file source-path
comment was changed from a temporary absolute path to a portable relative
description; robot geometry, kinematics, dynamics and limits were not edited.

To keep this tutorial reasonably small, only the mesh directories referenced
by that generated URDF are included. The upstream Xacro/YAML source files are
retained for traceability; generating other robot variants may require the
remaining upstream meshes and a ROS/Xacro setup.
