"""Create actors (rigid bodies).

The actor (or rigid body) in Sapien is created through a sapien.ActorBuilder. An
actor is an SAPIEN entity that typically consists of a rigid body component (for
physical simulation) and a visual component (for rendering). Note that can have
multiple collision and visual shapes, and they do not need to correspond.

Concepts:
    - Create an actor by primitives (box, sphere, capsule)
    - Create an actor by mesh files
    - sapien.Pose

"""

import sapien as sapien
from sapien.utils import Viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler2quat(x,y,z):
    euler = [x,y,z]
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion


def create_box(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    """Create a box.

    Args:
        scene: sapien.Scene to create a box.
        pose: 6D pose of the box.
        half_size: [3], half size along x, y, z axes.
        color: [4], rgba
        name: name of the actor.

    Returns:
        sapien.Entity
    """
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(
            half_size=half_size, material=sapien.physx.get_default_material()
        )
    )

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(
            half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])
        )
    )

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)

    # name and pose may be changed after added to scene
    # entity.set_name(name)
    # entity.set_pose(pose)

    return entity

def create_box_v2(
    scene: sapien.Scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    """Create a box.

    Args:
        scene: sapien.Scene to create a box.
        pose: 6D pose of the box.
        half_size: [3], half size along x, y, z axes.
        color: [3] or [4], rgb or rgba
        name: name of the actor.

    Returns:
        sapien.Entity
    """
    half_size = np.array(half_size)
    builder: sapien.ActorBuilder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)  # Add collision shape
    builder.add_box_visual(half_size=half_size, material=color)  # Add visual shape
    box: sapien.Entity = builder.build(name=name)
    box.set_pose(pose)
    return box

def create_sphere(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius,
    color=None,
    name="",
) -> sapien.Entity:
    """Create a sphere. See create_box."""
    builder = scene.create_actor_builder()
    builder.add_sphere_collision(radius=radius,density=1000.0)
    builder.add_sphere_visual(radius=radius, material=color)
    sphere = builder.build(name=name)
    sphere.set_pose(pose)
    return sphere

def create_capsule(
    scene: sapien.Scene,
    pose: sapien.Pose,
    radius,
    half_length,
    color=None,
    name="",
) -> sapien.Entity:
    """Create a capsule (x-axis <-> half_length). See create_box."""
    builder = scene.create_actor_builder()
    builder.add_capsule_collision(radius=radius, half_length=half_length)
    builder.add_capsule_visual(radius=radius, half_length=half_length, material=color)
    capsule = builder.build(name=name)
    capsule.set_pose(pose)
    return capsule

def create_table(
    scene: sapien.Scene,
    pose: sapien.Pose,
    size,
    height,
    thickness=0.1,
    color=(0.8, 0.6, 0.4),
    name="table",
) -> sapien.Entity:
    """Create a table (a collection of collision and visual shapes)."""
    builder = scene.create_actor_builder()

    # Tabletop
    tabletop_pose = sapien.Pose(
        [0.0, 0.0, -thickness / 2]
    )  # Make the top surface's z equal to 0
    tabletop_half_size = [size / 2, size / 2, thickness / 2]
    builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size)
    builder.add_box_visual(
        pose=tabletop_pose, half_size=tabletop_half_size, material=color
    )

    # Table legs (x4)
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (size - thickness) / 2
            y = j * (size - thickness) / 2
            table_leg_pose = sapien.Pose([x, y, -height / 2])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
            builder.add_box_collision(
                pose=table_leg_pose, half_size=table_leg_half_size
            )
            builder.add_box_visual(
                pose=table_leg_pose, half_size=table_leg_half_size, material=color
            )

    table = builder.build(name=name)
    table.set_pose(pose)
    return table

def create_rope(
        scene: sapien.Scene,
        pose: sapien.Pose,
        rope_size: list =[0.2,0.5],#diameter,length
        rope_links=30,
        twist_limit = 85.0,
        curve_limit = 85.0,
        joint_friction=0.2,
        joint_damping=0.3,
        density=1000.0,
        fixset: bool = False,
        color:list = [0.1,0.2,0.5],#RGB
        physical_material: sapien.physx.PhysxMaterial = None,
        name: str = 'rope',

) -> sapien.physx.PhysxArticulation:

    if rope_links <= 5:
        rope_links = 6
        print(" [WARN]:\'rope_links\' 不得小于 5")
    elif rope_links >85:
        rope_links = 85
        print(" [WARN]:\'rope_links\' 不得超过 85")
    rope_radius = rope_size[0]/2
    link_length = (rope_size[1]/rope_links)/2

    rp_builder=scene.create_articulation_builder()

    #save every segments by number
    links_builders=[]
    links_by_name={}

    sqrt_half = np.sqrt(0.5)

    # SAPIEN四元数顺序为[w, x, y, z],将关节默认的x旋转轴分别转换到y轴和z轴。
    joint_axis_y = [sqrt_half, 0, 0, sqrt_half]
    joint_axis_z = [sqrt_half, 0, -sqrt_half, 0]

    for i in range(rope_links):
        link_name = f'rope_{i}'

        if i == 0 :
            link = rp_builder.create_link_builder()
        else:
            parent_link = links_builders[i-1]

            helper0 = rp_builder.create_link_builder(parent_link)
            helper0.set_name(f"rope_helper0_{i}")
            helper0.set_joint_name(f"rope_joint_x_{i}")
            helper0.set_mass_and_inertia(
            mass=1e-5,
            cmass_local_pose=sapien.Pose(),
            inertia=[1e-8, 1e-8, 1e-8],
            )

            #using two helpers to connect 3 DoFs.
            helper0.set_joint_properties(
                "revolute",
                limits=[[-np.deg2rad(twist_limit), np.deg2rad(twist_limit)]],
                pose_in_parent=sapien.Pose(
                    p=[link_length, 0, 0],
                    ),
                pose_in_child=sapien.Pose(
                    p=[0, 0, 0],
                    ),
                friction=joint_friction,
                damping=joint_damping,
            )

            helper1 = rp_builder.create_link_builder(helper0)
            helper1.set_name(f"rope_helper1_{i}")
            helper1.set_joint_name(f"rope_joint_y_{i}")
            helper1.set_mass_and_inertia(
            mass=1e-5,
            cmass_local_pose=sapien.Pose(),
            inertia=[1e-8, 1e-8, 1e-8],
            )

            helper1.set_joint_properties(
                "revolute",
                limits=[[-np.deg2rad(curve_limit), np.deg2rad(curve_limit)]],
                pose_in_parent=sapien.Pose(
                    p=[0, 0, 0],
                    q=joint_axis_y,
                ),
                pose_in_child=sapien.Pose(
                    p=[0, 0, 0],
                    q=joint_axis_y,
                ),
                friction=joint_friction,
                damping=joint_damping,
            )

            link = rp_builder.create_link_builder(helper1)
            link.set_joint_name(f'rope_joint_z_{i}')
            link.set_joint_properties(
                "revolute",
                limits=[[-np.deg2rad(curve_limit), np.deg2rad(curve_limit)]],
                pose_in_parent=sapien.Pose(
                    p=[0, 0, 0],
                    q=joint_axis_z,
                ),
                pose_in_child=sapien.Pose(
                    p=[-link_length, 0, 0],
                    q=joint_axis_z,
                ),
                friction=joint_friction,
                damping=joint_damping,
            )

        link.set_name(link_name)

        link.add_capsule_collision(
            density=density,radius=rope_radius,
            half_length=link_length-rope_radius*1.2,
            material=physical_material,
            )
        link.add_capsule_visual(
            radius=rope_radius, half_length=link_length,
            material=color)

        links_builders.append(link)
        links_by_name[link_name] = link

    rp_builder.set_initial_pose(pose)
    if hasattr(rp_builder, "set_name"):
        # ManiSkill ArticulationBuilder：
        # 名称必须在build()时提供。
        rope = rp_builder.build(
            name=name,
            fix_root_link=fixset,
    )
    else:
        # 原生SAPIEN ArticulationBuilder：
        # build()不接受name，构建后再命名。
        rope = rp_builder.build(
            fix_root_link=fixset,
        )
        rope.set_name(name)
    rope.set_pose(pose)


    return rope

def set_material(
    static_friction:float = 0.3,
    dynamic_friction:float = 0.3,
    restitution:float = 0.1,
):
    default_material = sapien.physx.get_default_material()
    sapien.physx.set_default_material(static_friction=static_friction,
                                    dynamic_friction=dynamic_friction,
                                    restitution=restitution)
    return default_material

def main():
    engine = sapien.Engine()
    #renderer = sapien.SapienRenderer()
    #engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # ---------------------------------------------------------------------------- #
    # Add actors
    # ---------------------------------------------------------------------------- #
    scene.add_ground(altitude=0)  # The ground is in fact a special actor.
    box = create_box(
        scene,
        sapien.Pose(p=[0, 0.05, 1.0 + 0.1]),
        half_size=[0.05, 0.1, 0.1],
        color=[1.0, 0.0, 0.0],
        name="box",
    )
    sphere = create_sphere(
        scene,
        sapien.Pose(p=[0, -0.2, 1.0 + 0.05]),
        radius=0.05,
        color=[0.0, 1.0, 0.0],
        name="sphere",
    )

    table = create_table(
        scene,
        sapien.Pose(p=[0.0, 0, 1.0],q=[0,0,0,1.0]),
        size=1.0,
        height=1.0,
    )

    mt1 = set_material(0.3,0.2,0.5)

    rp1 = create_rope(
        scene,
        sapien.Pose(p=[-0.2, 0, 1.3],q=[1.0,0,0,0]),
        rope_size=(0.02,1),
        rope_links=35,
        density=5.0,
        joint_damping=0.2,
        joint_friction=0.1,
        color=[1.0,0,0],
        physical_material=mt1,

    )

    rp2 = create_rope(
        scene,
        sapien.Pose(p=[-0.2, 0.1, 1.3],q=[1.0,0.5,0,0]),
        rope_size=(0.02,1),
        rope_links=35,
        density=5.0,
        joint_damping=0.2,
        joint_friction=0.1,

        )


    #making some lights
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    #create a camera
    viewer = scene.create_viewer()

    viewer.set_camera_xyz(x=0, y=2, z=2.5)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=np.pi/2)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
