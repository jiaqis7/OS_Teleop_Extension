# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Functions to convert from USD."""

# Standard Library
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third Party
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

# NVIDIA
import nvidia.srl.urdf_xml.schema as urdf
import nvidia.srl.usd.prim_helper as prim_helper
from nvidia.srl.abc.srl import SRL
from nvidia.srl.basics.types import PathLike
from nvidia.srl.from_usd._from_usd_helper import NodeType, _get_prim_scale
from nvidia.srl.from_usd.transform_graph import TransformEdge, TransformGraph, TransformNode
from nvidia.srl.from_usd.transform_graph_tools import reduce_to_urdf
from nvidia.srl.math.transform import Rotation, Transform


class UsdToUrdf(SRL):
    """Class used to convert USD files to URDF files."""

    # The value the URDF joint limit effort is set to when the USD joint force limit is set to "Not
    # Limited".
    MAX_JOINT_EFFORT_VALUE = 10e6

    def __init__(
        self,
        stage: Usd.Stage,
        node_names_to_remove: Optional[str] = None,
        edge_names_to_remove: Optional[str] = None,
        root: Optional[str] = None,
        parent_link_is_body_1: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize a new `UsdToUrdf` object.

        Args:
            stage: USD stage that describes the scene.
            node_names_to_remove: List of node names to remove from the `TransformGraph` to break
                kinematic loops and make the graph transformable to something valid to create a URDF.
            edge_names_to_remove: List of edge names to remove from the `TransformGraph` to break
                kinematic loops and make the graph transformable to something valid to create a URDF.
            root: The root node name that will be set as the root of the kinematic structure of the
                new URDF.  This sets the "robot" element in the new URDF. The root node can either
                be specified with the prim path or with the node name.
            parent_link_is_body_1: A list of joint node names where the parent link is assumed to be
                the body 1 target prim, instead of the default body 0 target prim. Note, when only
                one body target is set, then the parent link is assumed to be the default prim, and
                the child link is the prim of whatever body target that is set.
            kinematics_only: If true, the re
            kwargs: Additional keyword arguments are passed to the parent class,
                `SRL <https://srl.gitlab-master-pages.nvidia.com/py/base/_api/nvidia.srl.abc.srl.html#nvidia.srl.abc.srl.SRL>`_.
        """  # noqa: E501
        # Initialize parent class
        super().__init__(**kwargs)

        # Initialize instance variables
        self._usd_path: Optional[Path] = None
        self._stage = stage
        log_level = kwargs.get("log_level", None)
        self._graph = TransformGraph.init_from_stage(
            stage,
            parent_link_is_body_1=parent_link_is_body_1,
            log_level=log_level,
        )

        nodes_to_remove: Optional[List[TransformNode]] = None
        if node_names_to_remove is not None:
            nodes_to_remove = list(
                map(lambda name_: self._graph.get_node(name_), node_names_to_remove)
            )

        edges_to_remove: Optional[List[TransformEdge]] = None
        if edge_names_to_remove is not None:
            edges_to_remove = list(
                map(lambda name_: self._graph.get_edge(name_), edge_names_to_remove)
            )

        if root is None:
            root_node = None
        else:
            root_node = self._graph.get_node(root)

        reduce_to_urdf(
            self._graph,
            nodes_to_remove=nodes_to_remove,
            edges_to_remove=edges_to_remove,
            root_node=root_node,
        )

        if not self._graph.is_possible_urdf():
            msg = (
                "Cannot build URDF from USDs that are not structured as kinematic trees. Consider"
                " restructuring your USD stage, or using the `node_names_to_remove`,"
                " `edge_names_to_remove`, and/or `parent_link_is_body_1` options to make the the"
                " transform graph into a tree. For more information see:\n"
                " https://srl.gitlab-master-pages.nvidia.com/py/usd_to_urdf/algorithm.html"
            )
            raise RuntimeError(msg)

    @classmethod
    def init_from_file(cls, usd_path: PathLike, **kwargs: Any) -> "UsdToUrdf":
        """Create a new `UsdToUrdf` object from a USD file.

        Args:
            usd_path: Path to USD file that describes the scene.
            kwargs: Additional keyword arguments are passed to
                 :class:`UsdToUrdf.__init__()<UsdToUrdf>`.

        Returns:
            UsdToUrdf: New `UsdToUrdf` object initialized from USD path.
        """
        if not isinstance(usd_path, str):
            usd_path = str(usd_path)
        stage = Usd.Stage.Open(usd_path)
        usd_to_urdf = cls(stage, **kwargs)
        usd_to_urdf._usd_path = Path(usd_path)
        return usd_to_urdf

    def save_to_file(self, urdf_output_path: PathLike, quiet: bool = False, **kwargs: Any) -> Path:
        """Convert the USD to a URDF and save to file.

        Args:
            urdf_output_path: The path to where the URDF file will be saved. If it is a file path
                then it is saved to that file. If it is a directory path, then it is a saved into
                that directory with the file name matching the USD name but with the .urdf
                extension. If the pathe doesn't exist, then file paths are assumed to have
                extensions (usually the ".urdf" extension) and directory paths are assumed to not
                have extensions.
            quiet: If true, nothing is printed or written to the logs.
            kwargs: Additional keyword arguments are passed to
                :func:`UsdToUrdf.to_str()<UsdToUrdf.to_str>`.
        """
        if not isinstance(urdf_output_path, Path):
            urdf_output_path = Path(urdf_output_path)

        if (
            urdf_output_path.exists() and urdf_output_path.is_file()
        ) or urdf_output_path.suffix != "":
            # `urdf_output_path` assumed to be a file
            output_file = urdf_output_path
            output_dir = urdf_output_path.parent
        elif (
            urdf_output_path.exists() and urdf_output_path.is_dir()
        ) or urdf_output_path.suffix == "":
            # `urdf_output_path` assumed to be a directory
            output_dir = urdf_output_path
            if self._usd_path is None:
                output_name = self._graph.name + ".urdf"
            else:
                output_name = self._usd_path.stem + ".urdf"
            output_file = output_dir / output_name

        else:
            msg = (
                f"The URDF output is not valid: {urdf_output_path}. It must be either a path to a"
                " file or a directory."
            )
            ValueError(msg)

        output_dir.mkdir(parents=True, exist_ok=True)

        urdf_str = self.to_str(output_dir=output_dir.as_posix(), quiet=True, **kwargs)

        with open(output_file.as_posix(), "w") as file:
            file.write(urdf_str)

        # Log result
        if not quiet:
            if self._usd_path is None:
                usd_path = self._stage.GetRootLayer().realPath
            else:
                usd_path = self._usd_path

            msg = "\n".join(
                [
                    "Converted USD to URDF.",
                    f"    Input file: {usd_path}",
                    f"    Output file: {output_file}",
                ]
            )
            self.logger.info(msg)

        return output_file

    def to_str(
        self,
        output_dir: Optional[PathLike] = None,
        mesh_dir: Optional[PathLike] = None,
        mesh_path_prefix: str = "",
        use_uri_file_prefix: bool = False,
        visualize_collision_meshes: bool = False,
        kinematics_only: bool = False,
        quiet: bool = False,
    ) -> str:
        """Convert the USD to URDF and return the URDF XML string.

        Args:
            output_dir: The directory to where the URDF file would be saved to. This is used to
                calculate the relative path between the mesh directory and the output directory.
            mesh_dir: The directory where the mesh files will be stored. Defaults to creating a
                "meshes" directory in the output directory if that is set, otherwise in the current
                working directory.
            mesh_path_prefix: Set the prefix to use for the URDF mesh filename. For example, to use
                an absolute path set this to '$(pwd)/'. Or to use a URI with the 'file' scheme,
                then set this to 'file://'.
            use_uri_file_prefix: If true, ignore the given `mesh_path_prefix` value and
                set it to use a URI with the 'file' scheme with the absolute path set to the output
                directory (i.e.  'file://<absolute path to output directory>/')."
            visualize_collision_meshes: If true, the collision meshes will be added to the set of
                visual elements in the link elements.
            kinematics_only: If true, the resulting URDF will not contain any visual or collision
                mesh information.
            quiet: If true, nothing is printed or written to the logs.

        Returns:
            The URDF XML string.
        """
        if output_dir is None:
            output_dir = Path.cwd().as_posix()

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if isinstance(mesh_dir, str):
            mesh_dir = Path(mesh_dir)

        if mesh_dir is None:
            mesh_dir = output_dir / "meshes"
        self._mesh_dir_path = mesh_dir.expanduser().resolve()

        self._output_dir_path = output_dir
        self._mesh_path_prefix = mesh_path_prefix
        self._use_uri_file_prefix = use_uri_file_prefix
        self._visualize_collision_meshes = visualize_collision_meshes
        self._kinematics_only = kinematics_only

        self.robot_joints: Dict[str, urdf.Joint] = {}
        self.robot_links: Dict[str, urdf.Link] = {}

        self._build_urdf()

        urdf_str = self.robot.to_xml_str()

        # Log result
        if not quiet:
            if self._usd_path is None:
                usd_path = self._stage.GetRootLayer().realPath
            else:
                usd_path = self._usd_path
            msg = "\n".join(
                [
                    "Converted USD to URDF.",
                    f"    Input file: {usd_path}",
                ]
            )
            self.logger.info(msg)

        return urdf_str

    def save_graphviz(
        self,
        output_dir: Optional[PathLike] = None,
        name: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """Save the `TransformGraph` of the USD as a graphviz dot file and rendered PNG file.

        Args:
            output_dir: The directory to where the graphviz files will be saved to. Defaults to the
            current working directory.

        Returns:
            dot_file_path: Path to the generated dot file.
            png_file_path: Path to the generated png file.
        """
        if output_dir is None:
            output_dir = Path.cwd().as_posix()

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        dot_file_path = self._graph.save_graphviz(output_dir=output_dir, name=name)
        png_file_path = self._graph.render_graphviz(output_dir=output_dir, name=name)

        return dot_file_path, png_file_path

    def _build_urdf(self) -> None:
        robot_name = self._graph.name
        self.robot = urdf.Robot(name=robot_name)
        for node in self._graph.sorted_nodes:
            if node.type == NodeType.GEOMETRY:
                self._add_geometry(node)
            elif node.type == NodeType.JOINT:
                self._add_joint(node)
            elif (
                node.type == NodeType.LINK
                or node.type == NodeType.SENSOR
                or node.type == NodeType.PHONY
            ):
                self._add_link(node)
            else:
                raise RuntimeError(f"Unknown node type for '{node.path}'")

        self._apply_geom_node_inertial_info()

    def _add_joint(self, node: TransformNode) -> None:
        if node.prim is None:
            raise RuntimeError("Something is wrong. A joint node shoud always have a prim value.")
        joint_prim = node.prim
        if node.is_root is None:
            raise RuntimeError("Something is wrong. A joint node should never be the root node.")
        parent_node = node.to_neighbors[0]
        child_node = node.from_neighbors[0]

        lower_limit, upper_limit = prim_helper.get_joint_limits(joint_prim)

        if prim_helper.is_a_fixed_joint(joint_prim) or prim_helper.is_an_unassigned_joint(
            joint_prim
        ):
            joint_type = "fixed"
        elif prim_helper.is_a_revolute_joint(joint_prim):
            if (
                lower_limit is None
                or np.abs(lower_limit) == np.inf
                or upper_limit is None
                or np.abs(upper_limit) == np.inf
            ):
                joint_type = "continuous"
            else:
                joint_type = "revolute"
        elif prim_helper.is_a_prismatic_joint(joint_prim):
            joint_type = "prismatic"
        else:
            schema_type = joint_prim.GetPrimTypeInfo().GetSchemaTypeName()
            msg = (
                f"Joint '{node.path}' will be skipped. The '{node.name}' joint is of type"
                f" '{schema_type}', which is not currently supported. Supported joint types:"
                " 'UsdPhysics.FixedJoint', 'UsdPhysics.RevoluteJoint',"
                " 'UsdPhysics.PrismaticJoint'."
            )
            raise RuntimeError(msg)

        # Add joint
        joint = urdf.Joint(name=node.name, type=joint_type)
        parent_link_t_joint = node.from_edges[0].transform
        joint_t_child_link = node.to_edges[0].transform
        child_link_t_joint = Transform.inverse(joint_t_child_link)
        parent_link_t_child_link = parent_link_t_joint @ joint_t_child_link
        joint.origin = UsdToUrdf._get_urdf_pose(parent_link_t_child_link)
        joint.parent = urdf.Parent(link=parent_node.name)
        joint.child = urdf.Child(link=child_node.name)
        joint_axis = prim_helper.get_joint_axis(joint_prim)
        if joint_axis is not None:
            joint_frame_axis_vector = np.array(joint_axis.value)
            child_frame_axis_vector = (
                Transform.get_rotation(child_link_t_joint) @ joint_frame_axis_vector
            )
            # NOTE (roflaherty): This removes very small numbers by rounding to the nearest 6th
            # decimal.  The `+ np.zeros(3)` is needed to remove the -0.0 that happens when a small
            # negative number gets rounded to zero.
            child_frame_axis_vector = np.round(child_frame_axis_vector, 6) + np.zeros(3)
            joint.axis = urdf.Axis(xyz=child_frame_axis_vector)
        if joint_type == "revolute" or joint_type == "prismatic":
            joint.limit = urdf.Limit()
            if lower_limit is not None:
                joint.limit.lower = (
                    math.radians(lower_limit) if joint_type == "revolute" else lower_limit
                )
            if upper_limit is not None:
                joint.limit.upper = (
                    math.radians(upper_limit) if joint_type == "revolute" else upper_limit
                )

            velocity_limit = prim_helper.get_joint_velocity_limit(joint_prim)
            if velocity_limit is not None:
                joint.limit.velocity = (
                    math.radians(velocity_limit) if joint_type == "revolute" else velocity_limit
                )

            effort_limit = prim_helper.get_joint_force_limit(joint_prim)

            if effort_limit is not None and np.abs(effort_limit) == np.inf:
                effort_limit = np.sign(effort_limit) * self.MAX_JOINT_EFFORT_VALUE

            if effort_limit is not None:
                joint.limit.effort = effort_limit

        self.robot.joints.append(joint)
        self.robot_joints[node.sku] = joint

    def _add_fixed_joint(self, node: TransformNode) -> None:
        if node.is_root:
            raise RuntimeError("Something is wrong. A joint node should never be the root node.")

        parent_node = node.to_neighbors[0]

        joint_name = f"joint_{parent_node.name}-{node.name}"

        joint = urdf.Joint(name=joint_name, type="fixed")
        parent_link_t_child_link = node.from_edges[0].transform
        joint.origin = UsdToUrdf._get_urdf_pose(parent_link_t_child_link)
        joint.parent = urdf.Parent(link=parent_node.name)
        joint.child = urdf.Child(link=node.name)
        self.robot.joints.append(joint)
        self.robot_joints[f"{node.sku}/fixed_joint"] = joint

    def _add_link(self, node: TransformNode) -> None:
        if (
            (
                node.type == NodeType.PHONY
                or node.type == NodeType.LINK
                or node.type == NodeType.SENSOR
            )
            and not node.is_root
            and (
                node.to_neighbors[0].type == NodeType.PHONY
                or node.to_neighbors[0].type == NodeType.LINK
            )
        ):
            self._add_fixed_joint(node)

        link = urdf.Link(name=node.name)

        # NOTE: Adding this attribute (`_urdf_obj`) is a bit of hack that is needed for the
        # `_apply_geom_node_inertial_info` method.
        node._urdf_obj = link  # type: ignore[attr-defined]

        if node.type != NodeType.PHONY:
            if node.prim is None:
                raise RuntimeError(
                    "Something is wrong. A link node should always have a prim value."
                )
            if node.prim.HasAPI(UsdPhysics.MassAPI):
                inertial = self._urdf_inertial_from_mass_api(node)
                if inertial is not None:
                    link.inertial = inertial

        self.robot.links.append(link)
        self.robot_links[node.sku] = link

    @staticmethod
    def _urdf_inertial_from_mass_api(node: TransformNode) -> Optional[urdf.Inertial]:
        """Extract the inertial data from the prim's MassAPI as a `urdf.Inertial` object."""
        if node.prim is None or not node.prim.HasAPI(UsdPhysics.MassAPI):
            msg = (
                "Something is wrong. This function should only be called with nodes that have a"
                " prim value that has the `MassAPI`."
            )
            raise RuntimeError(msg)

        mass_api = UsdPhysics.MassAPI(node.prim)
        mass_val = mass_api.GetMassAttr().Get()
        if mass_val <= 0:
            return None

        mass = urdf.Mass(value=mass_val)

        center_of_mass_xyz = list(mass_api.GetCenterOfMassAttr().Get())
        inertial_origin = urdf.Pose(xyz=center_of_mass_xyz)

        prin_axes_gf_quat = mass_api.GetPrincipalAxesAttr().Get()
        prin_axes_quat = np.array(
            list(prin_axes_gf_quat.GetImaginary()) + [prin_axes_gf_quat.GetReal()]
        )

        try:
            prin_axes_rot = Rotation.from_quat(prin_axes_quat)
        except ValueError:
            prin_axes_rot = Rotation.identity()

        diag_mat = np.diag(np.array(mass_api.GetDiagonalInertiaAttr().Get()))
        if np.all(diag_mat == np.zeros((3, 3))):
            inertial_origin = urdf.Pose()
            inertia = urdf.Inertia()
        else:
            inertia_mat = prin_axes_rot.as_matrix() @ diag_mat @ prin_axes_rot.as_matrix().T

            ixx = inertia_mat[0][0]
            ixy = inertia_mat[0][1]
            ixz = inertia_mat[0][2]
            iyy = inertia_mat[1][1]
            iyz = inertia_mat[1][2]
            izz = inertia_mat[2][2]
            inertia = urdf.Inertia(ixx, ixy, ixz, iyy, iyz, izz)

        inertial = urdf.Inertial(origin=inertial_origin, mass=mass, inertia=inertia)
        return inertial

    def _add_geometry(self, node: TransformNode) -> None:
        # A geometry node should never be the root node.
        assert not node.is_root

        # NOTE (roflaherty): This is a hack to deal with the hidden camera geometry prims that only
        # exist when converting from an open stage in Isaac Sim using the USD to URDF Exporter
        # extension.
        if "CameraModel" in node.name:
            msg = f"Skipping adding {node.name} geometry."
            self.logger.debug(msg)
            return

        if not self._kinematics_only:
            geometry_visual, geometry_collision, origin = self._get_geometry(node)

            node_parent = node.to_neighbors[0]
            if not node.is_leaf or (
                node_parent.type != NodeType.LINK and node_parent.type != NodeType.PHONY
            ):
                msg = (
                    f"The '{node.path}' geometry is not a leaf node or is connected to a joint"
                    " directly. The USD to URDF convertered currently only supports converting"
                    " USDs where geometries are leaf nodes. This means that geometries (i.e."
                    " meshes and primitives prims) must be child prims of Xform prims and must not"
                    " connect to joint prims directly."
                )
                raise RuntimeError(msg)

            link = self.robot_links[node_parent.sku]

            if geometry_visual is not None:
                urdf_visual = urdf.Visual(origin=origin, geometry=geometry_visual)
                if urdf_visual not in link.visuals:
                    link.visuals.append(urdf_visual)

            if geometry_collision is not None:
                urdf_collision = urdf.Collision(origin=origin, geometry=geometry_collision)
                if urdf_collision not in link.collisions:
                    link.collisions.append(urdf_collision)

                if self._visualize_collision_meshes:
                    urdf_visual = urdf.Visual(origin=origin, geometry=geometry_collision)
                    if urdf_visual not in link.visuals:
                        link.visuals.append(urdf_visual)

        return

    def _apply_geom_node_inertial_info(self) -> None:
        """Apply the geometry node inertial info to the connected to the link node."""
        geom_nodes_w_mass = list(
            filter(
                lambda node_: node_.type == NodeType.GEOMETRY
                and node_.prim is not None
                and node_.prim.HasAPI(UsdPhysics.MassAPI),
                self._graph.nodes,
            )
        )
        for geom_node in geom_nodes_w_mass:
            geom_inertial = self._urdf_inertial_from_mass_api(geom_node)
            if geom_inertial is None:
                continue
            if len(geom_node.to_neighbors) != 1:
                raise RuntimeError(
                    "Something is wrong. A geom node should always be connected to a single link."
                )
            link_node = geom_node.to_neighbors[0]
            link___geom = geom_node.from_edges[0].transform
            if geom_inertial.origin.rpy != [0, 0, 0]:
                msg = (
                    "Something is wrong. The inertial.origin.rpy value is"
                    f" `{geom_inertial.origin.xyz}` and it should always be `[0, 0, 0]`."
                )
                raise RuntimeError(msg)

            # Calculate center of mass in the link frame
            com_xyz = list(geom_inertial.origin.xyz)
            com_homo_vec = com_xyz + [1]
            geom___com = np.array(com_homo_vec, dtype=float)
            link___com = link___geom @ geom___com
            link_inertial_origin_xyz = link___com[0:3].tolist()
            link_inertial_origin = urdf.Pose(xyz=link_inertial_origin_xyz)

            # Calculate the inertia tensor in the link frame
            # - Convert inertia matrix data wrt geom to matrix form
            ixx = geom_inertial.inertia.ixx
            ixy = geom_inertial.inertia.ixy
            ixz = geom_inertial.inertia.ixz
            iyy = geom_inertial.inertia.iyy
            iyz = geom_inertial.inertia.iyz
            izz = geom_inertial.inertia.izz
            geom___inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            # - Get just the rotation portion of the geom to link transformation matrix
            link___geom_rot = Transform.get_rotation(link___geom)
            # - Apply the inertia transformation
            link___inertia_mat = link___geom_rot @ geom___inertia @ link___geom_rot.T
            # - Extract out the components out of the inertial tensor in link frame
            ixx = link___inertia_mat[0][0]
            ixy = link___inertia_mat[0][1]
            ixz = link___inertia_mat[0][2]
            iyy = link___inertia_mat[1][1]
            iyz = link___inertia_mat[1][2]
            izz = link___inertia_mat[2][2]
            link_inertia = urdf.Inertia(ixx, ixy, ixz, iyy, iyz, izz)
            # - Apply the transformed inertial tensor to the link urdf object
            link_inertial = urdf.Inertial(
                origin=link_inertial_origin, mass=geom_inertial.mass, inertia=link_inertia
            )
            link_node._urdf_obj.inertial = link_inertial  # type: ignore[attr-defined]

    @staticmethod
    def _get_urdf_pose(transform: np.ndarray) -> urdf.Pose:
        xyz = Transform.get_translation(transform)
        with warnings.catch_warnings():
            msg = (
                "Gimbal lock detected. Setting third angle to zero since it is not possible to"
                " uniquely determine all angles."
            )
            warnings.filterwarnings("ignore", message=msg)
            rpy = Transform.get_rotation(transform, as_rpy=True)
        return urdf.Pose(xyz=xyz, rpy=rpy)

    def _get_geometry(
        self, node: TransformNode
    ) -> Tuple[Optional[urdf.Geometry], Optional[urdf.Geometry], urdf.Pose]:
        """Get urdf.Geometry objects for the prim.

        Args:
            node: The node to get the geometry for

        Returns:
            visual geometry: The visual geometry for the prim.
            collision geometry: The collision geometry for the prim.
            origin: TBD.
        """
        if node.prim is None:
            raise RuntimeError(
                "Something is wrong. A geometry node shoud always have a prim value."
            )
        prim = node.prim

        link_t_geometry = node.from_edges[0].transform
        prim_scale = _get_prim_scale(prim)
        transform_scale = Transform.get_scale(node.scale)

        geometry_visual: Optional[urdf.Geometry] = None
        geometry_collision: Optional[urdf.Geometry] = None

        rotation_correction = Transform.identity()

        if prim.IsA(UsdGeom.Cylinder):
            if prim_scale is not None and np.allclose(transform_scale, np.ones(3)):
                scale = prim_scale
            else:
                scale = transform_scale

            radius = prim.GetAttribute("radius").Get()
            length = prim.GetAttribute("height").Get()
            axis = prim.GetAttribute("axis").Get()
            if axis == "X":
                rotation_correction = Transform.from_rotvec(np.array([0, np.pi / 2, 0]))
                scale_r1 = scale[1]
                scale_r2 = scale[2]
                scale_h = scale[0]
            elif axis == "Y":
                rotation_correction = Transform.from_rotvec(np.array([np.pi / 2, 0, 0]))
                scale_r1 = scale[0]
                scale_r2 = scale[2]
                scale_h = scale[1]
            elif axis == "Z":
                rotation_correction = Transform.identity()
                scale_r1 = scale[0]
                scale_r2 = scale[1]
                scale_h = scale[2]
            else:
                raise RuntimeError(f"Unknown cylinder axis value '{axis}' for '{node.path}'.")

            if not np.isclose(scale_r1, scale_r2):
                msg = (
                    "URDF cannot scale the radii of each axis of a cylinder different amounts."
                    f" Cylinder '{node.path}' has scaling factors of {scale_r1} and {scale_r2} for"
                    " non-height axes."
                )
                raise RuntimeError(msg)

            geometry_visual = urdf.Geometry(
                choice=urdf.Cylinder(radius=radius * scale_r1, length=length * scale_h)
            )
            geometry_collision = geometry_visual
        elif prim.IsA(UsdGeom.Cube):
            if prim_scale is not None and np.allclose(transform_scale, np.ones(3)):
                scale = prim_scale
            else:
                scale = transform_scale

            attr_size = prim.GetAttribute("size").Get()
            size = scale * attr_size
            geometry_visual = urdf.Geometry(choice=urdf.Box(size=size))
            geometry_collision = geometry_visual
        elif prim.IsA(UsdGeom.Sphere):
            if prim_scale is not None and np.allclose(transform_scale, np.ones(3)):
                scale = prim_scale
            else:
                scale = transform_scale

            if not np.isclose(scale[0], scale[1]) or not np.isclose(scale[0], scale[2]):
                msg = (
                    "URDF cannot scale the radii of each axis of a sphere different amounts."
                    f" Sphere '{node.path}' has scaling factors of {scale[0]}, {scale[1]}, and"
                    f" {scale[2]}."
                )
                raise RuntimeError(msg)

            radius = prim.GetAttribute("radius").Get() * scale[0]
            geometry_visual = urdf.Geometry(choice=urdf.Sphere(radius=radius))
            geometry_collision = geometry_visual
        elif prim.IsA(UsdGeom.Mesh):
            self._mesh_dir_path.mkdir(parents=True, exist_ok=True)

            # NOTE (roflaherty): This check is needed because some mesh file paths are longer than
            # the maximum allowed
            MAX_FILE_CHAR_LEN = 255  # noqa: N806
            node_name = node.name
            path_char_len = len(node_name) + len(self._mesh_dir_path.as_posix())
            if path_char_len > MAX_FILE_CHAR_LEN:
                chars_to_keep = MAX_FILE_CHAR_LEN - len(self._mesh_dir_path.as_posix()) - 4
                if chars_to_keep < 1:
                    raise RuntimeError(f"Mesh directory path is too long: {self._mesh_dir_path}")
                node_name = node.name[-chars_to_keep:-1]

            mesh_filename = f"{node_name}.obj"
            obj_output_path = self._mesh_dir_path / mesh_filename

            transform = Transform.from_scale(transform_scale)
            prim_helper.export_geometry_as_obj_file(prim, obj_output_path, transform=transform)

            if self._use_uri_file_prefix:
                obj_dir = "file://" + (self._mesh_dir_path / mesh_filename).as_posix()
            else:
                try:
                    rel_mesh_dir_path = self._mesh_dir_path.relative_to(
                        self._output_dir_path.resolve()
                    )
                except ValueError:
                    # If the relative path to the mesh directory from the output directory cannot be
                    # found then set the path to the value in the `_mesh_dir_path` attribute
                    # directly.
                    obj_dir = (self._mesh_dir_path / mesh_filename).as_posix()
                else:
                    obj_dir = (rel_mesh_dir_path / mesh_filename).as_posix()
                obj_dir = self._mesh_path_prefix + obj_dir
            urdf_mesh = urdf.Mesh(obj_dir)
            geometry_visual = urdf.Geometry(choice=urdf_mesh)
            geometry_collision = urdf.Geometry(choice=urdf_mesh)
        else:
            raise RuntimeError(
                "Invalid prim type. Valid types: `UsdGeom.Cylinder`, `UsdGeom.Cube`,"
                " `UsdGeom.Sphere`, `UsdGeom.Mesh`."
            )

        if not prim_helper.is_collider(prim):
            geometry_collision = None

        if prim_helper.get_attribute_value(prim, "purpose") in ["guide", "proxy"]:
            geometry_visual = None

        # NOTE (roflaherty): It is debatable on whether the prim's visibility should be used to
        # determine if the prim is a visual geometry. It is commented out for now, meaning it is not
        # used to determine if the prim is a visual geometry.

        # if not prim_helper.is_visible(prim):
        #     geometry_visual = None

        origin = UsdToUrdf._get_urdf_pose(link_t_geometry @ rotation_correction)

        return geometry_visual, geometry_collision, origin
