# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Helper functions for Usd.Prim objects."""

# Standard Library
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# Third Party
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# NVIDIA
from nvidia.srl.basics.enum import BodyType, Enum, PoseType
from nvidia.srl.basics.types import (
    Affine,
    AttachedState,
    BodyState,
    Interval,
    JointState,
    PathLike,
    Pose,
    Twist,
    Vector,
)
from nvidia.srl.math.transform import Rotation, Transform


def open_stage(usd_path: PathLike) -> Usd.Stage:
    """Open the stage for the USD at the given path.

    Args:
        usd_path: File path to load the USD from.

    Returns:
        The USD stage.
    """
    stage = Usd.Stage.Open(str(usd_path))
    return stage


def get_prim(stage: Usd.Stage, prim_path: str) -> Optional[Usd.Prim]:
    """Get the prim the given path."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    return prim


def get_prims(
    stage: Usd.Stage,
    path_pattern: Optional[str] = None,
    path_match: Optional[str] = None,
    path_pattern_exclusion: Optional[str] = None,
    prim_types: Optional[Union[str, List[str]]] = None,
    prim_types_exclusion: Optional[Union[str, List[str]]] = None,
    has_apis: Optional[List[type]] = None,
    case_sensitive: bool = False,
    return_path: bool = False,
) -> Union[List[Usd.Prim], List[str]]:
    """Get prims based on specified constraints.

    Args:
        stage: USD Stage.
        path_pattern: The RegEx (Regular Expression) path pattern to match.
        path_match: Python string matching.  Faster than regex matching.
        path_pattern_exclusion: The RegEx path pattern to ignore.
        prim_types: List of prim types to include.
        prim_types_exclusion: List of prim types to ignore.
        has_apis: Include prims only if they have the given APIs.
        case_sensitive: If true, pattern and matching is case sensitive.
        return_path: If true, return prim path instead of prim.

    Returns:
        A list of prims or prim paths.
    """
    # regex setup
    re_flags = 0
    if case_sensitive:
        re_flags = re.IGNORECASE

    if path_pattern:
        path_pattern_regex = re.compile(path_pattern, flags=re_flags)

    if path_pattern_exclusion:
        path_exclusion_regex = re.compile(path_pattern_exclusion, flags=re_flags)

    gathered_prims: List[str] = []

    if (
        path_pattern is None
        and path_match is None
        and path_pattern_exclusion is None
        and prim_types is None
        and prim_types_exclusion is None
        and has_apis is None
    ):
        return gathered_prims

    if isinstance(prim_types, str):
        prim_types = [prim_types]

    if isinstance(prim_types_exclusion, str):
        prim_types_exclusion = [prim_types_exclusion]

    for prim in stage.Traverse():
        prim_path = get_path(prim)
        prim_type = str(prim.GetTypeName())

        if path_match:
            if case_sensitive:
                if path_match.lower() not in prim_path.lower():
                    continue
            else:
                if path_match not in prim_path:
                    continue
        if path_pattern:
            if not path_pattern_regex.search(prim_path):
                continue
        if path_pattern_exclusion:
            if path_exclusion_regex.search(prim_path):
                continue
        if prim_types:
            if case_sensitive:
                prim_types_set = {a_prim_type for a_prim_type in prim_types}
                if prim_type not in prim_types_set:
                    continue
            else:
                prim_types_set = {a_prim_type.lower() for a_prim_type in prim_types}
                if prim_type.lower() not in prim_types_set:
                    continue
        if prim_types_exclusion:
            if case_sensitive:
                prim_types_exclusion_set = {a_prim_type for a_prim_type in prim_types_exclusion}
                if prim_type in prim_types_exclusion_set:
                    continue
            else:
                prim_types_exclusion_set = {
                    a_prim_type.lower() for a_prim_type in prim_types_exclusion
                }
                if prim_type.lower() in prim_types_exclusion_set:
                    continue

        if has_apis:
            if any([not prim.HasAPI(an_api) for an_api in has_apis]):
                continue

        gathered_prims.append(prim_path if return_path else prim)
    return gathered_prims


def get_path(prim: Usd.Prim) -> str:
    """Get the path of the given prim."""
    return prim.GetPath().pathString


# TODO (roflaherty): Remove this function.
def get_prim_path(prim: Usd.Prim) -> str:
    """DEPRECATED: Use `get_path`."""
    msg = "`get_prim_path` is deprecated and will be removed in a future release. Use `get_path`."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return get_path(prim)


def get_root_prim(stage: Usd.Stage) -> Usd.Prim:
    """Get the root prim for the stage."""
    return stage.GetPrimAtPath("/")


def get_world_prim(stage: Usd.Stage) -> Usd.Prim:
    """Get the world prim for the stage."""
    return stage.GetDefaultPrim()


def get_attribute_names(prim: Usd.Prim) -> List[str]:
    """Get the attribute names for a prim."""
    return [attribute.GetName() for attribute in prim.GetAttributes()]


def get_attribute(prim: Usd.Prim, attribute_name: str) -> Optional[Usd.Attribute]:
    """Get an attribute if it exists."""
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        return None
    return attribute


def has_attribute(prim: Usd.Prim, attribute_name: str) -> bool:
    """Test whether a prim has an attribute."""
    return get_attribute(prim, attribute_name) is not None


def create_attribute(
    prim: Usd.Prim,
    attribute_name: str,
    sdf_type: Sdf.ValueTypeName,
    value: Optional[Any] = None,
    **kwargs: Any,
) -> Usd.Attribute:
    """Create an attribute if it does not exist."""
    attribute = get_attribute(prim, attribute_name)
    if attribute is None:
        attribute = prim.CreateAttribute(attribute_name, sdf_type)
    if value is not None:
        set_attribute_value(prim, attribute_name, value, **kwargs)
    return attribute


def get_attribute_value(
    prim: Usd.Prim, attribute_name: str, time_code: Optional[Usd.TimeCode] = None
) -> Optional[Any]:
    """Get the value of an attribute if it exists."""
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    attribute = get_attribute(prim, attribute_name)
    if attribute is None:
        return None
    return attribute.Get(time=time_code)


def set_attribute_value(
    prim: Usd.Prim,
    attribute_name: str,
    value: Any,
    time_code: Optional[Usd.TimeCode] = None,
) -> None:
    """Set the value of an attribute and create it if it does not exist.

    Raises:
        RuntimeError: If the attribute does not exist.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    attribute = get_attribute(prim, attribute_name)
    if attribute is None:
        msg = (
            f"Attribute '{attribute_name}' does not exist. Create the attribute first with the"
            " `create_attribute` function."
        )
        raise RuntimeError(msg)
    attribute.Set(value, time=time_code)


def get_stage_prims(stage: Usd.Stage) -> Iterable[Usd.Prim]:
    """Get the prims in a stage in depth-first order."""
    return iter(stage.Traverse())


def get_subtree_prims(prim: Usd.Prim) -> Iterable[Usd.Prim]:
    """Get the subtree starting at a prim in depth-first order."""
    return iter(Usd.PrimRange(prim))


def get_descendant_prims(prim: Usd.Prim) -> Iterable[Usd.Prim]:
    """Get the descendants of a prim using a preorder traversal."""
    yield from prim.GetChildren()
    for child_prim in prim.GetChildren():
        yield from get_descendant_prims(child_prim)


def get_ancestor_prims(prim: Usd.Prim) -> Iterable[Usd.Prim]:
    """Get the ancestors of a prim."""
    parent_prim = prim.GetParent()
    if parent_prim.IsValid():
        yield parent_prim
        yield from get_ancestor_prims(parent_prim)


def has_api(prim: Usd.Prim, api_name: str) -> bool:
    """Test whether a prim has an API."""
    return api_name in prim.GetAppliedSchemas()


def is_collider(prim: Usd.Prim) -> bool:
    """Check if the given prim is a collider object.

    A prim is a collider if it or any of its children have the "PhysicsCollisionAPI".
    """

    def collider_check(prim_: Usd.Prim) -> bool:
        """True if the prim has the "PhysicsCollisionAPI" applied and collision is enabled."""
        return has_api(prim_, api_name="PhysicsCollisionAPI") and has_attribute(
            prim_, attribute_name="physics:collisionEnabled"
        )

    return collider_check(prim) or any(map(collider_check, prim.GetChildren()))


def is_simulated(prim: Usd.Prim) -> bool:
    """Check if the given prim is a simulated body."""
    return has_api(prim, api_name="PhysicsRigidBodyAPI")


def is_fixed(prim: Usd.Prim) -> bool:
    """Check if the given prim is a fixed object (i.e. object that cannot move)."""
    if not is_entity(prim):
        return False
    stage = prim.GetStage()
    for joint in filter(is_a_fixed_joint, stage.Traverse()):
        link_pair = get_links_for_joint(joint)
        for link0, link1 in (link_pair, reversed(link_pair)):
            if link0 is not None and (prim == link0 or is_ancestor(prim, link0)):
                if link1 is None or link1 == get_world_prim(stage):
                    return True
    return not is_simulated(prim)


def is_floating(prim: Usd.Prim) -> bool:
    """Check if the given prim is a floating object (i.e. object that can move)."""
    return is_entity(prim) and not is_fixed(prim)


def is_a_joint(prim: Usd.Prim) -> bool:
    """Check if the given prim is a joint prim.

    A prim is a joint if it is any joint type.
    """
    return prim.IsA(UsdPhysics.Joint)


def is_an_unassigned_joint(prim: Usd.Prim) -> bool:
    """Check if the given prim is an unassigned joint prim.

    A prim is an unassigned joint if it is of type `UsdPhysics.Joint`.
    """
    return is_a_joint(prim) and not is_a_fixed_joint(prim) and not is_a_movable_joint(prim)


def is_a_fixed_joint(prim: Usd.Prim) -> bool:
    """Check if the given prim is a fixed joint prim.

    A prim is a fixed joint if it is of type `UsdPhysics.FixedJoint`.
    """
    return prim.IsA(UsdPhysics.FixedJoint)


def is_a_revolute_joint(prim: Usd.Prim) -> bool:
    """Check if the given prim is a revolute joint prim.

    A prim is a revolute joint if it is of type `UsdPhysics.RevoluteJoint`.
    """
    return prim.IsA(UsdPhysics.RevoluteJoint)


def is_a_prismatic_joint(prim: Usd.Prim) -> bool:
    """Check if the given prim is a prismatic joint prim.

    A prim is a prismatic joint if it is of type `UsdPhysics.PrismaticJoint`.
    """
    return prim.IsA(UsdPhysics.PrismaticJoint)


def is_a_movable_joint(prim: Usd.Prim) -> bool:
    """Check if the given prim is a movable joint prim.

    A prim is a movable joint if it is either of type `UsdPhysics.RevoluteJoint`,
    `UsdPhysics.PrismaticJoint`.
    """
    supported_joint_types = [
        UsdPhysics.RevoluteJoint,
        UsdPhysics.PrismaticJoint,
    ]
    return any(map(prim.IsA, supported_joint_types))


def is_a_drive_joint(prim: Usd.Prim) -> bool:
    """Check if the given prim is a drive joint prim.

    A prim is a drive joint if it is a joint and has any of the "PhysicsDriveAPI" schema.
    """
    physics_drive_apis = ["PhysicsDriveAPI:angular", "PhysicsDriveAPI:linear"]
    return is_a_joint(prim) and any(has_api(prim, api_name=api) for api in physics_drive_apis)


# TODO (roflaherty): Remove this function.
def is_articulated_root(prim: Usd.Prim) -> bool:
    """DEPRECATED: Use `is_articulated`."""
    msg = (
        "`is_articulated_root ` is deprecated and will be removed in a future release. Use"
        " `is_articulated `."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return is_articulated(prim)


def is_articulated(prim: Usd.Prim) -> bool:
    """Check if the given prim is an articulated (i.e. object with joints)."""
    return has_api(prim, api_name="PhysicsArticulationRootAPI")


def is_rigid(prim: Usd.Prim) -> bool:
    """Check if the given prim is a rigid (i.e. prim without joints)."""
    return not is_articulated(prim)


def is_robot(prim: Usd.Prim) -> bool:
    """Check if the given prim is a robot (i.e. prim with drive joints)."""
    if not is_entity(prim):
        return False
    stage = prim.GetStage()
    for joint in filter(is_a_drive_joint, stage.Traverse()):
        for link in get_links_for_joint(joint):
            if link is not None and get_path(link).startswith(get_path(prim)):
                return True
    else:
        return False


def is_entity(prim: Usd.Prim) -> bool:
    """Check if the given prim is an entity, a thing with distinct and independent existence.

    Currently, a prim is an entity if it is a child prim of the world and is one of the types below.

    Object types:
        * `UsdGeom.Xform`
        * `UsdGeom.Cube`
        * `UsdGeom.Sphere`
        * `UsdGeom.Mesh`
    """
    if not is_child(prim, get_world_prim(prim.GetStage())):
        return False

    entity_types = [
        UsdGeom.Xform,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Capsule,
        UsdGeom.Cone,
        UsdGeom.Mesh,
    ]
    return any(map(prim.IsA, entity_types))


def is_object(prim: Usd.Prim) -> bool:
    """Check if the given prim is an object, an entity that is not a robot."""
    return is_entity(prim) and not is_robot(prim)


def is_camera(prim: Usd.Prim) -> bool:
    """Check if the given prim is a camera."""
    return prim.IsA(UsdGeom.Camera)


def is_ancestor(prim: Usd.Prim, other_prim: Usd.Prim) -> bool:
    """Check if `prim` is an ancestor prim of `other_prim`."""
    return prim in set(get_ancestor_prims(other_prim))


def is_descendant(prim: Usd.Prim, other_prim: Usd.Prim) -> bool:
    """Check if `prim` is a descendant prim of `other_prim`."""
    return is_ancestor(other_prim, prim)


def get_parent(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Get the parent of prim if it exists."""
    parent_prim = prim.GetParent()
    if not parent_prim.IsValid():
        return None
    return parent_prim


def is_parent(prim: Usd.Prim, other_prim: Usd.Prim) -> bool:
    """Check if `prim` is a parent prim of `other_prim`."""
    return prim == get_parent(other_prim)


def is_child(prim: Usd.Prim, other_prim: Usd.Prim) -> bool:
    """Check if `prim` is a child prim of `other_prim`."""
    return is_parent(other_prim, prim)


def is_sibling(prim: Usd.Prim, other_prim: Usd.Prim) -> bool:
    """Check if `prim` is a sibling prim of `other_prim`."""
    if prim == other_prim:
        return False
    parent_prim = get_parent(prim)
    other_parent_prim = get_parent(other_prim)
    if (parent_prim is None) or (other_parent_prim is None):
        return False
    return parent_prim == other_parent_prim


def is_root(prim: Usd.Prim) -> bool:
    """Check if `prim` is the root prim (i.e. path is "/")."""
    return prim == get_root_prim(prim.GetStage())


def is_visible(prim: Usd.Prim) -> bool:
    """Check if the given prim is visible."""
    is_invisible = UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
    return not is_invisible


def get_body_type(prim: Usd.Prim) -> BodyType:
    """Get the body type of the given prim."""
    return BodyType.RIGID if is_rigid(prim) else BodyType.ARTICULATED


def get_pose_type(prim: Usd.Prim) -> PoseType:
    """Get the pose type of the given prim."""
    return PoseType.FIXED if is_fixed(prim) else PoseType.FLOATING


def get_transform_local(prim: Usd.Prim, time_code: Optional[Usd.TimeCode] = None) -> Affine:
    """Get the transform of the given prim relative to the parent prim.

    Note:
        This is the transform of the prim relative to its parent. This transform is an element of
        GA(3), meaning there may be a scaling of each of the 3 columns in the rotation matrix.

    Args:
        prim: The prim to set the transform for.
        time_code: Time index to get the transform for.

    Returns:
        transform: 4 x 4 transform matrix that represents the prim's transform relative to its
            parent.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()

    xform = UsdGeom.Xformable(prim)
    local_transformation = xform.GetLocalTransformation(time_code)
    # NOTE (roflaherty): The xformOp:transform seems to be messed up. It
    # stores the 4 x 4 transpose matrix as the transpose of what it
    # should be.
    transform = np.array(local_transformation).T
    return transform


# TODO (roflaherty): Remove this function.
def get_pose(prim: Usd.Prim, time_code: Optional[Usd.TimeCode] = None) -> Affine:
    """DEPRECATED: Use `get_transform_local`."""
    msg = (
        "`get_pose` is deprecated and will be removed in a future release. Use"
        " `get_transform_local`."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return get_transform_local(prim)


def get_transform_world(prim: Usd.Prim, time_code: Optional[Usd.TimeCode] = None) -> Affine:
    """Get the transform of the given prim relative to the world frame.

    Note:
        This is the transform of the prim relative to its the world frame. This transform is an
        element of SE(3)-scaled, meaning there may be a scaling of each of the 3 columns in the
        rotation matrix.

    Args:
        prim: The prim to set the transform for.
        time_code: Time index to get the transform for.

    Returns:
        transform: 4 x 4 transform matrix that represents the prim's transform relative to the
            world frame.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()

    xform = UsdGeom.Xformable(prim)
    world_transformation = xform.ComputeLocalToWorldTransform(time_code)
    # NOTE (roflaherty): The xformOp:transform seems to be messed up. It
    # stores the 4 x 4 transpose matrix as the transpose of what it
    # should be.
    transform = np.array(world_transformation).T
    return transform


def get_transform_relative(
    prim: Usd.Prim, other_prim: Usd.Prim, time_code: Optional[Usd.TimeCode] = None
) -> Affine:
    """Get the transform of the given prim relative to the other prim.

    Note:
        This is the transform of the prim relative to the other prim frame. This transform is an
        element of SE(3)-scaled, meaning there may be a scaling of each of the 3 columns in the
        rotation matrix.

    Args:
        prim: The prim to set the transform for.
        other_prim: The other prim that the transform will be relative to.
        time_code: Time index to get the transform for.

    Returns:
        transform: 4 x 4 transform matrix that represents the prim's transform relative to the
            other prim's frame.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()

    world___prim = get_transform_world(prim, time_code)
    world___other = get_transform_world(other_prim, time_code)
    other___world = Transform.inverse(world___other)

    other___prim = other___world @ world___prim

    return other___prim


def set_transform_local(
    prim: Usd.Prim, transform: Affine, time_code: Optional[Usd.TimeCode] = None
) -> None:
    """Set the transform for the given prim relative to the parent prim.

    Args:
        prim: The prim to set the transform for.
        transform: 4 x 4 transform matrix that represents the desired transform.
        time_code: Time index to create animations. If not provide a time index will not set. If
            provided the time index is set and the prim will be put into animation mode (i.e.
            `physics:kinematicsEnabled = True` and `physics:rigidBodyEnabled = False`).
    """
    animation_mode = time_code is not None
    if time_code is None:
        time_code = Usd.TimeCode.Default()

    if prim.GetAttribute("xformOp:transform").IsValid():
        # NOTE (roflaherty): The xformOp:transform seems to be messed up. It
        # stores the 4 x 4 transpose matrix as the transpose of what it
        # should be.
        prim.GetAttribute("xformOp:transform").Set(Gf.Matrix4d(transform.T), time_code)
    else:
        translation = (Transform.get_translation(transform)).tolist()
        rot = Rotation.from_matrix(Transform.get_rotation(transform))
        rot_quat = rot.as_quat()
        rot_quat_real = rot_quat[-1]
        rot_quat_imag = Gf.Vec3f(rot_quat[0:3].tolist())
        orientation = Gf.Quatf(rot_quat_real, rot_quat_imag)

        if prim.GetAttribute("xformOp:translate").IsValid():
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(translation), time_code)
        else:
            UsdGeom.Xformable(prim).AddTranslateOp()
            UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(translation), time_code)

        if prim.GetAttribute("xformOp:orient").IsValid():
            attribute = prim.GetAttribute("xformOp:orient")
            attribute.SetTypeName(Sdf.ValueTypeNames.Quatf)
            attribute.Set(orientation, time_code)
        else:
            orient_attr = UsdGeom.Xformable(prim).AddOrientOp()
            orient_attr.Set(orientation, time_code)

    if animation_mode:
        kinematics_enable_attr = prim.GetAttribute("physics:kinematicEnabled")
        if not kinematics_enable_attr.IsValid():
            kinematics_enable_attr = prim.CreateAttribute(
                "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool
            )
        kinematics_enable_attr.Set(True)

        rigid_body_enable_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if not rigid_body_enable_attr.IsValid():
            rigid_body_enable_attr = prim.CreateAttribute(
                "physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool
            )
        rigid_body_enable_attr.Set(False)


# TODO (roflaherty): Remove this function.
def set_pose(prim: Usd.Prim, transform: Pose, time_code: Optional[Usd.TimeCode] = None) -> None:
    """DEPRECATED: Use `set_transform_local`."""
    msg = (
        "`set_pose` is deprecated and will be removed in a future release. Use"
        " `set_transform_local`."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return set_transform_local(prim, transform, time_code)


def set_transform_world(
    prim: Usd.Prim, transform: Affine, time_code: Optional[Usd.TimeCode] = None
) -> None:
    """Set the transform for the given prim relative to the world frame.

    Note:
        This sets the prim's local transform calculated so that it is the correct transform relative
        to the world frame.

    Args:
        prim: The prim to set the transform for.
        transform: 4 x 4 transform matrix that represents the desired transform.
        time_code: Time index to create animations. If not provide a time index will not set. If
            provided the time index is set and the prim will be put into animation mode (i.e.
            `physics:kinematicsEnabled = True` and `physics:rigidBodyEnabled = False`).

    Raises:
        RuntimeError: If the given prim is the world prim (i.e. a prim path of "/").
    """
    if not prim.GetParent().IsValid():
        msg = (
            f"The given prim's ('{get_path(prim)}') parent is not a valid prim, thus its world"
            " transform cannot be set."
        )
        raise RuntimeError(msg)

    world___prim = transform
    world___parent = get_transform_world(prim.GetParent(), time_code)
    parent___world = Transform.inverse(world___parent)
    parent___prim = parent___world @ world___prim

    set_transform_local(prim, parent___prim, time_code)


def set_transform_relative(
    prim: Usd.Prim,
    other_prim: Usd.Prim,
    transform: Affine,
    time_code: Optional[Usd.TimeCode] = None,
) -> None:
    """Set the transform for the given prim relative to the other prim.

    Note:
        This sets the prim's local transform calculated so that it is the correct transform relative
        to the other prim frame.

    Args:
        prim: The prim to set the transform for.
        other_prim: The other prim that the transform will be relative to.
        transform: 4 x 4 transform matrix that represents the desired transform.
        time_code: Time index to create animations. If not provide a time index will not set. If
            provided the time index is set and the prim will be put into animation mode (i.e.
            `physics:kinematicsEnabled = True` and `physics:rigidBodyEnabled = False`).

    Raises:
        RuntimeError: If the given prim is the world prim (i.e. a prim path of "/").
    """
    if not prim.GetParent().IsValid():
        msg = (
            f"The given prim's ('{get_path(prim)}') parent is not a valid prim, thus its relative"
            " transform cannot be set."
        )
        raise RuntimeError(msg)

    other___prim = transform
    parent___other = get_transform_relative(other_prim, prim.GetParent(), time_code)
    parent___prim = parent___other @ other___prim

    set_transform_local(prim, parent___prim, time_code)


def get_twist(prim: Usd.Prim) -> Twist:
    """Get the twist of the given prim.

    Note:
        Currently hardcoded to return the "zero" twist.

    Return:
        Twist(linear=<Vector>, angular=<Vector>)
    """
    return Twist(linear=np.zeros(3), angular=np.zeros(3))


def get_joint_state(prim: Usd.Prim) -> JointState:
    """Get the joint state of the given prim.

    Return:
        JointState(name=<List[str] list of joint names>,
                   position=<Vector joint positions>,
                   velocity=<Vector joint velocities>)

    Raises:
        RuntimeError:
            * If the prim is not articulated.
            * If the length prim's joint names doesn't match the number of joint prims.
            * If the prim's joint names don't match the joint prim names.
    """
    if not is_articulated(prim):
        raise RuntimeError(f"The {get_path(prim)} is not articulated.")

    # Get all child prims that are joints (should be at least one because it is articulated)
    joint_prims = list(filter(is_a_movable_joint, get_joints_for_articulated_root(prim)))

    # Set number of joints
    n_joints = len(joint_prims)

    # Get joint names
    joint_names = [joint_prim.GetName() for joint_prim in joint_prims]

    joint_positions = np.zeros(n_joints)
    if prim.GetAttribute("joint_names").IsValid():
        attr_joint_names = list(prim.GetAttribute("joint_names").Get())
        if n_joints != len(attr_joint_names):
            raise RuntimeError(
                f"Length of {get_path(prim)} prim's 'joint_names' attribute must match"
                f" number of joint prims ({n_joints})"
            )
        if set(joint_names) != set(attr_joint_names):
            raise RuntimeError(
                f"Set of names in {get_path(prim)} prim's 'joint_names' attribute"
                f" ({attr_joint_names}) must match joint prim names ({joint_names})"
            )

        if prim.GetAttribute("joint_positions").IsValid():
            attr_joint_positions = np.array(prim.GetAttribute("joint_positions").Get())
            joint_positions = attr_joint_positions[
                [attr_joint_names.index(name) for name in joint_names]
            ]

    # Get joint velocities
    # TODO (roflaherty): Hardcoding this to zero for now.
    joint_velocities = np.zeros(len(joint_names))

    return JointState(name=joint_names, position=joint_positions, velocity=joint_velocities)


def set_joint_state(prim: Usd.Prim, joint_state: JointState) -> None:
    """Set the joint state of the given prim."""
    # Set `joint_positions` property
    joint_positions = joint_state["position"]
    joint_positions_attr = prim.GetAttribute("joint_positions")
    if not joint_positions_attr.IsValid():
        joint_positions_attr = prim.CreateAttribute(
            "joint_positions", Sdf.ValueTypeNames.FloatArray
        )
    joint_positions_attr.Set(joint_positions)

    # Set `joint_names` property
    joint_names_attr = prim.GetAttribute("joint_names")
    if not joint_names_attr.IsValid():
        joint_names_attr = prim.CreateAttribute("joint_names", Sdf.ValueTypeNames.StringArray)
    joint_names_attr.Set(joint_state["name"])

    # Check that state dict joints match prim joints
    stage = prim.GetStage()
    for name in joint_state["name"]:
        joint_prim = stage.GetPrimAtPath("/".join([get_path(prim), name]))

        if joint_prim is None:
            state_dict_joints = joint_state["name"]
            prim_joint_state = get_joint_state(prim)
            prim_joints = prim_joint_state["name"] if prim_joint_state is not None else None
            msg = "\n".join(
                [
                    "State dict joints do not match prim joints.",
                    f"  State dict joints: {state_dict_joints}",
                    f"  Prim joints: {prim_joints}",
                ]
            )
            raise RuntimeError(msg)


def get_attached_state(
    prim: Usd.Prim, time_code: Optional[Usd.TimeCode] = None
) -> Optional[AttachedState]:
    """Get the attached state for the given prim."""
    parent_path = get_attribute_value(
        prim, attribute_name="AttachedState:parent", time_code=time_code
    )
    if parent_path is None:
        return None
    world_prim = get_world_prim(prim.GetStage())
    relative_path = os.path.relpath(parent_path, get_path(world_prim))
    parts = relative_path.split(os.sep)
    body = parts[0]
    body_path = os.path.join(get_path(world_prim), body)
    frame = None if len(parts) == 1 else parts[-1]
    pose = get_attribute_value(prim, attribute_name="AttachedState:transform", time_code=time_code)
    if pose is None:
        raise RuntimeError(f"Attribute AttachedState:transform is not specified for Prim: {prim}")
    return AttachedState(body=body_path, frame=frame, pose=np.array(pose))


def get_state(prim: Usd.Prim) -> BodyState:
    """Get the state for the given prim."""
    body_state = BodyState(
        body_type=str(get_body_type(prim)),
        pose_type=str(get_pose_type(prim)),
        pose=get_transform_local(prim),
        twist=get_twist(prim),
    )
    try:
        body_state["joint_state"] = get_joint_state(prim)
    except RuntimeError:
        pass
    attached_state = get_attached_state(prim)
    if attached_state is not None:
        body_state["attached_state"] = attached_state
    return body_state


def get_joints_for_articulated_root(
    prim: Usd.Prim, joint_selector_func: Optional[Callable] = None
) -> List[Usd.Prim]:
    """Get all the child joint prims from the given articulated root prim.

    This returns a list of all joints associated with the given articulated object. See
    :func:`~srl.util.prim_helper.get_child_joints_for_link` to get immediate child joint prims for a
    given link prim.

    Args:
        prim: Articulated root prim to get the joints for.
        joint_selector_func: Filter function to select which joints from the articulated root to
        return.
    """
    if joint_selector_func is None:
        joint_selector_func = is_a_joint
    stage = prim.GetStage()
    joint_prims = []
    for joint in filter(joint_selector_func, stage.Traverse()):
        if any(link is not None and is_ancestor(prim, link) for link in get_links_for_joint(joint)):
            joint_prims.append(joint)
    return joint_prims


def get_links_for_articulated_root(
    prim: Usd.Prim, joint_selector_func: Optional[Callable] = None
) -> List[Usd.Prim]:
    """Get all child link prims from the given articulated prim.

    This returns a list of all links associated with the given articulated object. See
    :func:`~srl.util.prim_helper.get_child_links_for_joint` to get immediate child link prims for a
    given joint prim.

    Args:
        prim: Articulated root prim to get the joints for.
        joint_selector_func: Filter function to select which joints from the articulated root to
        return.
    """
    if not is_articulated(prim):
        raise ValueError(
            f"Invalid prim {prim.GetPath()}. Prim must have the 'PhysicsArticulationRootAPI' schema"
            " applied to it."
        )
    stage = prim.GetStage()
    joint_prims = get_joints_for_articulated_root(prim, joint_selector_func)
    link_prims = set()
    for joint_prim in joint_prims:
        joint_api = UsdPhysics.Joint(joint_prim)
        body0_rel_targets = joint_api.GetBody0Rel().GetTargets()
        body1_rel_targets = joint_api.GetBody1Rel().GetTargets()
        if len(body0_rel_targets) > 1 or len(body1_rel_targets) > 1:
            raise NotImplementedError(
                "`get_links_for articulated_root` does not currently handle more than one relative"
                f" body target in the joint. joint_prim: {joint_prim}, body0_rel_targets:"
                f" {body0_rel_targets}, body1_rel_targets: {body1_rel_targets}"
            )
        for target in body0_rel_targets:
            if target != stage.GetDefaultPrim().GetPrimPath():
                link_prims.add(stage.GetPrimAtPath(target))
        for target in body1_rel_targets:
            if target != stage.GetDefaultPrim().GetPrimPath():
                link_prims.add(stage.GetPrimAtPath(target))
    return list(link_prims)


def get_child_joints_for_link(prim: Usd.Prim) -> List[Usd.Prim]:
    """Get all immediate child joint prims from the given link prim.

    Note:
        This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
        that the parent link is specified in `body0_rel_targets` and the child links is specified
        in `body1_rel_targets`.
    """
    all_joint_prims = get_joints_for_articulated_root(prim.GetParent())
    child_joint_prims = []
    for joint_prim in all_joint_prims:
        joint_api = UsdPhysics.Joint(joint_prim)
        body0_rel_targets = joint_api.GetBody0Rel().GetTargets()
        if prim.GetPath() in body0_rel_targets:
            child_joint_prims.append(joint_prim)
    return child_joint_prims


def get_parent_joint_for_link(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Get the parent joint prim for the given link prim.

    Note:
        This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
        that the parent link is specified in `body0_rel_targets` and the child links is specified
        in `body1_rel_targets`.
    """
    all_joint_prims = get_joints_for_articulated_root(prim.GetParent())
    for joint_prim in all_joint_prims:
        joint_api = UsdPhysics.Joint(joint_prim)
        body0_rel_targets = joint_api.GetBody0Rel().GetTargets()
        body1_rel_targets = joint_api.GetBody1Rel().GetTargets()
        if len(body1_rel_targets) == 0:
            if prim.GetPath() in body0_rel_targets:
                return get_world_prim(prim.GetStage())
            else:
                continue
        if prim.GetPath() in body1_rel_targets:
            return joint_prim
    else:
        return None


def get_links_for_joint(prim: Usd.Prim) -> Tuple[Optional[Usd.Prim], Optional[Usd.Prim]]:
    """Get all link prims from the given joint prim.

    Note:
        This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
        that the parent link is specified in `body0_rel_targets` and the child links is specified
        in `body1_rel_targets`.
    """
    stage = prim.GetStage()
    joint_api = UsdPhysics.Joint(prim)

    rel0_targets = joint_api.GetBody0Rel().GetTargets()
    if len(rel0_targets) > 1:
        raise NotImplementedError(
            "`get_links_for_joint` does not currently handle more than one relative"
            f" body target in the joint. joint_prim: {prim}, body0_rel_targets:"
            f" {rel0_targets}"
        )
    link0_prim = None
    if len(rel0_targets) != 0:
        link0_prim = stage.GetPrimAtPath(rel0_targets[0])

    rel1_targets = joint_api.GetBody1Rel().GetTargets()
    if len(rel1_targets) > 1:
        raise NotImplementedError(
            "`get_links_for_joint` does not currently handle more than one relative"
            f" body target in the joint. joint_prim: {prim}, body1_rel_targets:"
            f" {rel0_targets}"
        )
    link1_prim = None
    if len(rel1_targets) != 0:
        link1_prim = stage.GetPrimAtPath(rel1_targets[0])

    return (link0_prim, link1_prim)


def get_parent_link_for_joint(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Get the parent link prim for the given joint prim.

    Note:
        This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
        that the parent link is specified in `body0_rel_targets` and the child links is specified
        in `body1_rel_targets`.
    """
    return get_links_for_joint(prim)[0]


def get_child_link_for_joint(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Get the child link prim from the given joint prim.

    Note:
        This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
        that the parent link is specified in `body0_rel_targets` and the child links is specified
        in `body1_rel_targets`.
    """
    return get_links_for_joint(prim)[1]


def get_subtree_links(link_prim: Usd.Prim) -> Iterable[Usd.Prim]:
    """Get the kinematic subtree rooted at a link prim using a preorder traversal.

    Note:
        This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
        that the parent link is specified in `body0_rel_targets` and the child links is specified
        in `body1_rel_targets`.
    """
    yield link_prim
    yield from get_descendant_links(link_prim)


def get_descendant_links(link_prim: Usd.Prim) -> Iterable[Usd.Prim]:
    """Get the kinematic descendants of a link prim using a preorder traversal.

    Note:
        This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
        that the parent link is specified in `body0_rel_targets` and the child links is specified
        in `body1_rel_targets`.
    """
    for joint_prim in get_child_joints_for_link(link_prim):
        child_prim = get_child_link_for_joint(joint_prim)
        if child_prim is not None:
            yield child_prim
            yield from get_descendant_links(child_prim)


def get_ancestor_links(link_prim: Usd.Prim) -> Iterable[Usd.Prim]:
    """Get the kinematic ancestors of a link prim.

    Note:
        This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
        that the parent link is specified in `body0_rel_targets` and the child links is specified
        in `body1_rel_targets`.
    """
    joint_prim = get_parent_joint_for_link(link_prim)
    if (joint_prim is None) or not joint_prim.IsValid():
        return
    parent_prim = get_parent_link_for_joint(joint_prim)
    if parent_prim is None:
        return
    yield parent_prim
    yield from get_ancestor_links(parent_prim)


class JointAxis(Enum):
    """Enum to specify a joint axis."""

    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)


def get_joint_axis(prim: Usd.Prim) -> Optional[JointAxis]:
    """Get the joint axis for the given joint prim.

    Returns:
        The joint axis as a `JointAxis` enum if it is set, otherwise returns `None`.

    Raises:
        RuntimeError if the joint axis is not `None`, "X", "Y", or "Z".
    """
    axis_name = prim.GetAttribute("physics:axis").Get()
    if axis_name is None:
        return None
    elif axis_name == "X":
        return JointAxis.X
    elif axis_name == "Y":
        return JointAxis.Y
    elif axis_name == "Z":
        return JointAxis.Z
    else:
        msg = f"Unsupported joint axis '{axis_name}'. Valid joint axes: 'X', 'Y', 'Z'."
        raise RuntimeError(msg)


def get_joint_limits(prim: Usd.Prim) -> Tuple[Optional[float], Optional[float]]:
    """Get the joint limits for the given joint prim.

    Returns:
        A tuple with the lower limit value in the first element and the upper limit value in the
        second element. Either value may be `None` if it is not set.
    """
    lower_limit = get_attribute_value(prim, "physics:lowerLimit")
    upper_limit = get_attribute_value(prim, "physics:upperLimit")
    return (lower_limit, upper_limit)


def get_joint_velocity_limit(prim: Usd.Prim) -> Optional[float]:
    """Get the joint velocity limit for the given joint prim.

    Returns:
        The velocity limit as a float. It will return `None` if it is not set.
    """
    return get_attribute_value(prim, "physxJoint:maxJointVelocity")


def get_joint_force_limit(prim: Usd.Prim) -> Optional[float]:
    """Get the joint force limit for the given joint prim.

    Returns:
        The force limit as a float. It will return `None` if it is not set.
    """
    if is_a_revolute_joint(prim):
        return get_attribute_value(prim, "drive:angular:physics:maxForce")
    elif is_a_prismatic_joint(prim):
        return get_attribute_value(prim, "drive:linear:physics:maxForce")
    else:
        msg = (
            f"Invalid prim type: {prim.GetTypeName()}. Only revolute and prismatic joints are"
            " supported."
        )
        raise RuntimeError(msg)


def get_joint_transform(prim: Usd.Prim, body_rel: int = 0) -> Pose:
    """Get the transform for the given joint prim.

    Args:
        prim: Joint prim to get the transform from.
        body_rel: Determines if the transform is relative to body 0 or body 1.

    Returns:
        Returns the 4 x 4 transform matrix.
    """
    if body_rel not in [0, 1]:
        raise ValueError(f"`body_rel` ({body_rel}) must be either 0 or 1")
    pos_val = prim.GetAttribute("physics:localPos" + str(body_rel)).Get()
    rot_val = prim.GetAttribute("physics:localRot" + str(body_rel)).Get()

    w = rot_val.GetReal()
    x, y, z = np.array(rot_val.GetImaginary())
    rot = Rotation.from_quat([x, y, z, w])
    xform = Transform.from_rotmat(rot.as_matrix(), np.array(pos_val))
    return xform


def get_link_geometry(prim: Usd.Prim) -> Dict[str, np.ndarray]:
    """Return the geometry elements (vertices and faces) for the given prim.

    Args:
        prim: The prim to get the geometry elements for. It must be a `UsdGeom.Mesh` prim.

    Returns:
        The geometry components (vertices and faces) stored in a dict.
            "vertices": Vertices stored as a numpy array.
            "faces": Faces stored as a numpy array.

    Raises:
        RuntimeError if the prim is not a `UsdGeom.Mesh` prim.
    """
    if not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("Invalid prim type. Prim must be of type `UsdGeom.Mesh`.")
    faces = prim.GetAttribute("faceVertexIndices").Get()
    faces = np.array(faces, dtype=np.int32).reshape(-1, 3)
    vertices = prim.GetAttribute("points").Get()
    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    return dict(vertices=vertices, faces=faces)


def get_bounds(root_prim: Usd.Prim, time_code: Optional[Usd.TimeCode] = None) -> Interval:
    """Get the world axis-aligned bounds of geometry rooted at a prim.

    Args:
        root_prim: The root prim of the subtree.
        time_code: The time index to get the bounds for.

    Returns:
        The lower and upper world axis-aligned bounds of the subtree's geometry.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    prim_bbox = bbox_cache.ComputeWorldBound(root_prim)
    prim_range = prim_bbox.ComputeAlignedRange()
    lower = np.array(prim_range.GetMin())
    upper = np.array(prim_range.GetMax())
    return lower, upper


# TODO (roflaherty): Remove this function.
def get_prim_bounds(prim: Usd.Prim) -> Interval:
    """DEPRECATED: Use `get_bounds`."""
    msg = (
        "`get_prim_bounds` is deprecated and will be removed in a future release. Use `get_bounds`."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return get_bounds(prim)


def get_geom_subsets_from_mesh(prim: Usd.Prim) -> List[Usd.Prim]:
    """Get a list of all GeomSubset prims under a given mesh prim.

    Args:
        prim: The USD prim of the mesh.

    Returns:
        A list of all GeomSubset prims under the mesh.
    """
    # Check if the prim is a mesh prim
    if not prim.IsA(UsdGeom.Mesh):
        msg = f"The given prim must be a `USDGeom.Mesh`. It is `{prim.GetTypeName()}`."
        raise RuntimeError(msg)

    # List to hold the found GeomSubset prims
    geom_subsets = []

    # Iterate over all the child prims of the mesh prim
    for child in prim.GetChildren():
        # Check if the child is a GeomSubset
        if UsdGeom.Subset(child):
            geom_subsets.append(child)

    return geom_subsets


def export_geometry_as_obj_file(
    prim: Usd.Prim, obj_output_path: PathLike, transform: Optional[Vector] = None
) -> None:
    """Export the prim's geometry as an OBJ file.

    Args:
        prim: The mesh prim to export the geometry for.
        obj_output_path: The path to where the new OBJ file will be saved (note: the file extension
            should be '.obj').
        transform: A 4x4 homogenous transform to apply to each point in the mesh. Defaults
            to identity.
    """

    @dataclass
    class GeomData:
        """Helper class for storing USD geometry data."""

        name: str
        face_indices: np.ndarray
        material_prim: Optional[Usd.Prim] = None
        mtl_str: Optional[str] = None

    # Check if the prim is a mesh prim
    if not prim.IsA(UsdGeom.Mesh):
        msg = f"The given prim must be a `USDGeom.Mesh`. It is `{prim.GetTypeName()}`."
        raise RuntimeError(msg)

    # Set default transform value
    if transform is None:
        transform = Transform.identity()
    assert (4, 4) == np.array(transform).shape

    obj_output_path = Path(obj_output_path)

    face_vertex_cnts = np.array(get_attribute_value(prim, "faceVertexCounts"), dtype=np.int32)
    face_vertex_inds = np.array(get_attribute_value(prim, "faceVertexIndices"), dtype=np.int32)
    if np.sum(face_vertex_cnts) != len(face_vertex_inds):
        msg = (
            f"Sum of `faceVertexCounts` ({np.sum(face_vertex_cnts)}) does not equal length of"
            f" `faceVertexIndices` ({len(face_vertex_inds)}) for prim '{get_prim_path(prim)}'."
        )
        raise RuntimeError(msg)
    points = np.array(prim.GetAttribute("points").Get(), dtype=np.float32)
    if points.shape[1] != 3:
        msg = (
            f"Dimension of `points` ({points.shape[1]}) is not 3 for prim '{get_prim_path(prim)}'."
        )
        raise RuntimeError(msg)

    # Get GeomSubset prims
    geom_subset_prims = get_geom_subsets_from_mesh(prim)

    geom_data_list: List[GeomData] = []

    has_material = False
    if len(geom_subset_prims) > 0:
        for geom_subset_prim in geom_subset_prims:
            name = geom_subset_prim.GetName()
            face_indices = np.array(
                get_attribute_value(geom_subset_prim, "indices"), dtype=np.int32
            )
            material_prim = get_material_prim(geom_subset_prim)
            mtl_str = None
            if material_prim is not None:
                has_material = True
                mtl_str = get_material_mtl_str(material_prim)
            geom_data = GeomData(
                name=name, face_indices=face_indices, material_prim=material_prim, mtl_str=mtl_str
            )
            geom_data_list.append(geom_data)
    else:
        # If there are no geom_subsets, create a subset that consists of all the faces
        name = prim.GetName()
        face_indices = np.arange(0, len(face_vertex_cnts))
        material_prim = get_material_prim(prim)
        mtl_str = None
        if material_prim is not None:
            has_material = True
            mtl_str = get_material_mtl_str(material_prim)
        geom_data = GeomData(
            name=name, face_indices=face_indices, material_prim=material_prim, mtl_str=mtl_str
        )
        geom_data_list.append(geom_data)
    # Check that the sum of the lengths of all the subset face indices equal the total number of
    # faces
    sum_geo_data_face_indices = np.sum([len(geo_data.face_indices) for geo_data in geom_data_list])
    if len(face_vertex_cnts) != sum_geo_data_face_indices:
        has_material = False
        msg = (
            f"Sum of `faceVertexCounts` ({np.sum(face_vertex_cnts)}) does not equal sum of length"
            f" of `GeomSubset` `indices` ({sum_geo_data_face_indices}) for"
            f" prim '{get_prim_path(prim)}'. Material mtl files will not be created."
        )
        warnings.warn(msg, stacklevel=2)

    # Set mesh directory
    mesh_dir_path = obj_output_path.parent
    mesh_dir_path.mkdir(parents=True, exist_ok=True)

    # Create `.mtl` file if the object has material
    if has_material:
        mtl_output_dir = Path(obj_output_path).parent
        mtl_filename = f"{obj_output_path.stem}.mtl"
        mtl_file_path = mtl_output_dir / mtl_filename

        mtl_content = []

        for geom_data in geom_data_list:
            if geom_data.mtl_str is not None:
                mtl_content.append(geom_data.mtl_str)
            mtl_content.append("")

        with open(str(mtl_file_path), "w") as mtl_file:
            mtl_file.write("\n".join(mtl_content))

    with open(str(obj_output_path), "w") as fout:
        # Write the 'mtllib' line if there is a material
        if has_material:
            fout.write(f"mtllib {mtl_filename}\n")

        # Write vertex positions
        for i in range(points.shape[0]):
            transformed_point = Transform.transform_vector(np.array(transform), points[i, :])
            vertex_str = f"v {transformed_point[0]} {transformed_point[1]} {transformed_point[2]}"
            fout.write(f"{vertex_str}\n")

        # Build face data array
        mark = 0
        face_vertex_data = []
        for cnt in face_vertex_cnts:
            vertex_inds_set = face_vertex_inds[mark : (mark + cnt)]
            # NOTE: OBJ file indexing starts at 1
            vertex_inds_set_at_1 = vertex_inds_set + 1
            face_vertex_data.append(vertex_inds_set_at_1)
            mark += cnt

        # Write face for each subset
        for geom_data in geom_data_list:
            # Write 'usemtl' line if a material is available
            if geom_data.material_prim is not None:
                fout.write(f"usemtl {geom_data.material_prim.GetName()}\n")

            for face_ind in geom_data.face_indices:
                vertex_inds_set_at_1 = np.array(face_vertex_data[face_ind], dtype=np.int32)
                vertex_inds_set_as_str = " ".join(map(str, vertex_inds_set_at_1.tolist()))
                face_str = f"f {vertex_inds_set_as_str}"
                fout.write(f"{face_str}\n")


def get_material_prim(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Get the material prim bound to the given mesh prim or subset prim.

    Args:
        prim: The mesh prim to get the material prim for.

    Returns:
        Material prim if there is a material bound to the given prim, otherwise returns
        `None`.
    """
    # Check if the prim is a mesh prim
    if not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdGeom.Subset):
        msg = (
            "The given prim must be a `UsdGeom.Mesh` or `UsdGeom.Subset`. It is"
            f" `{prim.GetTypeName()}`."
        )
        raise RuntimeError(msg)

    # Get the material bound to the mesh prim
    material_binding_api = UsdShade.MaterialBindingAPI(prim)
    bound_material_tuple = material_binding_api.ComputeBoundMaterial()

    # Check if the bound material tuple contains a valid UsdShade.Material
    bound_material = bound_material_tuple[0]

    if not bound_material:
        return None

    # Get the UsdShade material prim
    material_prim = bound_material.GetPrim()

    return material_prim


def get_material_mtl_str(prim: Usd.Prim) -> str:
    """Get the mlt string for a given material prim.

    Args:
        prim: The USD material prim for which the material will be extracted.

    Returns:
        The string that would go under the `usemtl` in an `.mtl` file for the given USD material.

    Raises:
        RuntimeError: If the given prim is not UsdShade.Material prim.
    """
    # Check if the prim is a material prim
    if not prim.IsA(UsdShade.Material):
        msg = f"The given prim must be a `UsdShade.Material`. It is `{prim.GetTypeName()}`."
        raise RuntimeError(msg)

    # Get the material object and set file names
    material = UsdShade.Material(prim)
    material_name = prim.GetName()

    # Initialize an empty list to store the content of the .mtl file
    mtl_content = []

    # Add 'newmtl' line to the mtl content
    mtl_content.append(f"newmtl {material_name}")

    # Get the surface output object
    surface_output = material.GetSurfaceOutput()

    # Find the shader connected to the material's surface output
    connected_source = surface_output.GetConnectedSource()
    if not connected_source:
        # Check if it's an MDL material first
        mdl_surface_output = material.GetOutput("mdl:surface")
        if mdl_surface_output:
            connected_source = mdl_surface_output.GetConnectedSource()
        else:
            msg = f"No valid shader connected to the surface output for material: {prim.GetPath()}"
            raise RuntimeError(msg)

    shader_prim = connected_source[0].GetPrim()
    shader = UsdShade.Shader(shader_prim)

    # Get diffuse color (default to white if not available)
    diffuse_input = shader.GetInput("diffuseColor")
    if not diffuse_input:
        diffuse_input = shader.GetInput("diffuse_color_constant")

    if diffuse_input:
        diffuse_color = diffuse_input.Get()
        if isinstance(diffuse_color, Gf.Vec3f):  # Check if it's a Gf.Vec3f type
            mtl_content.append(f"Kd {diffuse_color[0]} {diffuse_color[1]} {diffuse_color[2]}")
        else:
            mtl_content.append("Kd 1.0 1.0 1.0")  # Default to white
    else:
        mtl_content.append("Kd 1.0 1.0 1.0")  # Default to white

    # Set 'Ka' based on the diffuse color (or use a default value if not found)
    # TODO (roflaherty): Need to figure out how to get the ambient color from the shader
    ambient_input = False
    if ambient_input:
        # TODO (roflaherty): Need figure out what to put here
        ambient_color = None
        if isinstance(ambient_color, Gf.Vec3f):
            mtl_content.append(f"Ka {ambient_color[0]} {ambient_color[1]} {ambient_color[2]}")
    else:
        mtl_content.append("Ka 1.0 1.0 1.0")  # Default white

    # Get specular color (default to gray if not available)
    specular_input = shader.GetInput("specularColor")
    if specular_input:
        specular_color = specular_input.Get()
        if specular_color and isinstance(specular_color, tuple) and len(specular_color) == 3:
            mtl_content.append(f"Ks {specular_color[0]} {specular_color[1]} {specular_color[2]}")
        else:
            mtl_content.append("Ks 0.5 0.5 0.5")  # Default to gray
    else:
        mtl_content.append("Ks 0.5 0.5 0.5")  # Default to gray

    # Get roughness and calculate shininess (Ns in .mtl format)
    roughness_input = shader.GetInput("roughness")
    if not roughness_input:
        roughness_input = shader.GetInput("reflection_roughness_constant")

    if roughness_input:
        roughness_value = roughness_input.Get()
        if roughness_value is not None:
            # NOTE (roflaherty): This a heuristic approximation used to convert roughness (a common
            # property in physically based rendering (PBR) materials) into shininess, which is used
            # in the legacy Phong shading model for .mtl files. Here's the reasoning behind this
            # conversion:
            #
            # 1. Phong Model (Shininess in .mtl Format):
            #  In the .mtl file format (used for OBJ models), the Ns (shininess) parameter is part
            #  of the Phong reflection model, which controls the specular highlight size: Higher
            #  values of Ns produce smaller, sharper specular highlights (smooth surfaces).  Lower
            #  values produce broader, softer highlights (rough surfaces).  The range of shininess
            #  (Ns) in the .mtl format is typically between 0 and 1000, but values are commonly kept
            #  below 128 for most uses. This controls how "focused" the specular reflection is.
            #
            # 2. Roughness in PBR: In physically based rendering (PBR) models:
            #  Roughness (often a value between 0.0 and 1.0) defines how rough or smooth a surface
            #  is.  A value of 0.0 means the surface is perfectly smooth (sharp reflections, like a
            #  mirror).  A value of 1.0 means the surface is very rough (diffuse reflections, like
            #  matte materials).  The roughness parameter in PBR is inversely related to shininess:
            #  smoother surfaces (low roughness) have more focused specular reflections, while
            #  rougher surfaces (high roughness) have more diffused, broader reflections.
            shininess = 128 * (1.0 - roughness_value)  # Convert roughness to shininess
            mtl_content.append(f"Ns {shininess}")
        else:
            mtl_content.append("Ns 10.0")  # Default shininess
    else:
        mtl_content.append("Ns 10.0")  # Default shininess

    # Set 'Tf' based on transmission or opacity if available
    transmission_input = shader.GetInput("transmission")
    if transmission_input:
        transmission_color = transmission_input.Get()
        if isinstance(transmission_color, Gf.Vec3f):  # Transmission color in Vec3f
            mtl_content.append(
                f"Tf {transmission_color[0]} {transmission_color[1]} {transmission_color[2]}"
            )
        elif isinstance(transmission_color, float):  # Transmission scalar (same for RGB)
            mtl_content.append(f"Tf {transmission_color} {transmission_color} {transmission_color}")
    else:
        mtl_content.append("Tf 1.0 1.0 1.0")  # Default: fully transparent to light

    # Add transparency (if available)
    opacity_input = shader.GetInput("opacity")
    if opacity_input:
        opacity_value = opacity_input.Get()
        if opacity_value is not None:
            mtl_content.append(f"d {opacity_value}")  # Transparency in .mtl (1.0 = opaque)
        else:
            mtl_content.append("d 1.0")  # Default to fully opaque
    else:
        mtl_content.append("d 1.0")  # Default to fully opaque

    # Set texture map for diffuse color if available
    diffuse_texture_input = shader.GetInput("diffuse_texture")
    if diffuse_texture_input and diffuse_texture_input.HasConnectedSource():
        connected_source = diffuse_texture_input.GetConnectedSource()
        texture_file = (
            connected_source[0].GetPrim().GetAttribute("inputs:file").Get()
        )  # Assuming texture source is file-based
        if texture_file:
            mtl_content.append(f"map_Kd {texture_file}")  # Path to diffuse texture

    # Set texture map for specular color if available
    specular_texture_input = shader.GetInput("specular_texture")
    if specular_texture_input and specular_texture_input.HasConnectedSource():
        connected_source = specular_texture_input.GetConnectedSource()
        texture_file = connected_source[0].GetPrim().GetAttribute("inputs:file").Get()
        if texture_file:
            mtl_content.append(f"map_Ks {texture_file}")  # Path to specular texture

    # Set texture map for opacity if available
    opacity_texture_input = shader.GetInput("opacity_texture")
    if opacity_texture_input and opacity_texture_input.HasConnectedSource():
        connected_source = opacity_texture_input.GetConnectedSource()
        texture_file = connected_source[0].GetPrim().GetAttribute("inputs:file").Get()
        if texture_file:
            mtl_content.append(f"map_d {texture_file}")  # Path to opacity texture

    mtl_str = "\n".join(mtl_content)

    return mtl_str
