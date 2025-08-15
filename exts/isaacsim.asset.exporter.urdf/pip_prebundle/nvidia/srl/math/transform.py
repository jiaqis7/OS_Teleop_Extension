# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""SRL general utility / helper functions for transforms (4 x 4 homogeneous transform matrices)."""

# Standard Library
from typing import Any, List, Optional, Union

# Third Party
import numpy as np
from scipy.spatial.transform import Rotation

# NVIDIA
from nvidia.srl.basics.types import Affine, Vector

# NOTE: This interface is inspired by the Scipy Spatial Transformations Rotation interface.
# Documentation is here:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html


class Transform:
    """Transform in 3 dimensions (i.e. element of GA(3) - general affine group of dimension 3).

    Note:
    A transform is a 4 x 4 Numpy array. This class just provides helper functions for
    creating and manipulating these 4 x 4 numpy arrays. It is not meant to be instantiated as a
    stand alone type.

    Perhaps in the future this class can be refactored into an object that is meant to be
    instantiated. One option for that is to have it inherit from `numpy.ndarray`.
    Reference:
    * https://numpy.org/doc/stable/user/basics.subclassing.html
    """

    @staticmethod
    def identity() -> Affine:
        """Return identity transform matrix."""
        return np.eye(4)

    @staticmethod
    def random(random_scale: bool = False) -> Affine:
        """Return a random 4 x 4 SE(3) transform matrix."""
        transform = np.eye(4)
        if random_scale:
            scale = np.diag(np.random.uniform(0, 2, size=3))
        else:
            scale = np.eye(3)
        transform[0:3, 0:3] = Rotation.random().as_matrix() @ scale
        transform[0:3, [3]] = np.random.uniform(-1, 1, size=(3, 1))
        return transform

    @staticmethod
    def get_translation(transform: Affine) -> np.ndarray:
        """Return the translation component of the given 4 x 4 transform matrix.

        Args:
            transform: A 4 x 4 matrix transform to extract the translation portion from.

        Returns:
            A 3 element vector that is the translational portion of the given transform.
        """
        return np.squeeze(transform[0:3, 3:4])

    @staticmethod
    def from_translation(translation: Vector) -> Affine:
        """Construct a transform matrix that is a pure translation.

        The rotation portion of the transform is set to identity.

        Args:
            translation: A 3-dimensional vector that defines the translational portion of the
                transform matrix.

        Returns:
            A 4 x 4 matrix containing a pure translation transform.
        """
        transform = np.eye(4)
        transform[0:3, 3:4] = Transform._format_vector3d(translation)
        return transform

    @staticmethod
    def set_translation(transform: Affine, translation: Vector) -> None:
        """Set the translation portion of the transform with the given `translation` vector.

        Args:
            transform: The transform to update.
            translation: The new translation vector to update the transform with.
        """
        transform[0:3, 3] = translation

    @staticmethod
    def apply_translation(transform: Affine, translation: Vector) -> None:
        """Apply the given translation to the given transform.

        Think of this like `T += v` where `T` is the transform and `v` is the translation vector.

        Args:
            transform: The transform to update.
            translation: The translation vector to apply to the transform.
        """
        transform[0:3, 3] += translation

    @staticmethod
    def get_rotation(
        transform: Affine, as_quat: bool = False, as_wxyz: bool = False, as_rpy: bool = False
    ) -> np.ndarray:
        """Return the rotation component of the given 4 x 4 transform matrix.

        Note:
            This returns the rotation portion of the transform matrix without the scaling factor.

        Args:
            transform: A 4 x 4 matrix transform to extract the rotation portion from.
            as_quat: If true, return the rotation as quaternion, else as a 3 x 3 matrix.
            as_wxyz: If true and if `as_quat` is true, then return the quaternion in wxyz format.
            as_rpy: If true, then return the rotation as Euler angles in radians as fixed XYZ
                transform; or the "roll" angle around fixed X-axis, then "pitch" angle around the
                fixed Y-axis, finally "yaw" around the fixed Z-axis.

        Returns:
            The rotation portion of the given transform as either a 3 x 3 rotation matrix if
            `as_quat` is false, or a 4-dimensional vector quaternion in the scalar-last format (x,
            y, z, w) if `as_quat` is true and `as_wxyz` is false, or scalar-first format (w, x, y,
            z) if `as_quat` is true and `as_wxyz` is true, or a 3-dimensional vector as the roll,
            pitch, yaw angles if `as_rpy` is true.
        """
        scale = Transform.get_scale(transform)
        rot_mat_w_scale = transform[0:3, 0:3]
        rot_mat = rot_mat_w_scale @ np.diag(1 / scale)
        if as_quat:
            val = Rotation.from_matrix(rot_mat).as_quat()
            if as_wxyz:
                val = np.roll(val, shift=1)
        elif as_rpy:
            val = Rotation.from_matrix(rot_mat).as_euler("xyz", degrees=False)
        else:
            val = rot_mat

        return val

    @staticmethod
    def from_rotation(rotation: Rotation, translation: Optional[Vector] = None) -> Affine:
        """Initialize a transform matrix from a `Rotation` object.

        Args:
            rotation: A `Rotation` object that holds the desired rotation data.
            translation: A 3-dimensional vector that defines the translational portion of the
                transform matrix. Defaults to all zeros.

        Returns:
            A 4 x 4 matrix containing a pure rotation transform.
        """
        rot = rotation.as_matrix()
        trans = Transform._format_vector3d(translation)
        return np.block([[rot, trans], [0, 0, 0, 1]])

    @staticmethod
    def set_rotation(transform: Affine, rotation: Rotation) -> None:
        """Set the rotation portion of the transform with the given `Rotation` object.

        Args:
            transform: The transform to update.
            rotation: The new `Rotation` object to update the transform with.
        """
        transform[0:3, 0:3] = Transform.from_rotation(rotation)[0:3, 0:3]

    @staticmethod
    def from_rotvec(
        rotvec: Vector,
        translation: Optional[Vector] = None,
        degrees: bool = False,
    ) -> Affine:
        """Initialize a transform matrix from a rotation vector.

        Args:
            rotvec: A 3-dimensional vector that is co-directional to the axis of rotation and whose
                norm gives the angle of rotation. This rotation defines the rotational portion of
                the transform matrix.
            translation: A 3-dimensional vector that defines the translational portion of the
                transform matrix. Defaults to all zeros.
            degrees: If True, then the given magnitudes are assumed to be in degrees.
                (Default: False).

        Returns:
            A 4 x 4 matrix containing a transform defined by the input arguments.
        """
        rot = Rotation.from_rotvec(rotvec, degrees=degrees).as_matrix()
        trans = Transform._format_vector3d(translation)
        return np.block([[rot, trans], [0, 0, 0, 1]])

    @staticmethod
    def set_rotvec(transform: Affine, rotvec: Vector, degrees: bool = False) -> None:
        """Set the rotation portion of the transform with the given rotation vector.

        Args:
            transform: The transform to update.
            rotvec: A 3-dimensional vector that is co-directional to the axis of rotation and whose
                norm gives the angle of rotation. This rotation defines the rotational portion of
                the transform matrix.
            degrees: If True, then the given magnitudes are assumed to be in degrees.
                (Default: False).
        """
        transform[0:3, 0:3] = Transform.from_rotvec(rotvec, degrees=degrees)[0:3, 0:3]

    @staticmethod
    def from_rotmat(
        rotmat: np.ndarray, translation: Optional[Union[np.ndarray, List[float]]] = None
    ) -> Affine:
        """Initialize a transform matrix from a rotation matrix.

        Note:
            If `rotmat` is not a valid rotation matrix, it will be projected to the closest rotation
            matrix.

        Args:
            rotmat: A 3 x 3 rotation matrix that is co-directional to the axis of rotation and whose
                norm gives the angle of rotation. This rotation defines the rotational portion of
                the transform matrix.
            translation: A 3-dimensional vector that defines the translational portion of the
                transform matrix. Defaults to all zeros.

        Returns:
            A 4 x 4 matrix containing a transform defined by the input arguments.
        """
        rot = Rotation.from_matrix(rotmat).as_matrix()
        trans = Transform._format_vector3d(translation)
        return np.block([[rot, trans], [0, 0, 0, 1]])

    @staticmethod
    def set_rotmat(transform: Affine, rotmat: np.ndarray) -> None:
        """Set the rotation portion of the transform with the given rotation matrix.

        Note:
            If `rotmat` is not a valid rotation matrix, it will be projected to the closest rotation
            matrix.

        Args:
            transform: The transform to update.
            rotmat: A 3 x 3 rotation matrix that is co-directional to the axis of rotation and whose
                norm gives the angle of rotation. This rotation defines the rotational portion of
                the transform matrix.
        """
        transform[0:3, 0:3] = Transform.from_rotmat(rotmat)[0:3, 0:3]

    @staticmethod
    def from_quat(
        quat: np.ndarray,
        translation: Optional[Union[np.ndarray, List[float]]] = None,
        as_wxyz: bool = False,
    ) -> Affine:
        """Initialize a transform matrix from a rotation represented as quaternion.

        Note:
            If `quat` is not normalized, it will be normalized before initializing the transform.

        Args:
            quat: A 4-dimensional vector quaternion in either the scalar-last format (x, y, z, w) or
                the scalar-first format (w, x, y, z).
            translation: A 3-dimensional vector that defines the translational portion of the
                transform matrix. Defaults to all zeros.
            as_wxyz: If true, `quat` should be in the (w, x, y, z) format.

        Returns:
            A 4 x 4 matrix containing a transform defined by the input arguments.
        """
        if as_wxyz:
            quat = np.roll(quat, shift=-1)
        rot = Rotation.from_quat(quat).as_matrix()
        trans = Transform._format_vector3d(translation)
        return np.block([[rot, trans], [0, 0, 0, 1]])

    @staticmethod
    def set_quat(transform: Affine, quat: np.ndarray, as_wxyz: bool = False) -> None:
        """Set the rotation portion of the transform with the given quaternion.

        Note:
            If `quat` is not normalized, it will be normalized before initializing the transform.

        Args:
            transform: The transform to update.
            quat: A 4-dimensional vector quaternion in either the scalar-last format (x, y, z, w) or
                the scalar-first format (w, x, y, z).
            as_wxyz: If true, `quat` should be in the (w, x, y, z) format.
        """
        if as_wxyz:
            quat = np.roll(quat, shift=-1)
        transform[0:3, 0:3] = Transform.from_quat(quat)[0:3, 0:3]

    @staticmethod
    def from_rpy(
        rpy: Vector, translation: Optional[Vector] = None, degrees: bool = False
    ) -> Affine:
        """Initialize a transform matrix from Euler angles.

        The Euler angles are assumed to be in the fixed XYZ transform convention (i.e. first the
        "roll" angle is applied around the fixed X-axis, then "pitch" angle is applied around the
        fixed Y-axis, and finally "yaw" angle is applied around the fixed Z-axis).

        Args:
            rpy: Euler angles as a fixed XYZ transform.
            degrees: If true, the Euler angles are assumed to be in degrees, otherwise in
                radians.

        Returns:
            A 4 x 4 matrix containing a transform of the Euler angles.
        """
        return Transform.from_rotation(
            Rotation.from_euler("xyz", rpy, degrees=degrees), translation
        )

    @staticmethod
    def get_scale(transform: Affine) -> np.ndarray:
        """Return the scale component of the given 4 x 4 transform matrix.

        Args:
            transform: A 4 x 4 matrix transform to extract the translation portion from.

        Returns:
            A 3 element vector that contain the scaling factors along the orthogonal columns of the
            rotation matrix of the transform.
        """
        rot_w_scaling = transform[0:3, 0:3]
        scale = np.linalg.norm(rot_w_scaling, axis=0)
        return scale

    @staticmethod
    def from_scale(scale: Vector) -> Affine:
        """Construct a transform matrix that is just scaling.

        The rotation portion of the transform is set to identity and the translation portion is set
        to zero.

        Args:
            scale: A 3-dimensional vector that defines the scaling in each axis.

        Returns:
            A 4 x 4 matrix containing a just scaling.
        """
        transform = np.eye(4)
        transform[0:3, 0:3] = np.diag(scale)
        return transform

    @staticmethod
    def remove_scale(transform: Affine) -> np.ndarray:
        """Return a transform with scale factor removed, but the rotation and translation the same.

        Basically this converts from GA(3) to SE(3).

        Args:
            transform: A 4 x 4 matrix GA(3) transform to remove the scale portion from.

        Returns:
            A 4 x 4 matrix SE(3) transform which has equal rotation and translation as the input
            transform.
        """
        rotmat = Transform.get_rotation(transform)
        trans = Transform.get_translation(transform)
        return Transform.from_rotmat(rotmat, trans)

    @staticmethod
    def inverse(transform: Affine) -> Affine:
        """Return the inverse of the given transform.

        Args:
            transform: A 4 x 4 matrix transform to take the inverse of.

        Returns:
            A 4 x 4 matrix transform that is the inverse of the given transform.
        """
        # This works, but it is not as fast as just using np.linalg.inv.
        # On average this takes 0.3 ms where np.linalg.inv takes 0.2 ms.

        # scale = Transform.get_scale(transform)
        # scale_inv = np.diag(1/scale)
        # rot_inv = scale_inv @ Transform.get_rotation(transform).T
        # trans_inv = -1 * (rot_inv @ Transform.get_translation(transform).reshape((3, 1)))
        # return np.block([[rot_inv, trans_inv], [0, 0, 0, 1]])

        return np.linalg.inv(transform)

    @staticmethod
    def is_transform(val: Any) -> bool:
        """Check if the given input value is a valid transform.

        Args:
            val: Any type of object to check if it is a valid transform.

        Returns:
            True if the given input value is a valid transform.
        """
        if not isinstance(val, np.ndarray):
            return False
        if val.shape != (4, 4):
            return False
        if not np.all(val[3, :] == np.array([0, 0, 0, 1])):
            return False
        rot = val[0:3, 0:3]
        if not np.allclose(np.eye(3), rot @ rot.T):
            return False
        return True

    @staticmethod
    def transform_vectors(
        transform: Affine, vectors: Union[List[Vector], np.ndarray]
    ) -> List[Vector]:
        """Pre-multiply a list of vectors by the given transform.

        Args:
            transform: A 4 x 4 matrix transform.
            vectors: A list of 3D vectors.

        Returns:
            The list of vectors transformed by the transformation. The type will match the type of
            the input vectors.
        """
        homogenous_vectors = np.vstack([np.array(vectors).T, np.ones(len(vectors))])
        output_vectors = transform.dot(homogenous_vectors)[:3, :].T
        if isinstance(vectors[0], np.ndarray):
            return output_vectors
        else:
            return output_vectors.tolist()

    @staticmethod
    def transform_vector(transform: Affine, vector: Vector) -> Vector:
        """Pre-multiply a vector by the given transform.

        Args:
            transform: A 4 x 4 matrix transform.
            vector: A 3D vector.

        Returns:
            The vector transformed by the transformation. The type will match the type of the input
            vector.
        """
        return Transform.transform_vectors(transform, [vector])[0]

    @staticmethod
    def _format_vector3d(vector: Optional[Vector] = None) -> np.ndarray:
        """Helper function to format 3D vectors into the correct shape and type.

        Args:
           vector: Vector to reformat.

        Returns:
            3 x 1 numpy array vector with the elements from the vector that was provided.
        """
        if vector is None:
            return np.zeros([3, 1])

        if isinstance(vector, list):
            vector_out = np.array(vector)
        elif isinstance(vector, tuple):
            vector_out = np.array(vector)
        elif isinstance(vector, np.ndarray):
            vector_out = vector
        else:
            raise TypeError(
                f"Invalid argument. Value: {vector} Type: {type(vector)}. Valid types: `None`,"
                " `List`, `np.ndarray`"
            )
        return vector_out.reshape((3, 1))
