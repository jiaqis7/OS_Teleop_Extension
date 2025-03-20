from scipy.spatial.transform import Rotation as R
import numpy as np


def transformation_matrix_to_pose(transformation_matrix):
    """
    Convert a 4x4 transformation matrix to position and quaternion.

    Args:
        transformation_matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        tuple: Position array (x, y, z) and quaternion array (w, x, y, z).
    """
    # Extract the translation part
    position = transformation_matrix[:3, 3]

    # Extract the rotation part and convert to quaternion
    rotation_matrix = transformation_matrix[:3, :3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    # Reorder quaternion to (w, x, y, z)
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

    return position, quaternion


def pose_to_transformation_matrix(position, quaternion):
    """
    Convert position and quaternion to a 4x4 transformation matrix.

    Args:
        position (np.ndarray): Position array (x, y, z).
        quaternion (np.ndarray): Quaternion array (w, x, y, z).

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)

    # Set the translation part
    transformation_matrix[:3, 3] = position

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()

    # Set the rotation part
    transformation_matrix[:3, :3] = rotation_matrix

    return transformation_matrix