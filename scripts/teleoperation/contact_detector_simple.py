import numpy as np
import torch
import json
import os
import time
from datetime import datetime
from scipy.spatial.transform import Rotation


class ContactDetectorSimple:
    def __init__(self, contact_threshold=0.008, z_threshold=0.003, use_z_check=True):
        """
        Initialize contact detector for colored blocks and white square.
        
        The setup:
        - White square: 0.05 x 0.05 x 0.01 m (large flat platform)
        - Colored blocks: 0.02 x 0.004 x 0.004 m (tiny elongated blocks)
        
        Args:
            contact_threshold: Maximum distance to consider as contact (default 8mm)
            z_threshold: Maximum Z distance above white square to consider as contact (default 3mm)
            use_z_check: If True, require blocks to be on top of white square. If False, only check X-Y area.
                        Now defaults to True for proper 3D contact detection.
        """
        self.contact_threshold = contact_threshold
        self.z_threshold = z_threshold
        self.use_z_check = use_z_check
        self.contact_states = {
            "red_white": False,
            "green_white": False,
            "blue_white": False
        }
        self.success = False
        
        # Store the actual dimensions (full size, not half)
        # White square: Should be large platform, checking actual size
        self.white_size = np.array([0.05, 0.05, 0.01])  # 5cm x 5cm x 1cm
        self.colored_size = np.array([0.02, 0.004, 0.004])  # 2cm x 4mm x 4mm
        
        # Store last positions for debugging
        self.last_positions = {}
        
        # Timing tracking
        self.start_time = None  # Will be set on first update
        self.success_time = None
        self.first_contact_times = {
            "red_white": None,
            "green_white": None,
            "blue_white": None
        }
        self.contact_history = []
        self.timing_started = False
        self.success_snapshot = None
        self.success_snapshot = None
    
    def quaternion_to_rotation_matrix(self, quat):
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        # Handle both [w,x,y,z] and [x,y,z,w] conventions
        if len(quat) == 4:
            # Assume [w, x, y, z] format
            r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        else:
            raise ValueError("Quaternion must have 4 elements")
        return r.as_matrix()
    
    def get_oriented_bounding_box_corners(self, center, size, quat):
        """
        Get the 8 corners of an oriented bounding box.
        
        Args:
            center: Center position [x, y, z]
            size: Size of the box [length, width, height]
            quat: Orientation quaternion [w, x, y, z]
            
        Returns:
            corners: 8x3 array of corner positions in world coordinates
        """
        # Create corners in local coordinates (centered at origin)
        half_size = np.array(size) / 2.0
        corners_local = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [ half_size[0], -half_size[1], -half_size[2]],
            [-half_size[0],  half_size[1], -half_size[2]],
            [ half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0], -half_size[1],  half_size[2]],
            [-half_size[0],  half_size[1],  half_size[2]],
            [ half_size[0],  half_size[1],  half_size[2]]
        ])
        
        # Apply rotation
        rot_matrix = self.quaternion_to_rotation_matrix(quat)
        corners_rotated = corners_local @ rot_matrix.T
        
        # Translate to world position
        corners_world = corners_rotated + center
        
        return corners_world
    
    def check_oriented_box_overlap_xy(self, block_pos, block_quat, block_size, 
                                      white_pos, white_quat, white_size):
        """
        Check if an oriented block overlaps with the white square in X-Y plane.
        Uses Separating Axis Theorem (SAT) for accurate oriented bounding box collision.
        
        Args:
            block_pos: Position of colored block CENTER [x, y, z]
            block_quat: Orientation quaternion of block [w, x, y, z]
            block_size: Size of the block [length, width, height]
            white_pos: Position of white square CENTER [x, y, z]
            white_quat: Orientation quaternion of white square [w, x, y, z]
            white_size: Size of white square [length, width, height]
            
        Returns:
            bool: True if there's overlap in X-Y plane
        """
        # For X-Y plane check, we can project to 2D
        # Get rotation matrices
        block_rot = self.quaternion_to_rotation_matrix(block_quat)
        white_rot = self.quaternion_to_rotation_matrix(white_quat)
        
        # Extract 2D rotation (just X-Y components)
        block_rot_2d = block_rot[:2, :2]
        white_rot_2d = white_rot[:2, :2]
        
        # Half sizes in 2D
        block_half = np.array([block_size[0]/2, block_size[1]/2])
        white_half = np.array([white_size[0]/2, white_size[1]/2])
        
        # Centers in 2D
        block_center_2d = block_pos[:2]
        white_center_2d = white_pos[:2]
        
        # Separating Axis Theorem: Check all 4 potential separating axes
        # (2 from each box's local axes)
        axes = []
        axes.append(block_rot_2d[:, 0])  # Block's X axis
        axes.append(block_rot_2d[:, 1])  # Block's Y axis
        axes.append(white_rot_2d[:, 0])  # White's X axis
        axes.append(white_rot_2d[:, 1])  # White's Y axis
        
        # Check each axis
        for axis in axes:
            # Project both boxes onto this axis
            # Distance between centers projected onto axis
            center_dist = np.abs(np.dot(white_center_2d - block_center_2d, axis))
            
            # Project block's half-extents
            block_proj = (np.abs(np.dot(block_rot_2d[:, 0], axis)) * block_half[0] +
                         np.abs(np.dot(block_rot_2d[:, 1], axis)) * block_half[1])
            
            # Project white's half-extents
            white_proj = (np.abs(np.dot(white_rot_2d[:, 0], axis)) * white_half[0] +
                         np.abs(np.dot(white_rot_2d[:, 1], axis)) * white_half[1])
            
            # If separation on this axis, no collision
            if center_dist > block_proj + white_proj:
                return False
        
        # No separating axis found, boxes overlap
        return True
    
    def check_block_within_xy_area(self, block_pos, block_quat, white_pos, white_quat):
        """
        Check if a tiny colored block overlaps with the white square in X-Y plane.
        Now properly handles rotated blocks using oriented bounding box collision detection.
        
        Args:
            block_pos: Position of colored block CENTER [x, y, z] 
            block_quat: Orientation quaternion of block [w, x, y, z]
            white_pos: Position of white square CENTER [x, y, z]
            white_quat: Orientation quaternion of white square [w, x, y, z]
            
        Returns:
            bool: True if block overlaps with white square in X-Y plane
        """
        # Use the oriented bounding box overlap check
        return self.check_oriented_box_overlap_xy(
            block_pos, block_quat, self.colored_size,
            white_pos, white_quat, self.white_size
        )
    
    def check_block_contact_with_z(self, block_pos, block_quat, white_pos, white_quat):
        """
        Check if a tiny colored block is in contact with white square including Z check.
        Block must overlap in X-Y area AND within Z threshold of the white square's top surface.
        Now properly handles rotated blocks.
        
        Args:
            block_pos: Position of colored block CENTER [x, y, z] 
            block_quat: Orientation quaternion of block [w, x, y, z]
            white_pos: Position of white square CENTER [x, y, z]
            white_quat: Orientation quaternion of white square [w, x, y, z]
            
        Returns:
            bool: True if block is in contact (overlaps in X-Y and within Z threshold)
        """
        # First check X-Y overlap (now handles rotation)
        xy_within = self.check_block_within_xy_area(block_pos, block_quat, white_pos, white_quat)
        if not xy_within:
            return False
        
        # For Z check with rotation, we need to find the lowest point of the rotated block
        # Get all corners of the rotated block
        block_corners = self.get_oriented_bounding_box_corners(
            block_pos, self.colored_size, block_quat
        )
        
        # Find the lowest Z coordinate of the block (minimum Z of all corners)
        block_min_z = np.min(block_corners[:, 2])
        
        # Get the top surface of the white square (considering its rotation too)
        white_corners = self.get_oriented_bounding_box_corners(
            white_pos, self.white_size, white_quat
        )
        
        # Find the highest Z coordinate of the white square (maximum Z of all corners)
        white_max_z = np.max(white_corners[:, 2])
        
        # Calculate distance from block's lowest point to white's highest point
        z_distance = block_min_z - white_max_z
        
        # Contact criteria:
        # - z_distance >= -0.003: Block can be embedded up to 3mm (physics settling)
        # - z_distance <= z_threshold: Block should be close to surface
        z_within = -0.003 <= z_distance <= self.z_threshold
        
        return z_within
    
    def update_contacts(self, scene):
        """
        Update contact states between colored blocks and white square.
        
        Args:
            scene: The simulation scene containing the objects
            
        Returns:
            dict: Current contact states and success flag
        """
        # Start timing on first contact check of this session
        if not self.timing_started or self.start_time is None:
            self.start_time = time.time()
            self.timing_started = True
            print(f"[ContactDetector] Starting contact detection timer")
        
        try:
            # Get positions and orientations of all objects
            red_pos = scene["cube_rigid_1"].data.root_pos_w[0].cpu().numpy()
            red_quat = scene["cube_rigid_1"].data.root_quat_w[0].cpu().numpy()
            
            green_pos = scene["cube_rigid_2"].data.root_pos_w[0].cpu().numpy()
            green_quat = scene["cube_rigid_2"].data.root_quat_w[0].cpu().numpy()
            
            blue_pos = scene["cube_rigid_3"].data.root_pos_w[0].cpu().numpy()
            blue_quat = scene["cube_rigid_3"].data.root_quat_w[0].cpu().numpy()
            
            white_pos = scene["cube_rigid_4"].data.root_pos_w[0].cpu().numpy()
            white_quat = scene["cube_rigid_4"].data.root_quat_w[0].cpu().numpy()
            
            # Check contacts for each colored block with the white square
            if self.use_z_check:
                # Use full XYZ contact check
                self.contact_states["red_white"] = self.check_block_contact_with_z(
                    red_pos, red_quat, white_pos, white_quat
                )
                self.contact_states["green_white"] = self.check_block_contact_with_z(
                    green_pos, green_quat, white_pos, white_quat
                )
                self.contact_states["blue_white"] = self.check_block_contact_with_z(
                    blue_pos, blue_quat, white_pos, white_quat
                )
            else:
                # Use X-Y area check only
                self.contact_states["red_white"] = self.check_block_within_xy_area(
                    red_pos, red_quat, white_pos, white_quat
                )
                self.contact_states["green_white"] = self.check_block_within_xy_area(
                    green_pos, green_quat, white_pos, white_quat
                )
                self.contact_states["blue_white"] = self.check_block_within_xy_area(
                    blue_pos, blue_quat, white_pos, white_quat
                )
            
            # Track first contact times
            current_time = time.time()
            elapsed_time = current_time - self.start_time if self.start_time else 0.0
            
            for block_color in ["red_white", "green_white", "blue_white"]:
                # Track new contacts
                if self.contact_states[block_color] and self.first_contact_times[block_color] is None:
                    self.first_contact_times[block_color] = elapsed_time
                    print(f"[ContactDetector] {block_color.split('_')[0].upper()} block made contact at {elapsed_time:.2f}s")
                
                # Track lost contacts (for debugging)
                if len(self.contact_history) > 0:
                    last_state = self.contact_history[-1]["contacts"].get(block_color, False)
                    current_state = self.contact_states[block_color]
                    if last_state and not current_state:
                        # Get current position for debugging
                        if block_color == "red_white":
                            block_pos = red_pos
                        elif block_color == "green_white":
                            block_pos = green_pos
                        else:
                            block_pos = blue_pos
                        
                        # Check why contact was lost
                        if block_color == "red_white":
                            block_quat = red_quat
                        elif block_color == "green_white":
                            block_quat = green_quat
                        else:
                            block_quat = blue_quat
                            
                        xy_ok = self.check_block_within_xy_area(block_pos, block_quat, white_pos, white_quat)
                        
                        # Get actual z-distance with rotation
                        block_corners = self.get_oriented_bounding_box_corners(block_pos, self.colored_size, block_quat)
                        white_corners = self.get_oriented_bounding_box_corners(white_pos, self.white_size, white_quat)
                        z_dist = np.min(block_corners[:, 2]) - np.max(white_corners[:, 2])
                        
                        print(f"[ContactDetector] WARNING: {block_color.split('_')[0].upper()} block LOST contact at {elapsed_time:.2f}s")
                        print(f"  Position: {block_pos}, White: {white_pos}")
                        print(f"  XY overlap: {xy_ok}, Z-distance: {z_dist:.4f}m (tolerance: -0.003 to {self.z_threshold})")
            
            # Check if all blocks have made first contact (not necessarily simultaneously)
            all_have_touched = (self.first_contact_times["red_white"] is not None and
                               self.first_contact_times["green_white"] is not None and
                               self.first_contact_times["blue_white"] is not None)
            
            # Record success snapshot when the LAST block makes first contact
            if all_have_touched and self.success_snapshot is None:
                # Find which block just made contact (the one with elapsed_time as its first contact)
                tolerance = 0.1  # Allow small time difference
                just_touched = []
                for block_color in ["red_white", "green_white", "blue_white"]:
                    if (self.first_contact_times[block_color] is not None and
                        abs(self.first_contact_times[block_color] - elapsed_time) < tolerance):
                        just_touched.append(block_color.split('_')[0].upper())
                
                if just_touched:
                    print(f"[ContactDetector] SUCCESS! All blocks have now made contact at {elapsed_time:.2f}s")
                    print(f"[ContactDetector] Last block(s) to touch: {', '.join(just_touched)}")
                
                # Save success snapshot at this moment
                self.success_snapshot = {
                    "timestamp": elapsed_time,
                    "positions": {
                        "white": {"pos": white_pos.tolist()},
                        "red": {"pos": red_pos.tolist()},
                        "green": {"pos": green_pos.tolist()},
                        "blue": {"pos": blue_pos.tolist()}
                    },
                    "first_contact_achieved": {
                        "red_white": self.first_contact_times["red_white"] is not None,
                        "green_white": self.first_contact_times["green_white"] is not None,
                        "blue_white": self.first_contact_times["blue_white"] is not None
                    },
                    "current_contact_states": {k: bool(v) for k, v in self.contact_states.items()}
                }
            
            # Still track current success state for other uses
            self.success = all(self.contact_states.values())
            
            # Record contact history (convert numpy bools to Python bools)
            self.contact_history.append({
                "timestamp": elapsed_time,
                "contacts": {k: bool(v) for k, v in self.contact_states.items()},
                "success": bool(self.success)
            })
            
            # Calculate Z distances for debugging with rotation awareness
            # Get corners for all objects
            white_corners = self.get_oriented_bounding_box_corners(white_pos, self.white_size, white_quat)
            white_max_z = np.max(white_corners[:, 2])
            
            # Red block
            red_corners = self.get_oriented_bounding_box_corners(red_pos, self.colored_size, red_quat)
            red_min_z = np.min(red_corners[:, 2])
            red_z_dist = red_min_z - white_max_z
            
            # Green block
            green_corners = self.get_oriented_bounding_box_corners(green_pos, self.colored_size, green_quat)
            green_min_z = np.min(green_corners[:, 2])
            green_z_dist = green_min_z - white_max_z
            
            # Blue block
            blue_corners = self.get_oriented_bounding_box_corners(blue_pos, self.colored_size, blue_quat)
            blue_min_z = np.min(blue_corners[:, 2])
            blue_z_dist = blue_min_z - white_max_z
            
            # Store positions for final logging (convert numpy bools to Python bools)
            self.last_positions = {
                "white": {
                    "pos": white_pos.tolist(), 
                    "bounds_x": [float(white_pos[0]-0.025), float(white_pos[0]+0.025)], 
                    "bounds_y": [float(white_pos[1]-0.025), float(white_pos[1]+0.025)],
                    "top_z": float(white_max_z)
                },
                "red": {
                    "pos": red_pos.tolist(), 
                    "detected": bool(self.contact_states["red_white"]),
                    "z_distance_from_white_top": float(red_z_dist)
                },
                "green": {
                    "pos": green_pos.tolist(), 
                    "detected": bool(self.contact_states["green_white"]),
                    "z_distance_from_white_top": float(green_z_dist)
                },
                "blue": {
                    "pos": blue_pos.tolist(), 
                    "detected": bool(self.contact_states["blue_white"]),
                    "z_distance_from_white_top": float(blue_z_dist)
                }
            }
            
        except Exception as e:
            print(f"[ContactDetector] Error updating contacts: {e}")
        
        return {
            "contacts": self.contact_states.copy(),
            "success": self.success
        }
    
    def save_final_state(self, log_dir):
        """
        Save the final contact state and positions to a JSON file.
        This is called when logging stops to record the final result.
        
        Args:
            log_dir: Directory to save the success log
        """
        # Ensure directory exists
        os.makedirs(log_dir, exist_ok=True)
        # Determine if each block made contact at least once
        red_touched = self.first_contact_times["red_white"] is not None
        green_touched = self.first_contact_times["green_white"] is not None
        blue_touched = self.first_contact_times["blue_white"] is not None
        
        # Success is true if ALL blocks touched at least once
        all_touched = red_touched and green_touched and blue_touched
        
        # Calculate the latest first-touch time (when the last block made contact)
        first_touch_times = []
        if red_touched:
            first_touch_times.append(self.first_contact_times["red_white"])
        if green_touched:
            first_touch_times.append(self.first_contact_times["green_white"])
        if blue_touched:
            first_touch_times.append(self.first_contact_times["blue_white"])
        
        latest_first_touch = max(first_touch_times) if first_touch_times else None
        
        final_data = {
            # Individual block success based on FIRST contact
            "red": red_touched,
            "green": green_touched,
            "blue": blue_touched,
            "success": all_touched,  # True if ALL blocks touched at least once
            
            # Backwards compatibility
            "red_white_contact": red_touched,
            "green_white_contact": green_touched,
            "blue_white_contact": blue_touched,
            
            "timing": {
                "total_duration": float(time.time() - self.start_time) if self.start_time else 0.0,
                "success_time": float(latest_first_touch) if latest_first_touch is not None else None,  # Time when last block touched
                "first_contact_times": {
                    "red": float(self.first_contact_times["red_white"]) if self.first_contact_times["red_white"] is not None else None,
                    "green": float(self.first_contact_times["green_white"]) if self.first_contact_times["green_white"] is not None else None,
                    "blue": float(self.first_contact_times["blue_white"]) if self.first_contact_times["blue_white"] is not None else None
                }
            },
            "final_positions": self.last_positions,
            "success_snapshot": self.success_snapshot
        }
        
        success_path = os.path.join(log_dir, "success.json")
        with open(success_path, "w") as f:
            json.dump(final_data, f, indent=4)
        
        if all_touched:
            print(f"[ContactDetector] SUCCESS! All blocks made contact. Final state saved to {success_path}")
            print(f"[ContactDetector] Success achieved at {latest_first_touch:.2f}s (when last block touched)")
        else:
            print(f"[ContactDetector] Task incomplete. Final state saved to {success_path}")
            print(f"[ContactDetector] First contacts achieved: Red={red_touched}, "
                  f"Green={green_touched}, Blue={blue_touched}")
            
            # Print debug info about why contacts might have failed
            if self.use_z_check and hasattr(self, 'last_positions'):
                print(f"[ContactDetector] Z-check enabled. Checking final positions:")
                for color in ['red', 'green', 'blue']:
                    if color in self.last_positions:
                        z_dist = self.last_positions[color].get('z_distance_from_white_top', 'N/A')
                        detected = self.last_positions[color].get('detected', False)
                        print(f"  {color.upper()}: Z-distance={z_dist}m, Detected={detected}")
                print(f"  Z-tolerance: -0.003m to +{self.z_threshold}m")
        
        # Also save a debug file with detailed position info
        debug_path = os.path.join(log_dir, "contact_debug.json")
        debug_data = {
            "white_square": {
                "center": self.last_positions.get("white", {}).get("pos", []),
                "x_bounds": self.last_positions.get("white", {}).get("bounds_x", []),
                "y_bounds": self.last_positions.get("white", {}).get("bounds_y", []),
                "size": self.white_size.tolist()
            },
            "blocks": {
                "red": self.last_positions.get("red", {}),
                "green": self.last_positions.get("green", {}),
                "blue": self.last_positions.get("blue", {})
            },
            "timing_summary": {
                "total_duration": float(time.time() - self.start_time) if self.start_time else 0.0,
                "success_achieved": self.success,
                "success_time": self.success_time,
                "time_to_success": self.success_time if self.success_time else "N/A",
                "individual_contact_times": self.first_contact_times
            },
            "detection_info": {
                "method": "X-Y area check + Z check" if self.use_z_check else "X-Y area check only (ignoring Z)",
                "z_threshold": self.z_threshold if self.use_z_check else "N/A",
                "z_tolerance": "-0.003m to +{}m".format(self.z_threshold) if self.use_z_check else "N/A",
                "white_square_expected_pos": "[0.0, 0.055, 0.0]",
                "white_square_thickness": self.white_size[2],
                "colored_block_height": self.colored_size[2],
                "note": "Z-distance is measured from lowest block corner to highest white corner (rotation-aware)",
                "rotation_handling": "Full 3D rotation support using Separating Axis Theorem for X-Y overlap"
            }
        }
        with open(debug_path, "w") as f:
            json.dump(debug_data, f, indent=4)
        
        # Save contact history timeline
        timeline_path = os.path.join(log_dir, "contact_timeline.json")
        timeline_data = {
            "start_time": float(self.start_time) if self.start_time else None,
            "success_time": float(self.success_time) if self.success_time is not None else None,
            "duration": float(time.time() - self.start_time) if self.start_time else 0.0,
            "history": self.contact_history[-100:]  # Keep last 100 entries to avoid huge files
        }
        with open(timeline_path, "w") as f:
            json.dump(timeline_data, f, indent=4)
        
        return True
    
    def reset(self):
        """Reset contact states and success flag."""
        self.contact_states = {
            "red_white": False,
            "green_white": False,
            "blue_white": False
        }
        self.success = False
        self.start_time = None  # Will be set on first update
        self.success_time = None
        self.first_contact_times = {
            "red_white": None,
            "green_white": None,
            "blue_white": None
        }
        self.contact_history = []
        self.timing_started = False
        self.success_snapshot = None