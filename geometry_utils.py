"""Geometric utility functions for 3D transformations and distance calculations"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_points(points, rotation_axis, angle_degrees, center):
    """
    Rotate points around an axis through a center point
    
    Args:
        points: Nx3 array of points to rotate
        rotation_axis: 3D vector defining rotation axis
        angle_degrees: Rotation angle in degrees
        center: 3D point that the rotation axis passes through
        
    Returns:
        Rotated points as Nx3 array
    """
    if np.linalg.norm(rotation_axis) < 1e-10:
        return points  # No rotation for zero-length axis
    
    # Translate points so center is at origin
    translated_points = points - center
    
    # Create rotation object and apply
    rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
    rotation = R.from_rotvec(np.radians(angle_degrees) * rotation_axis_normalized)
    rotated_points = rotation.apply(translated_points)
    
    # Translate back
    return rotated_points + center


def angle_between_vectors(v1, v2):
    """
    Calculate angle between two vectors
    
    Args:
        v1, v2: 3D vectors
        
    Returns:
        Angle in degrees
    """
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 < 1e-10 or norm_v2 < 1e-10:
        return 0.0
    
    cosine_angle = dot_prod / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    return np.degrees(angle_radians)


def point_segment_distance(point, segment_start, segment_end):
    """
    Calculate minimum distance from a point to a line segment
    
    Uses the formula: |(B-A) * (P-A)| / |B-A|
    Also checks distances to endpoints
    
    Args:
        point: 3D point
        segment_start: Start point of line segment
        segment_end: End point of line segment
        
    Returns:
        Minimum distance from point to segment
    """
    axis = segment_end - segment_start
    length = np.linalg.norm(axis)
    
    if length < 1e-10:
        # Degenerate segment (point)
        return np.linalg.norm(point - segment_start)
    
    # Distance to infinite line
    cross_product = np.cross(axis, point - segment_start)
    dist_to_line = np.linalg.norm(cross_product) / length
    
    # Distance to endpoints
    dist_to_start = np.linalg.norm(point - segment_start)
    dist_to_end = np.linalg.norm(point - segment_end)
    
    return min(dist_to_line, dist_to_start, dist_to_end)


def segment_segment_distance(a1, a2, b1, b2):
    """
    Credit: Arfana

    Calculate the minimum distance between two finite line segments in 3D space
    
    Uses Ericson's method (Real-Time Collision Detection, Ch. 5) which correctly
    clamps to segment extents, unlike the infinite-line formula
    
    Args:
        a1, a2: Start and end points of first segment
        b1, b2: Start and end points of second segment
        
    Returns:
        Minimum distance between the segments
    """
    a1, a2, b1, b2 = map(np.array, (a1, a2, b1, b2))
    da = a2 - a1   # Direction of segment A
    db = b2 - b1   # Direction of segment B
    r = a1 - b1
    
    len_a_sq = np.dot(da, da)  # Squared length of A
    len_b_sq = np.dot(db, db)  # Squared length of B
    f = np.dot(db, r)
    
    # Handle degenerate cases (zero-length segments)
    if len_a_sq < 1e-10 and len_b_sq < 1e-10:
        return np.linalg.norm(r)
    
    if len_a_sq < 1e-10:
        # Segment A is a point - clamp s=0, find closest point on B
        s = 0.0
        t = np.clip(f / len_b_sq, 0.0, 1.0)
    else:
        c = np.dot(da, r)
        if len_b_sq < 1e-10:
            # Segment B is a point - clamp t=0, find closest point on A
            t = 0.0
            s = np.clip(-c / len_a_sq, 0.0, 1.0)
        else:
            # General non-degenerate case
            b_dot = np.dot(da, db)
            denom = len_a_sq * len_b_sq - b_dot * b_dot  # Always >= 0
            
            if denom > 1e-10:
                # Segments are not parallel
                s = np.clip((b_dot * f - c * len_b_sq) / denom, 0.0, 1.0)
            else:
                # Segments are parallel
                s = 0.0
            
            # Compute t for the clamped s, then clamp t
            t = (b_dot * s + f) / len_b_sq
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / len_a_sq, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b_dot - c) / len_a_sq, 0.0, 1.0)
    
    closest_a = a1 + s * da
    closest_b = b1 + t * db
    return np.linalg.norm(closest_a - closest_b)


def random_vector_in_plane(normal, scale=1.0):
    """
    Generate a random vector in the plane perpendicular to a given normal
    
    Args:
        normal: 3D normal vector defining the plane
        scale: Scaling factor for the output vector
        
    Returns:
        Random 3D vector perpendicular to normal, scaled by scale
    """
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)
    
    # Find a non-parallel vector to construct basis
    if abs(n[0]) < abs(n[1]) and abs(n[0]) < abs(n[2]):
        q = np.array([1.0, 0.0, 0.0])
    elif abs(n[1]) < abs(n[2]):
        q = np.array([0.0, 1.0, 0.0])
    else:
        q = np.array([0.0, 0.0, 1.0])
    
    # Construct orthonormal basis in the plane
    u = np.cross(n, q)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    
    # Random linear combination
    rand_u_coeff = np.random.uniform(-1, 1)
    rand_v_coeff = np.random.uniform(-1, 1)
    
    return (rand_u_coeff * u + rand_v_coeff * v) * scale


def cylinders_collide(cyl1, cyl2, tolerance=0.0):
    """
    Check if two cylinders collide
    
    Args:
        cyl1, cyl2: ChainCylinder objects
        tolerance: Additional separation distance required
        
    Returns:
        True if cylinders collide, False otherwise
    """
    # Calculate distance between cylinder axes
    axis_distance = segment_segment_distance(
        cyl1.start, cyl1.end,
        cyl2.start, cyl2.end
    )
    
    # Cylinders collide if axis distance is less than sum of radii
    return axis_distance < (cyl1.radius + cyl2.radius + tolerance)


def find_plane_intersection_parameter(point_on_plane, plane_normal, segment_start, segment_end):
    """
    Find parameter t where line segment intersects a plane
    
    The segment is parameterized as: P(t) = start + t*(end - start) for t in [0,1]
    
    Args:
        point_on_plane: Any point on the plane
        plane_normal: Normal vector to the plane
        segment_start: Start point of line segment
        segment_end: End point of line segment
        
    Returns:
        tuple: (crosses_plane, t_parameter)
            - crosses_plane: bool indicating if segment crosses plane
            - t_parameter: parameter value where intersection occurs (None if no crossing)
    """
    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    
    # Signed distances from plane
    f_start = np.dot(plane_normal_normalized, segment_start - point_on_plane)
    f_end = np.dot(plane_normal_normalized, segment_end - point_on_plane)
    
    # Check if segment crosses plane (endpoints on opposite sides)
    crosses = f_start * f_end < 0
    
    if not crosses:
        return False, None
    
    # Calculate parameter t for intersection point
    t = (-f_start) / (f_end - f_start)
    return True, t


def point_on_segment(segment_start, segment_end, t):
    """
    Get point on line segment at parameter t
    
    Args:
        segment_start: Start point of segment
        segment_end: End point of segment  
        t: Parameter value (0 = start, 1 = end)
        
    Returns:
        3D point on segment
    """
    return segment_start + t * (segment_end - segment_start)