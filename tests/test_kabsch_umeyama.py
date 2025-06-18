import numpy as np
import pytest
from numpy.testing import assert_allclose
from aruco_estimator.opt import kabsch_umeyama
from aruco_estimator.utils import get_transformation_matrix_4x4

def apply_transformation(points, R, c, t):
    """Apply transformation to points: t + c * R @ points"""
    return t + (c * R @ points.T).T


def apply_transformation_4x4(points, T):
    """Apply 4x4 transformation matrix to points"""
    # Convert to homogeneous coordinates
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    # Apply transformation
    transformed_homo = (T @ points_homo.T).T
    # Convert back to 3D coordinates
    return transformed_homo[:, :3]


def test_identical_point_sets():
    """Test with identical point sets - should give identity transformation"""
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    R, c, t = kabsch_umeyama(points, points)
    
    # Should be close to identity rotation
    assert_allclose(R, np.eye(3), atol=1e-10)
    # Should be close to unit scaling
    assert_allclose(c, 1.0, atol=1e-10)
    # Should be close to zero translation
    assert_allclose(t, np.zeros(3), atol=1e-10)

def test_pure_translation():
    """Test with pure translation"""
    pointset_A = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    translation = np.array([5.0, 3.0, -2.0])
    pointset_B = pointset_A + translation  # B = A + translation
    
    R, c, t = kabsch_umeyama(pointset_A, pointset_B)
    
    # Should be identity rotation, unit scaling
    # Translation should map B back to A: A = t + c*R*B = t + B (since c=1, R=I)
    # So t = A - B = -translation
    assert_allclose(R, np.eye(3), atol=1e-10)
    assert_allclose(c, 1.0, atol=1e-10)
    assert_allclose(t, -translation, atol=1e-10)

def test_pure_scaling():
    """Test with pure uniform scaling"""
    pointset_A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    scale_factor = 2.5
    pointset_B = pointset_A * scale_factor  # B = scale_factor * A
    
    R, c, t = kabsch_umeyama(pointset_A, pointset_B)
    
    # Should be identity rotation, inverse scaling (to map B back to A), zero translation
    # A = c * R * B = c * B (since R=I), so c = 1/scale_factor
    assert_allclose(R, np.eye(3), atol=1e-10)
    assert_allclose(c, 1.0/scale_factor, atol=1e-10)
    assert_allclose(t, np.zeros(3), atol=1e-10)

def test_rotation_around_z_axis():
    """Test with known rotation around Z-axis"""
    # Create points in XY plane
    pointset_A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
    
    # Rotate 90 degrees around Z-axis
    angle = np.pi / 2
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    pointset_B = (rotation_matrix @ pointset_A.T).T  # B = R * A
    
    R, c, t = kabsch_umeyama(pointset_A, pointset_B)
    
    # Should recover the inverse rotation matrix (to map B back to A)
    # A = R_recovered * B, so R_recovered = R^(-1) = R.T
    assert_allclose(R, rotation_matrix.T, atol=1e-10)
    assert_allclose(c, 1.0, atol=1e-10)
    assert_allclose(t, np.zeros(3), atol=1e-10)

def test_combined_transformation():
    """Test with combined rotation, scaling, and translation"""
    pointset_A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    # Define transformation parameters
    angle = np.pi / 4  # 45 degrees
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    scale_factor = 1.5
    translation = np.array([2.0, -1.0, 3.0])
    
    # Apply transformation: B = scale * R @ A + translation
    pointset_B = (scale_factor * rotation_matrix @ pointset_A.T).T + translation
    
    R, c, t = kabsch_umeyama(pointset_A, pointset_B)
    
    # To map B back to A: A = c * R * B + t
    # Since B = scale * rot * A + trans, the inverse is:
    # A = (1/scale) * rot.T * (B - trans)
    # So: R = rot.T, c = 1/scale
    assert_allclose(R, rotation_matrix.T, atol=1e-10)
    assert_allclose(c, 1.0/scale_factor, atol=1e-10)
    # The translation t will be calculated by the algorithm to make it work

def test_transformation_application():
    """Test that applying the recovered transformation aligns the point sets"""
    # Create random point sets
    np.random.seed(42)
    pointset_A = np.random.rand(10, 3) * 10
    
    # Apply arbitrary transformation: B = scale * R @ A + t
    angle = np.pi / 3
    R_applied = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    c_applied = 2.0
    t_applied = np.array([5.0, -3.0, 1.0])
    
    pointset_B = (c_applied * R_applied @ pointset_A.T).T + t_applied
    
    # Recover transformation (should find the inverse transformation)
    R, c, t = kabsch_umeyama(pointset_A, pointset_B)
    
    # Apply recovered transformation to B to get back to A
    pointset_B_transformed = apply_transformation(pointset_B, R, c, t)
    
    # Should be very close to A
    assert_allclose(pointset_B_transformed, pointset_A, atol=1e-10)

def test_4x4_transformation_matrix():
    """Test 4x4 transformation matrix generation and application"""
    pointset_A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # Apply known transformation: B = scale * R @ A + translation
    R_applied = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    c_applied = 1.5
    t_applied = np.array([2.0, -1.0, 3.0])
    
    pointset_B = (c_applied * R_applied @ pointset_A.T).T + t_applied
    
    # Recover transformation
    R, c, t = kabsch_umeyama(pointset_A, pointset_B)
    
    # Create 4x4 matrix
    T = get_transformation_matrix_4x4(R, c, t)
    
    # Test matrix structure
    assert T.shape == (4, 4)
    assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-15)
    assert_allclose(T[:3, :3], c * R, atol=1e-15)
    assert_allclose(T[:3, 3], t, atol=1e-15)
    
    # Test that 4x4 matrix gives same result as individual transformation
    pointset_B_transformed_4x4 = apply_transformation_4x4(pointset_B, T)
    pointset_B_transformed_individual = apply_transformation(pointset_B, R, c, t)
    
    assert_allclose(pointset_B_transformed_4x4, pointset_B_transformed_individual, atol=1e-15)
    # Both should transform B back to A
    assert_allclose(pointset_B_transformed_4x4, pointset_A, atol=1e-10)

def test_rotation_matrix_properties():
    """Test that the recovered rotation matrix has proper rotation matrix properties"""
    np.random.seed(123)
    pointset_A = np.random.rand(5, 3)
    pointset_B = np.random.rand(5, 3)
    
    R, c, t = kabsch_umeyama(pointset_A, pointset_B)
    
    # Test orthogonality: R @ R.T should be identity
    assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    
    # Test determinant should be 1 (proper rotation, not reflection)
    assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

def test_minimum_points():
    """Test with minimum number of points (3 for 3D)"""
    pointset_A = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    
    pointset_B = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0]
    ])
    
    R, c, t = kabsch_umeyama(pointset_A, pointset_B)
    
    # Should not raise any errors and should give reasonable results
    assert R.shape == (3, 3)
    assert isinstance(c, (int, float, np.number))
    assert t.shape == (3,)
    
    # Verify transformation works (B transformed should give A)
    transformed = apply_transformation(pointset_B, R, c, t)
    assert_allclose(transformed, pointset_A, atol=1e-10)

def test_error_on_mismatched_shapes():
    """Test that algorithm raises error for mismatched point set shapes"""
    pointset_A = np.array([[1, 2, 3], [4, 5, 6]])
    pointset_B = np.array([[1, 2, 3]])
    
    with pytest.raises(AssertionError):
        kabsch_umeyama(pointset_A, pointset_B)

