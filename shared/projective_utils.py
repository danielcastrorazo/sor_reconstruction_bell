import numpy as np
from loguru import logger

from shared.dataclasses_ import ImageMetadata


def two_points_to_line(x, y):
    """Given 2 points in a plane return the homogeneous representation of a line.
    Accepts 2d points in the form [x,y] or homogeneous points in the form [x, y, z]
    """
    assert len(x) == len(y) and 4 > len(x) > 1
    if len(x) == 2:
        return np.array([x[1] - y[1], y[0] - x[0], x[0] * y[1] - x[1] * y[0]])
    return np.cross(x, y)

def return_corners_image(width, height):
    """Returns the corner points of an image of size width x height in homogeneous form.
    The order is top-left, top-right, bot-left, bot-right."""
    return np.array([
        np.array([0, 0, 1]),
        np.array([width - 1, 0, 1]),
        np.array([0, height - 1, 1]),
        np.array([width - 1, height - 1, 1])
    ])

def intersection_of_line_in_rectangle(line, width, height):
    """Returns the two intersections points of a line and a rectangle (0 : width - 1, 0 : height - 1)"""
    corners = return_corners_image(width, height)

    top = two_points_to_line(corners[0], corners[1])
    bot = two_points_to_line(corners[2], corners[3])
    left = two_points_to_line(corners[0], corners[2])
    right = two_points_to_line(corners[1], corners[3])

    line_ = line.astype(float)

    ans = []
    for edge_line in (top, bot, left, right):
        x, y, z = np.cross(line_, edge_line.astype(float))
        if z == 0:
            continue
        x = x / z
        y = y / z
        if 0 <= x < width and 0 <= y < height:
            ans.append([x, y])
            if len(ans) == 2:
                return ans

    return ans

def calculate_intrinsics_camera_matrix(metadata: ImageMetadata):
    f = metadata.focal_length
    sensor_width, sensor_height = metadata.sensor_dimensions
    height, width = metadata.image_size

    if height > width:
        logger.trace('Height > Width. Swapping sensor width and sensor height.')
        sensor_width, sensor_height = sensor_height, sensor_width

    fx = f * width / sensor_width
    fy = f * height / sensor_height
    cx, cy = width / 2.0, height / 2.0
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # fov_x = np.rad2deg(2 * np.arctan2(width, 2 * fx))
    # fov_y = np.rad2deg(2 * np.arctan2(height, 2 * fy))   
    # logger.info(f'Field of view x : {fov_x}')
    # logger.info(f'Field of view y : {fov_y}')

    return K

def rotation_matrix(axis: str, angle: float):
    c, s = np.cos(angle), np.sin(angle)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

def apply_transformation(T, x, divide=True):
    tx = (T @ x.T).T
    if divide:
        tx /= tx[:, [-1]] if len(tx.shape) > 1 else tx[-1]
    return tx

def to_homogeneous(x):
    if isinstance(x, np.ndarray):
        return np.c_[x, np.ones(len(x))]
    elif isinstance(x, tuple):
        if isinstance(x[0], np.ndarray):
            return np.c_[x[0], x[1], np.ones(len(x[0]))]
        return np.c_[x[0], x[1], 1.0]
    raise TypeError('Bad type')

def cross_ratio(p1, p2, p3, p4):
    a, b = np.linalg.lstsq(np.matrix([p1, p2]).T, np.matrix(p3).T, rcond=None)[0]
    c, d = np.linalg.lstsq(np.matrix([p1, p2]).T, np.matrix(p4).T, rcond=None)[0]
    return ((b / a) / (d / c))[(0, 0)] 

def project_points_onto_line(points: np.array, axis: np.array) -> np.array:
    """
    Orthogonal projection of a point onto a line.
    """
    v = np.array([-axis[1], axis[0], 0.0])
    if np.abs(axis[1]) > 1e-10:
        p0 = np.array([0.0, -axis[2] / axis[1], 1.0])
    else:
        p0 = np.array([-axis[2] / axis[0], 0.0, 1.0])

    P = np.outer(v, v) / np.dot(v, v)
    Q = np.identity(3) - P

    Qp0 = apply_transformation(Q, p0)

    projection = apply_transformation(P, points, divide=False) + Qp0

    return projection

def intersection_between_conic_and_line(C, line, eps=1e-15):
    # Book Persepectives on Projective Geometry - Jurgen Richter Gebert
    # 11.3 Intersecting a Conic and a Line perspective on projective geometry
    line /= np.linalg.norm(line[:2])
    line = np.real(line)

    L = np.array([
        [0.0,   line[2], -line[1]],
        [-line[2],  0.0, line[0]],
        [line[1], -line[0], 0.0]
    ])

    D = L.T @ C @ L
    if np.abs(line[2]) < eps:
        alpha = (1.0 / line[0]) * np.sqrt(-np.linalg.det(D[1:, 1:]))
    else:
        alpha = (1.0 / line[2]) * np.sqrt(-np.linalg.det(D[:2, :2]))

    B = D + alpha * L

    ir, jc = np.transpose(np.nonzero(B))[0]
    p, q = np.copy(B[ir, :]), np.copy(B[:, jc])

    p /= p[2]
    q /= q[2]

    return p, q

def get_normal_line_form(line: np.array) -> np.array:
    A, B, C = line
    d = np.sqrt(A ** 2 + B ** 2)
    a2, b2, p = A / d, B / d, C / d
    theta = np.arctan2(b2, a2)

    return np.array([theta, p])

def find_perpendicular_to_axis(x: float, y: float, axis: np.array) -> None:
    d = axis[1] * x - axis[0] * y
    normal = np.array([-axis[1], axis[0], d])
    return normal

def check_contour_orientation(data):
    n = data.shape[0]

    total_signed_area = 0
    for i in range(n):
        p1, p2, p3 = data[i], data[(i + 1) % n], data[(i + 2) % n]
        signed_area = 0.5 * np.linalg.det(np.vstack((p1, p2, p3)))
        total_signed_area += signed_area

    return total_signed_area > 0

def reflect_points_on_line(points, line):
    a, b, c = line
    reflection_matrix = np.array([
        [b**2 - a**2, -2*a*b, -2*a*c],
        [-2*a*b, a**2 - b**2, -2*b*c],
        [0, 0, a**2 + b**2]
    ]) / (a**2 + b**2)
    return apply_transformation(reflection_matrix, points)

def get_angle_of_line(line, eps=1e-10):
    a, b, _ = line
    if np.abs(b) < eps:
        return np.pi / 2
    
    return np.arctan(-a / b)

def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    top_left = (np.argmax(rows), np.argmax(cols))
    bottom_right = (len(rows) - np.argmax(rows[::-1]) - 1, len(cols) - np.argmax(cols[::-1]) - 1)

    return top_left, bottom_right