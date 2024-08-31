
import numpy as np
from scipy.optimize import least_squares

from shared.projective_utils import apply_transformation, rotation_matrix


def axis_rectification_to_yz_plane(camera_matrix: np.array, axis: np.array):
    l1 = apply_transformation(camera_matrix.T, axis, divide=False)
    # l1 /= np.linalg.norm(l1)
    l1 /= np.linalg.norm(l1[:2])

    def calculate_rotation_b():
        P = np.identity(3) - np.outer(l1, l1)
        xo = np.array([0.0, 0.0, 1.0])
        xop = apply_transformation(P, xo)
        if np.linalg.norm(xo - xop) < 1e-10:
            return np.identity(3)

        v = np.cross(xop, xo)
        nb = v / np.linalg.norm(v)
        thetab = np.arccos(np.dot(xop, xo) / np.linalg.norm(xop) * np.linalg.norm(xo))

        R = np.array([
            [0.0, -nb[2], nb[1]],
            [nb[2], 0.0, -nb[0]],
            [-nb[1], nb[0], 0.0]
        ])
        manual_rodrigues = np.identity(3) + np.sin(thetab) * R + (1.0 - np.cos(thetab)) * np.dot(R, R)
        return manual_rodrigues


    Rb = calculate_rotation_b()

    l2 = apply_transformation(np.linalg.inv(Rb.T), l1, divide=False)
    # l2 /= np.linalg.norm(l2)
    l2 /= np.linalg.norm(l2[:2])

    def calculate_rotation_a():
        rot_matrix = np.array([
            [l2[0], l2[1], 0.0],
            [-l2[1], l2[0], 0.0],
            [0.0, 0.0, 1.0]
        ])
        if np.arccos(l2[0]) < np.pi / 2.0:
            return rot_matrix
        return rot_matrix @ rotation_matrix('z', np.pi)

    Ra = calculate_rotation_a()

    l3 = apply_transformation(np.linalg.inv(Ra.T), l2, divide=False)
    l3 /= l3[0]

    return Ra, Rb

def calculate_xs_ys_depth(points, normals, dz):
    depth = list(map(lambda p, n: (dz * n[0]) / (n[0] - n[2] * p[0]), points, normals))

    xs = list(map(lambda p, d: np.sqrt((d * p[0]) ** 2 + (d - dz) ** 2), points, depth))
    ys = list(map(lambda p, d: d * p[1], points, depth))

    return np.array(xs), np.array(ys), np.array(depth)

def rectification_using_x_angle(points, normals, angles, dz=1.0):
    rectification = []
    for angle in angles:
        angle = (angle + np.pi / 2.0) % np.pi
        rotation_matrix_rectification = rotation_matrix('x', angle)

        points_rectify = apply_transformation(rotation_matrix_rectification, points)
        normals_rectify = apply_transformation(rotation_matrix_rectification, normals, divide=False)
        normals_rectify /= np.linalg.norm(normals_rectify, axis=1)[:, np.newaxis]

        xs, ys, depth = calculate_xs_ys_depth(points_rectify, normals_rectify, dz)
        rectification.append((xs, ys, depth))
    return rectification

def search_angle_rotation_x(ellipses_coefficients):
    def equations(theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        result = [e - (f * cos_theta ** 2 - g * sin_theta * cos_theta + h * sin_theta ** 2)for (e, f, g, h) in ellipses_coefficients]
        return sum(result)
    result = least_squares(equations, np.pi / 2.0, bounds=(0, 2 * np.pi), verbose=0)
    return result