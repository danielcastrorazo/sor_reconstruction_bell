import numpy as np

from shared.projective_utils import (
    apply_transformation,
    cross_ratio,
    intersection_between_conic_and_line,
)


def calculate_intersection_of_ellipses(coefficients1, coefficients2):
    a1, b1, c1, d1, e1, f1 = coefficients1
    a2, b2, c2, d2, e2, f2 = coefficients2

    def p0():
        return f1 ** 2 * c2 ** 2 - f1 * c2 * e2 * e1 - 2 * f1 * c2 * f2 * c1 + f1 * e2 ** 2 * c1 + c2 * f2 * e1 ** 2 - e2 * f2 * c1 * e1 + f2 ** 2 * c1 ** 2

    def p1():
        return -2 * c2 * f1 * c1 * d2 - b1 * c2 * f1 * e2 + 2 * c2 ** 2 * d1 * f1 - c2 * e1 * b2 * f1 + 2 * e2 * c1 * b2 * f1 + c2 * d2 * e1 ** 2 - c1 * e2 * e1 * d2 \
               + 2 * c1 ** 2 * f2 * d2 + 2 * b1 * c2 * f2 * e1 - b1 * c1 * e2 * f2 - c2 * d1 * e2 * e1 - 2 * c2 * c1 * f2 * d1 + c1 * e2 ** 2 * d1 - c1 * b2 * e1 * f2

    def p2():
        return 2 * c2 ** 2 * f1 * a1 - b1 * c2 * b2 * f1 - 2 * c2 * f1 * c1 * a2 + c1 * b2 ** 2 * f1 - c2 * e1 * e2 * a1 - 2 * c2 * c1 * f2 * a1 + c1 * e2 ** 2 * a1 \
               + c1 ** 2 * d2 ** 2 + 2 * b1 * c2 * d2 * e1 - b1 * c1 * e2 * d2 - 2 * c2 * c1 * d2 * d1 - c1 * b2 * e1 * d2 + b1 ** 2 * c2 * f2 - b1 * c2 * d1 * e2 \
               - b1 * f2 * c1 * b2 + c2 * e1 ** 2 * a2 - c1 * e2 * e1 * a2 + 2 * c1 ** 2 * f2 * a2 + c2 ** 2 * d1 ** 2 - c2 * e1 * b2 * d1 + 2 * e2 * c1 * b2 * d1

    def p3():
        return -2 * c2 * a1 * c1 * d2 - b1 * c2 * e2 * a1 + 2 * c2 ** 2 * a1 * d1 - c2 * e1 * b2 * a1 + 2 * e2 * c1 * b2 * a1 + b1 ** 2 * c2 * d2 - b1 * d2 * c1 * b2 \
               + 2 * c1 ** 2 * a2 * d2 + 2 * b1 * c2 * e1 * a2 - b1 * c1 * e2 * a2 - b1 * c2 * b2 * d1 - 2 * c2 * c1 * a2 * d1 - c1 * b2 * e1 * a2 + c1 * b2 ** 2 * d1

    def p4():
        return a2 * c2 * b1 ** 2 - b1 * b2 * a1 * c2 - b1 * a2 * c1 * b2 + a1 ** 2 * c2 ** 2 - 2 * a1 * c1 * a2 * c2 + c1 * b2 ** 2 * a1 + c1 ** 2 * a2 ** 2

    def ty(x_):
        return (
                       x_ ** 3 * a1 * b2 - x_ ** 2 * a2 * c1 + x_ ** 3 * a2 * b1 * b2 ** 2 * c1 ** 2 - x_ ** 3 * a1 * b2 ** 3 * c1 ** 2 + x_ ** 2 * a1 * c2 + x_ ** 3 * a2 * b1 ** 3 * (
                       -1 + c2) * c2 - x_ ** 3 * a1 * b1 ** 2 * b2 * (
                               -1 + c2) * c2 + x_ ** 2 * b2 * d1 + x_ ** 2 * b1 * b2 * d1 - x_ ** 2 * b2 ** 2 * c1 * d1 - x_ ** 2 * b1 * b2 ** 2 * c1 * d1 - x_ ** 2 * b2 ** 3 * c1 ** 2 * d1 + x_ * c2 * d1 + x_ ** 2 * b1 * b2 * c2 * d1 + x_ ** 2 * b1 ** 2 * b2 * c2 * d1 + 2 * x_ ** 2 * b1 * b2 ** 2 * c1 * c2 * d1 - x_ ** 2 * b1 ** 2 * b2 * c2 ** 2 * d1 - x_ * c2 * d1 ** 2 - x_ * c2 ** 2 * d1 ** 2 - x_ * b1 * c2 ** 2 * d1 ** 2 - x_ * b2 * c1 * c2 ** 2 * d1 ** 2 + x_ * b1 * c2 ** 3 * d1 ** 2 - x_ ** 2 * b1 * d2 - x_ ** 2 * b1 ** 2 * d2 - x_ * c1 * d2 + x_ ** 2 * a1 * c1 * d2 + x_ ** 2 * b1 * b2 * c1 * d2 + x_ ** 2 * b1 ** 2 * b2 * c1 * d2 - 2 * x_ ** 2 * a2 * c1 ** 2 * d2 + x_ ** 2 * b1 * b2 ** 2 * c1 ** 2 * d2 + 2 * x_ ** 2 * a2 * b1 * c1 ** 2 * (
                               -1 + c2) * d2 - x_ ** 2 * b1 ** 2 * c2 * d2 - x_ ** 2 * b1 ** 3 * c2 * d2 - 2 * x_ ** 2 * b1 ** 2 * b2 * c1 * c2 * d2 + x_ ** 2 * b1 ** 3 * c2 ** 2 * d2 + x_ * c1 * d1 * d2 + 2 * x_ * c1 * c2 * d1 * d2 + 2 * x_ * b1 * c1 * c2 * d1 * d2 + 2 * x_ * b2 * c1 ** 2 * c2 * d1 * d2 - 2 * x_ * b1 * c1 * c2 ** 2 * d1 * d2 - x_ * c1 ** 2 * d2 ** 2 - x_ * b1 * c1 ** 2 * d2 ** 2 - x_ * b2 * c1 ** 3 * d2 ** 2 + x_ * b1 * c1 ** 2 * c2 * d2 ** 2 - x_ ** 2 * a2 * e1 + x_ ** 2 * a2 * b2 * c1 * e1 + x_ ** 2 * a2 * b2 ** 2 * c1 ** 2 * e1 - x_ * a1 * c2 ** 3 * d1 * e1 - x_ * d2 * e1 - x_ * b1 * d2 * e1 + x_ * b2 * c1 * d2 * e1 + x_ * b1 * b2 * c1 * d2 * e1 + x_ * a2 * c1 ** 2 * d2 * e1 + x_ * b2 ** 2 * c1 ** 2 * d2 * e1 - x_ * b1 * c2 * d2 * e1 - x_ * b1 ** 2 * c2 * d2 * e1 - 2 * x_ * b1 * b2 * c1 * c2 * d2 * e1 - x_ * a2 * c1 ** 2 * c2 * d2 * e1 + x_ * b1 ** 2 * c2 ** 2 * d2 * e1 + x_ * a1 * c1 * c2 ** 2 * d2 * e1 - x_ * a1 * c1 * c2 * d2 * (
                               -2 * x_ * (1 + b1) + e1) + x_ ** 2 * a2 * b1 * b2 * c1 * (
                               x_ + (1 - 2 * c2) * e1) - x_ ** 2 * a2 * b1 * (
                               x_ + (1 + c2) * e1) - x_ ** 2 * a2 * b1 ** 2 * (
                               x_ + x_ * b2 * c1 * (-1 + 2 * c2) + c2 * (
                               x_ - (-1 + c2) * e1)) + x_ ** 2 * a1 * a2 * c1 * (
                               x_ + 2 * c2 * (x_ + (-1 + c2) * e1)) + x_ * a2 * c1 * d1 * (
                               x_ + c2 * (2 * x_ + (-1 + c2) * e1)) - x_ ** 2 * a2 ** 2 * c1 ** 2 * (
                               x_ + b1 * (x_ - x_ * c2) + (-1 + c2) * e1 + c1 * (
                               x_ * b2 - e2)) + x_ ** 2 * a1 * e2 + x_ ** 2 * a1 * b1 * e2 - x_ ** 2 * a1 * b2 * c1 * e2 + x_ ** 2 * a1 * b1 * (
                               1 + b1) * c2 * e2 - x_ ** 2 * a1 * b1 ** 2 * c2 ** 2 * e2 + x_ * d1 * e2 + x_ * b1 * d1 * e2 - x_ * b2 * c1 * d1 * e2 - x_ * b1 * b2 * c1 * d1 * e2 - x_ * b2 ** 2 * c1 ** 2 * d1 * e2 + x_ * b1 * c2 * d1 * e2 + x_ * b1 ** 2 * c2 * d1 * e2 + 2 * x_ * b1 * b2 * c1 * c2 * d1 * e2 - x_ * a2 * c1 ** 2 * c2 * d1 * e2 - x_ * b1 ** 2 * c2 ** 2 * d1 * e2 + x_ * a1 * c1 ** 2 * d2 * e2 + x_ * a2 * c1 ** 3 * d2 * e2 - x_ * a1 * c1 ** 2 * c2 * d2 * e2 + x_ ** 2 * a1 * a2 * c1 ** 2 * (
                               2 * c2 * (x_ * b2 - e2) + e2) - x_ * a1 * c2 ** 2 * d1 * (
                               2 * x_ + 2 * x_ * b1 - e1 - c1 * e2) - x_ * a1 * c2 * d1 * (
                               2 * x_ + c1 * e2) - x_ ** 2 * a1 * b2 ** 2 * c1 * (
                               x_ + b1 * (x_ - 2 * x_ * c2) + c1 * e2) - x_ ** 2 * a1 ** 2 * c2 * (
                               x_ + c2 * (x_ + x_ * b2 * c1 + b1 * (x_ - x_ * c2) + (-1 + c2) * e1) - c1 * (
                               -1 + c2) * e2) + x_ ** 2 * a1 * b1 * b2 * (x_ * (1 + c2) + c1 * (
                       -1 + 2 * c2) * e2) + x_ * b2 * f1 + x_ * b1 * b2 * f1 - x_ * b2 ** 2 * c1 * f1 - x_ * b1 * b2 ** 2 * c1 * f1 - x_ * b2 ** 3 * c1 ** 2 * f1 + c2 * f1 - x_ * a1 * c2 * f1 + x_ * b1 * b2 * c2 * f1 + x_ * b1 ** 2 * b2 * c2 * f1 + 2 * x_ * b1 * b2 ** 2 * c1 * c2 * f1 - x_ * a1 * c2 ** 2 * f1 - x_ * b1 ** 2 * b2 * c2 ** 2 * f1 - c2 * d1 * f1 - c2 ** 2 * d1 * f1 - b1 * c2 ** 2 * d1 * f1 - b2 * c1 * c2 ** 2 * d1 * f1 + b1 * c2 ** 3 * d1 * f1 + c1 * c2 * d2 * f1 + b1 * c1 * c2 * d2 * f1 + b2 * c1 ** 2 * c2 * d2 * f1 - b1 * c1 * c2 ** 2 * d2 * f1 + a1 * c2 ** 2 * e1 * f1 - a1 * c2 ** 3 * e1 * f1 + a2 * c1 * c2 * (
                               x_ + (
                               -1 + c2) * e1) * f1 + e2 * f1 + b1 * e2 * f1 - b2 * c1 * e2 * f1 - b1 * b2 * c1 * e2 * f1 - b2 ** 2 * c1 ** 2 * e2 * f1 + b1 * c2 * e2 * f1 + b1 ** 2 * c2 * e2 * f1 - a1 * c1 * c2 * e2 * f1 + 2 * b1 * b2 * c1 * c2 * e2 * f1 - a2 * c1 ** 2 * c2 * e2 * f1 - b1 ** 2 * c2 ** 2 * e2 * f1 + a1 * c1 * c2 ** 2 * e2 * f1 + x_ * a2 * b2 * c1 ** 2 * c2 * (
                               2 * x_ * d1 + f1) - x_ * a1 * b2 * c1 * c2 ** 2 * (
                               2 * x_ * d1 + f1) + x_ * a1 * b1 * c2 ** 3 * (
                               2 * x_ * d1 + f1) - x_ * a2 * b1 * c1 * (
                               -1 + c2) * c2 * (2 * x_ * (
                       x_ * a1 + d1) + f1) - x_ * b1 * f2 - x_ * b1 ** 2 * f2 - c1 * f2 + x_ * a1 * c1 * f2 + x_ * b1 * b2 * c1 * f2 + x_ * b1 ** 2 * b2 * c1 * f2 + x_ * b1 * b2 ** 2 * c1 ** 2 * f2 + x_ * a2 * b1 * c1 ** 2 * (
                               -1 + c2) * f2 - x_ * b1 ** 2 * c2 * f2 - x_ * b1 ** 3 * c2 * f2 - 2 * x_ * b1 ** 2 * b2 * c1 * c2 * f2 + x_ * b1 ** 3 * c2 ** 2 * f2 + c1 * d1 * f2 + c1 * c2 * d1 * f2 + b1 * c1 * c2 * d1 * f2 + b2 * c1 ** 2 * c2 * d1 * f2 - b1 * c1 * c2 ** 2 * d1 * f2 - c1 ** 2 * d2 * f2 - b1 * c1 ** 2 * d2 * f2 - b2 * c1 ** 3 * d2 * f2 + b1 * c1 ** 2 * c2 * d2 * f2 + a1 * c1 * c2 * (
                               x_ + x_ * b1 - e1) * f2 - e1 * f2 - b1 * e1 * f2 + b2 * c1 * e1 * f2 + b1 * b2 * c1 * e1 * f2 + b2 ** 2 * c1 ** 2 * e1 * f2 - b1 * c2 * e1 * f2 - b1 ** 2 * c2 * e1 * f2 - 2 * b1 * b2 * c1 * c2 * e1 * f2 + b1 ** 2 * c2 ** 2 * e1 * f2 + a1 * c1 * c2 ** 2 * e1 * f2 - a2 * c1 ** 2 * (
                               x_ + (
                               -1 + c2) * e1) * f2 + a1 * c1 ** 2 * e2 * f2 + a2 * c1 ** 3 * e2 * f2 - a1 * c1 ** 2 * c2 * e2 * f2 - x_ * a2 * b2 * c1 ** 3 * (
                               2 * x_ * d2 + f2) + x_ * a1 * b2 * c1 ** 2 * c2 * (
                               2 * x_ * d2 + f2) - x_ * a1 * b1 * c2 ** 2 * (f1 + c1 * (2 * x_ * d2 + f2))) / (
                       x_ ** 2 * a2 * c1 + x_ ** 2 * a2 * b1 * c1 + x_ * b2 * c1 - x_ ** 2 * a1 * b2 * c1 - x_ ** 2 * a1 * c2 - x_ * b1 * c2 - x_ * b2 * c1 * d1 - x_ * c2 * d1 + x_ * c1 * d2 + x_ * b1 * c1 * d2 - x_ * a2 * b2 * c1 ** 2 * e1 - c2 * e1 + x_ * a1 * c2 * e1 + x_ * a1 * b2 * c1 * c2 * e1 + x_ * a1 * c2 ** 2 * e1 + c2 * d1 * e1 + c2 ** 2 * d1 * e1 + b1 * c2 ** 2 * d1 * e1 + b2 * c1 * c2 ** 2 * d1 * e1 - b1 * c2 ** 3 * d1 * e1 - c1 * c2 * d2 * e1 - b1 * c1 * c2 * d2 * e1 - b2 * c1 ** 2 * c2 * d2 * e1 + b1 * c1 * c2 ** 2 * d2 * e1 - a1 * c2 ** 2 * e1 ** 2 + a1 * c2 ** 3 * e1 ** 2 - a2 * c1 * c2 * e1 * (
                       x_ + (
                       -1 + c2) * e1) + c1 * e2 - x_ * a1 * c1 * e2 + x_ * a2 * c1 ** 2 * e2 + x_ * a2 * b1 * c1 ** 2 * e2 - x_ * a1 * b2 * c1 ** 2 * e2 - x_ * a1 * c1 * c2 * e2 - c1 * d1 * e2 - c1 * c2 * d1 * e2 - b1 * c1 * c2 * d1 * e2 - b2 * c1 ** 2 * c2 * d1 * e2 + b1 * c1 * c2 ** 2 * d1 * e2 + c1 ** 2 * d2 * e2 + b1 * c1 ** 2 * d2 * e2 + b2 * c1 ** 3 * d2 * e2 - b1 * c1 ** 2 * c2 * d2 * e2 - a2 * c1 ** 2 * e1 * e2 + 2 * a1 * c1 * c2 * e1 * e2 + 2 * a2 * c1 ** 2 * c2 * e1 * e2 - 2 * a1 * c1 * c2 ** 2 * e1 * e2 - a1 * c1 ** 2 * e2 ** 2 - a2 * c1 ** 3 * e2 ** 2 + a1 * c1 ** 2 * c2 * e2 ** 2 - c2 * f1 - b1 * c2 * f1 + b2 * c1 * c2 * f1 + b1 * b2 * c1 * c2 * f1 + b2 ** 2 * c1 ** 2 * c2 * f1 - b1 * c2 ** 2 * f1 - b1 ** 2 * c2 ** 2 * f1 - 2 * b1 * b2 * c1 * c2 ** 2 * f1 + b1 ** 2 * c2 ** 3 * f1 + c1 * f2 + b1 * c1 * f2 - b2 * c1 ** 2 * f2 - b1 * b2 * c1 ** 2 * f2 - b2 ** 2 * c1 ** 3 * f2 + b1 * c1 * c2 * f2 + b1 ** 2 * c1 * c2 * f2 + 2 * b1 * b2 * c1 ** 2 * c2 * f2 - b1 ** 2 * c1 * c2 ** 2 * f2)

    return [(xi, ty(xi)) for xi in np.roots([p4(), p3(), p2(), p1(), p0()])]

def generate_entities(roots):
    combinations = (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1)
    )
    x = [np.array([*root, 1.0]) for root in roots]

    result = []
    for (a, b, c, d) in combinations:
        vanishing_line = np.cross(x[a], x[b])
        n_vanishing_line = np.cross(x[c], x[d])
        vertex = np.cross(vanishing_line, n_vanishing_line)

        lac = np.cross(x[a], x[c])
        lbc = np.cross(x[b], x[c])
        lad = np.cross(x[a], x[d])
        lbd = np.cross(x[b], x[d])

        axis = np.cross(np.cross(lac, lbd), np.cross(lad, lbc))
        axis /= axis[2]
        axis = np.real(axis)
        axis /= np.linalg.norm(axis)

        vanishing_line /= vanishing_line[2]
        vanishing_line /= np.linalg.norm(vanishing_line)

        result.append((axis, vertex, vanishing_line, x[a], x[b]))
    return result

def generate_w_homologies(points, tangent_lines, axis, vanishing_line, ellipse_w_matrix):    
    w_homologies = []
    char_invariants = []
    u_infs = []
    v_ws = []

    w_infs = []
    epoints = []
    for i, (p, t) in enumerate(zip(points, tangent_lines)):
        u_inf = np.cross(t, vanishing_line)
        u_inf /= u_inf[2]
        u_infs.append(u_inf)

        e_point_p, e_point_q = intersection_between_conic_and_line(ellipse_w_matrix, apply_transformation(ellipse_w_matrix, u_inf, divide=False))
        if np.sign(np.dot(axis, p)) == np.sign(np.dot(axis, e_point_p)):
            e_point = e_point_p
        else:
            e_point = e_point_q
        
        # TODO 
        # Case when the intersection with linf is inside the ellipse
        # TODO 
        # Case when both ellipse points are on one side of the axis.
        epoints.append(e_point)

        line_ep_p = np.cross(e_point, p)
        v_w = np.cross(line_ep_p, axis)
        v_w /= v_w[2]
        v_ws.append(v_w)

        w_inf = np.cross(line_ep_p, vanishing_line)
        w_inf /= w_inf[2]
        w_infs.append(w_inf)

        characteristic_invariant = cross_ratio(v_w, w_inf, e_point, p)
        char_invariants.append(characteristic_invariant)

        w_homology = np.real(np.identity(3) + (characteristic_invariant - 1.0) * np.outer(v_w, vanishing_line) / np.dot(v_w, vanishing_line))
        w_homology /= w_homology[2, 2]
        w_homologies.append(w_homology)
    return np.array(w_homologies), np.real(np.array(char_invariants)), np.array(u_infs), np.array(v_ws)

def create_rectification_matrix(ellipse_point, ellipse_matrix, vanishing_line, iac):
    center = apply_transformation(np.linalg.inv(ellipse_matrix), vanishing_line)
    xinf = np.cross(ellipse_point, center)
    xinf = np.cross(xinf, vanishing_line)
    xinf /= xinf[2]
    vanishing_point_axis = apply_transformation(np.linalg.inv(iac), vanishing_line)

    vanishing_line_m_inf = np.cross(xinf, vanishing_point_axis)
    vanishing_line_m_inf /= vanishing_line_m_inf[2]


    def calculate_intersection_of_m_and_w():
        w1, w2, w3, w4, w5, w6 = iac[0, 0], iac[0, 1], iac[1, 1], iac[0, 2], iac[1, 2], iac[2, 2]
        v1, v2, v3 = vanishing_point_axis
        x1, x2, x3 = xinf

        a = v1 ** 2 * w1 + 2 * v1 * v2 * w2 + v2 ** 2 * w3 + 2 * v1 * v3 * w4 + 2 * v2 * v3 * w5 + v3 ** 2 * w6
        b = 2 * v1 * w1 * x1 + 2 * v2 * w2 * x1 + 2 * v3 * w4 * x1 + 2 * v1 * w2 * x2 + 2 * v2 * w3 * x2 + 2 * v3 * w5 * x2 + 2 * v1 * w4 * x3 + 2 * v2 * w5 * x3 + 2 * v3 * w6 * x3
        c = w1 * x1 ** 2 + 2 * w2 * x1 * x2 + w3 * x2 ** 2 + 2 * w4 * x1 * x3 + 2 * w5 * x2 * x3 + w6 * x3 ** 2

        sol1 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        sol2 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        return [sol1, sol2]

    n_circular = calculate_intersection_of_m_and_w()

    def rectification_matrix(ij_point):
        point_in_m_inf = xinf + ij_point * vanishing_point_axis
        point_in_m_inf /= point_in_m_inf[1]

        ci, di = np.real(point_in_m_inf[0]), np.imag(point_in_m_inf[0])

        return np.array([
            [1.0 / di, -ci / di, 0.0],
            [0.0, 1.0, 0.0],
            [vanishing_line_m_inf[0], vanishing_line_m_inf[1], 1.0]
        ])

    M = rectification_matrix(n_circular[0])
    if np.linalg.det(M) < 0:
        M = rectification_matrix(n_circular[1])
    return np.real(M)