import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.interpolate import interp1d
from trimesh import transformations
from trimesh.visual import texture

from shared.optimization import generate_spline, normalize_range
from shared.projective_utils import get_angle_of_line, get_bounding_box


def create_model(curve, tangents, phi_n=64, image=None, image_mask=None, axis=None):
    vertices, vertex_normals, faces, uvs = prepare_model(curve, tangents, phi_n)
    
    if image is None:
        mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces,
                                visual=texture.TextureVisuals(uv=uvs), process=True, validate=True)
    else:
        tex = extract_texture_cylinder(image, image_mask, np.copy(curve), axis)
        tex = Image.fromarray(tex)
        mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces,
                                visual=texture.TextureVisuals(uv=uvs,image=tex), process=True, validate=True)
    rot_matrix = transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0], [0, 0, 0])
    mesh.apply_transform(rot_matrix)
    return mesh


def generate_model(data, parametrization='chord', outer_detail=128, inner_detail=32):
    curve = generate_spline(data[1:], k=3, parametrization=parametrization)
    
    t = np.linspace(0.0, 1.0, outer_detail - 1)
    outer_curve = np.vstack((curve(t).T, np.ones_like(t))).T
    outer_tangent = np.vstack((curve.derivative()(t).T, np.zeros_like(t))).T
    
    outer_curve[np.argmax(outer_curve[:, 1])][1] = 1.0

    first_point_curve = np.array([0.0, outer_curve[0, 1], 1.0])
    first_tangent_curve = np.array([1, 0, 0])

    outer_curve = np.vstack((first_point_curve, outer_curve))
    outer_tangent = np.vstack((first_tangent_curve, outer_tangent))

    curve = generate_spline(outer_curve, k=3, parametrization=parametrization)
    t = np.linspace(0.0, 1.0, inner_detail)
    inner_curve = np.vstack((curve(t).T, np.ones_like(t))).T
    inner_tangent = np.vstack((curve.derivative()(t).T, np.zeros_like(t))).T

    def get_inner_parallel_curve(x, y, d=1):
        gradient_x, gradient_y = np.gradient(x), np.gradient(y)
        normals = np.c_[gradient_y, -gradient_x]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return np.c_[x + normals[:, 0] * d, y + normals[:, 1] * d, np.ones_like(x)]
    
    inner_curve = get_inner_parallel_curve(inner_curve[:, 0], inner_curve[:, 1], d=0.05)

    mask = inner_curve[:, 1]  >= 0.0
    inner_curve, inner_tangent = inner_curve[mask][::-1], inner_tangent[mask][::-1]

    inner_curve[-1][0] = 0.0

    inner_tangent = -inner_tangent
    inner_tangent /= np.linalg.norm(inner_tangent, axis=1)[:, np.newaxis]
    outer_tangent /= np.linalg.norm(outer_tangent, axis=1)[:, np.newaxis]

    full_curve = np.vstack((outer_curve, inner_curve))
    full_tangents = np.vstack((outer_tangent, inner_tangent))

    full_curve = normalize_range(full_curve, xmin=0.0)

    return full_curve, full_tangents, outer_curve, inner_curve, outer_tangent, inner_tangent

def extract_texture_cylinder(image, mask, curve, axis):

    curve[:, 1] = 1.0 - curve[:, 1]

    x_values = curve[:, 1]
    y_values = curve[:, 0]

    sorted_indices = np.argsort(x_values)
    x_values = x_values[sorted_indices]
    y_values = y_values[sorted_indices]

    (top, left), (bottom, right) = get_bounding_box(mask)

    theta = get_angle_of_line(axis)
    if np.abs(theta - np.pi / 2) > 1e-10:
        if theta > 0.0:
            diff = theta - np.pi / 2
        else:
            diff = theta + np.pi / 2

        center = ((left + (right - left) / 2), top + (bottom - top) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, np.degrees(diff), 1.0)

        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rot_mat, mask.shape[1::-1], flags=cv2.INTER_NEAREST)

    (top, left), (bottom, right) = get_bounding_box(mask)
    image = image[top:bottom+1, left:right+1]
    mask = mask[top:bottom+1, left:right+1]

    mask_height = np.count_nonzero(np.any(mask, axis=1)) 
    mask_width = np.count_nonzero(np.any(mask, axis=0))
    
    nfirst = np.where(mask[:, 0])[0][0]
    nlast = np.where(mask[:, -1])[0][0]
    end = (nfirst + nlast) // 2

    curve_radii, curve_heights = curve[:, :2].T 
    curve_radii *= mask_width / (2 * max(curve_radii))
    curve_heights *= end 
    max_radii = max(curve_radii)

    # theta_ = np.linspace(-np.pi, 0, mask_width) # This one for the pictures.
    theta_ = np.linspace(0, 2 * np.pi, mask_width)

    x = np.linspace(max_radii, 0, mask_height - end - 1)
    
    end_interp = interp1d(np.arange(end + 1, mask_height), x, kind='cubic', bounds_error=False, fill_value=0)
    radius_interp = interp1d(curve_heights, curve_radii)

    tex2 = np.zeros((mask_height, mask_width, 4), dtype=image.dtype)
    for z in range(mask_height):
        r = radius_interp(z) if z <= end else end_interp(z)
        for theta in range(mask_width):
            angle = theta_[theta]
            x = np.clip(int(r * np.cos(angle)) + mask_width // 2, 0, mask_width - 1)
            if not mask[z, x]:
                continue
            tex2[z, theta] = image[z, x]
    return cv2.cvtColor(np.flipud(tex2).astype(np.float32), cv2.COLOR_BGRA2RGBA).astype(np.uint8)


def prepare_model(xy, dxy,  phi_n=64):
    t = np.linspace(0, np.pi * 2, phi_n, endpoint=True)

    # Vertex    
    xn = np.outer(xy[:, 0], np.cos(t))
    yn = np.outer(xy[:, 0], np.sin(t))
    zn = np.outer(xy[:, 1], np.ones(np.size(t)))
    vertices = np.c_[xn.ravel(order='F'), yn.ravel(order='F'), zn.ravel(order='F')]

    # Vertex normals
    vertex_normals = np.array([np.cross(pi, ti) for pi, ti in zip(xy, dxy)])
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]
    xnn = np.outer(vertex_normals[:, 0], np.cos(t))
    ynn = np.outer(vertex_normals[:, 0], np.sin(t))
    znn = np.outer(vertex_normals[:, 1], np.ones(np.size(t)))
    vertex_normals = np.c_[xnn.ravel(order='F'), ynn.ravel(order='F'), znn.ravel(order='F')]

    # Faces
    rows, cols = xn.shape
    mrc = rows * cols
    faces = []
    for c in range(cols):
        vertex_ids = np.arange(rows - 1, 0, -1) + rows * c
        for vid in vertex_ids:
            faces.append((vid, (vid + rows) % mrc, vid - 1))
            faces.append(((vid - 1 + rows) % mrc, vid - 1, (vid + rows) % mrc))
    # UV
    cumulative_lengths = np.cumsum(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1)))
    s = cumulative_lengths / cumulative_lengths[-1]
    u, v = np.meshgrid(1.0 - t / (2 * np.pi), np.r_[0, s])
    uvs = np.column_stack((u.ravel(order='F'), v.ravel(order='F')))

    return vertices, vertex_normals, faces, uvs