import numpy as np
from loguru import logger
from scipy.interpolate import interpn, make_interp_spline, splev, splprep
from scipy.optimize import minimize
from scipy.sparse.csgraph import depth_first_order, minimum_spanning_tree
from scipy.spatial import KDTree
from scipy.stats import median_abs_deviation

from shared.log import timeit
from shared.projective_utils import apply_transformation, get_normal_line_form


@timeit
def sort_by_distance(points, start_i, max_distance):
    kdtree = KDTree(points)
    distance_matrix = kdtree.sparse_distance_matrix(kdtree, max_distance=max_distance)
    path = depth_first_order(minimum_spanning_tree(distance_matrix), i_start=start_i, directed=False,
                             return_predecessors=False)
    return list(path)

def sort_by_distance_to_axis(points, axis):
    dist_from_axis = [np.abs(np.dot(axis, p)) / np.sqrt(axis[0] ** 2 + axis[1] ** 2) for p in points]
    idx = np.argmin(dist_from_axis)
    max_distance = max(np.mean(points[:, :2], axis=1)) * 0.3
    return sort_by_distance(points, idx, max_distance=max_distance)

def generate_and_evaluate_spline(points, t, k=3, s=0, parametrization='chord', weights=None):
    x, y = points[:, :2].T
    p = np.column_stack((x, y))

    def get_u_parametrization():
        if parametrization == 'chord':
            u = np.sum((p[1:, :] - p[:-1, :]) ** 2, axis=1)
            cord = np.sqrt(u).cumsum()
            return np.r_[0, cord]
        elif parametrization == 'centripetal':
            u = np.sum((p[1:, :] - p[:-1, :]) ** 2, axis=1)
            cent = np.cumsum(u ** 0.25)
            return np.r_[0, cent]
        elif parametrization == 'uniform':
            return np.linspace(0, 1, len(p))
        else:
            raise ValueError
    
    u = get_u_parametrization()
    u /= u.max()
    

    if weights is None:
        tck, _ = splprep([x, y], u=u, k=k, s=s)
    else:
        tck, _ = splprep([x, y], w=weights, u=u, k=k, s=s)

    return splev(t, tck), splev(t, tck, der=1)

def generate_spline(points, k=3, parametrization='chord', bc_type='natural'):
    x, y = points[:, :2].T
    p = np.column_stack((x, y))

    def get_u_parametrization():
        if parametrization == 'chord':
            u = np.sum((p[1:, :] - p[:-1, :]) ** 2, axis=1)
            cord = np.sqrt(u).cumsum()
            return np.r_[0, cord]
        elif parametrization == 'centripetal':
            u = np.sum((p[1:, :] - p[:-1, :]) ** 2, axis=1)
            cent = np.cumsum(u ** 0.25)
            return np.r_[0, cent]
        elif parametrization == 'uniform':
            return np.linspace(0, 1, len(p))
        else:
            raise ValueError
        
    u = get_u_parametrization()
    u /= u.max()

    t = np.linspace(0.0, 1.0, len(points))
    xn, yn = interpn((u,), p, t).T
    if k == 1:
        bc_type = None
    return make_interp_spline(t, np.c_[xn, yn], k=k, bc_type=bc_type)

def normalize_range2(points, xrange, yrange):
    xmin, xmax = xrange
    ymin, ymax = yrange

    xdiff, ydiff = xmax-xmin, ymax-ymin

    xyratio = xdiff / ydiff
    
    xs = np.array([(float(i) - xmin) / xdiff * xyratio for i in points[:, 0]])
    ys = np.array([(float(i) - ymin) / ydiff for i in points[:, 1]])
    
    return np.column_stack((xs, ys, np.ones_like(xs)))

def normalize_range(points, xmin=None):
    if xmin is None:
        xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    xdiff = xmax - xmin

    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    ydiff = ymax - ymin

    xyratio = xdiff / ydiff

    xs = np.array([(float(i) - xmin) / xdiff * xyratio for i in points[:, 0]])
    ys = np.array([(float(i) - ymin) / ydiff for i in points[:, 1]])
    
    return np.column_stack((xs, ys, np.ones_like(xs)))

def optimize_axis_of_symmetry(contour: np.array, initial_axis: np.array, camera_matrix: np.array) -> np.array:
    kdtree = KDTree(contour)
    KKT = camera_matrix @ camera_matrix.T
    KKT /= KKT[2, 2]

    def obj_func(xs, points):
        def create_harmonic_homology(axis_of_symmetry):
            return np.identity(3) - 2.0 * KKT @ np.outer(axis_of_symmetry, axis_of_symmetry) / np.dot(axis_of_symmetry, np.dot(KKT, axis_of_symmetry))
        theta, p = xs
        axis_candidate = [np.cos(theta), np.sin(theta), p]
        w = create_harmonic_homology(axis_candidate)
        return sum(kdtree.query(apply_transformation(w, points))[0]) / len(points)

    params = get_normal_line_form(initial_axis)
    logger.trace(f'Axis evaluation Before : {obj_func(params, contour)}')

    axis_angle = params[0]
    lower_bound = (axis_angle - axis_angle * 0.2)
    upper_bound = (axis_angle + axis_angle * 0.2)
    if upper_bound < lower_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    result = minimize(obj_func, params, args=contour, tol=1e-20, options={'maxiter': 30},
                       bounds=((lower_bound, upper_bound), (None, None)))
    logger.trace(f'Axis evaluation After : {obj_func(result.x, contour)}')
    
    estimated_axis = np.array([np.cos(result.x[0]), np.sin(result.x[0]), result.x[1]], dtype=float)
    estimated_axis /= np.linalg.norm(estimated_axis[:2])
    return estimated_axis

def filter_data_by_percentages(data, lower_percentage, upper_percentage):
    data_sorted = np.sort(data)
    n = len(data)

    lower_index = int(lower_percentage * n)
    upper_index = int(upper_percentage * n)

    lower_index = max(lower_index, 0)
    upper_index = min(upper_index, n)

    lower_bound = data_sorted[lower_index]
    upper_bound = data_sorted[upper_index - 1]

    logger.trace(f'Filter Mask thresholds: {lower_bound} <= d <= {upper_bound}')
    return (data >= lower_bound) & (data <= upper_bound)

def create_mask_for_outliers(data, m=2.0):
    return np.abs(0.6745 * (data - np.median(data)) / median_abs_deviation(data)) < m

def filter_outliers(*args, mask_values=None, mask=None, m=2.0, msg='', user=False):
    if mask is None:
        if mask_values is None:
            raise ValueError("Either 'mask' must be provied, or 'mask_values'.")
        else:
            mask = create_mask_for_outliers(mask_values, m)
    
    if not all(mask):
        logger.log('FILTER' if not user else 'FILTER-USER', f'{msg} Processed {sum(mask)} data points.')
        return tuple(arg[mask] for arg in args) if len(args) > 1 else args[mask]
    else:
        return args if len(args) > 1 else args[0]

def filter_function_unique(data, msg=''):
    filtered_indices = []
    unique_y = np.unique(-data[:, 1])
    unique_y = -unique_y
    for y in unique_y:
        same_y_indices = np.where(data[:, 1] == y)[0]
        if len(same_y_indices) > 0:
            max_x = np.max(data[same_y_indices, 0])
            max_x_indices = same_y_indices[data[same_y_indices, 0] == max_x][0]
            filtered_indices.append(max_x_indices)
    if np.array_equal(list(range(len(data))), filtered_indices):
        return True, None

    logger.trace(f'{msg}')
    return False, np.array(filtered_indices)
