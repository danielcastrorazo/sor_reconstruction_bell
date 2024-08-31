from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shared.ellipse_utils import (
    cart_to_pol,
    coeff_to_matrix_q_eq,
    get_ellipse_pts,
    matrix_q_eq_to_coeff,
    pol_to_cart,
)


@dataclass
class _Ellipse:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle: float

    def apply_transformation(self, T):
        C = self.get_matrix_form()
        C = np.linalg.inv(T).T @ C @ np.linalg.inv(T)
        C /= C[2, 2]

        coefficients = matrix_q_eq_to_coeff(C)

        self.set_coefficients(coefficients)

    def set_coefficients(self, coefficients):
        x0, y0, ap, bp, phi = cart_to_pol(coefficients)
        self.center = (x0, y0)
        self.axes = (ap, bp)
        self.angle = phi

    def get_pol_to_cart(self):
        return pol_to_cart((*self.center, *self.axes, self.angle))

    def get_matrix_form(self):
        matrix = coeff_to_matrix_q_eq(self.get_pol_to_cart())
        return matrix / matrix[2, 2]

    def set_matrix_form(self, C):
        coefficients = matrix_q_eq_to_coeff(C)
        self.set_coefficients(coefficients)

    def get_points(self, npts=100, tmin=0.0, tmax=2*np.pi):
        return get_ellipse_pts((*self.center, *self.axes, self.angle), npts, tmin, tmax)

    def __post_init__(self):
        if isinstance(self.center, list) and isinstance(self.axes, list):
            self.center = tuple(self.center)
            self.axes = tuple(self.axes)

    def __str__(self):
        a, b, c, d, e, f = self.get_pol_to_cart()
        print(f'{a:.15f} x^2 + {b:.15f} x y + {c:.15f} y^2 + {d:.15f} x + {e:.15f} y + {f:.15f} = 0')


@dataclass
class ImageMetadata:
    source: str
    image_size: tuple[int, int]
    scale: float
    sensor_dimensions: tuple[float, float]
    focal_length: float
    mask: dict[str, Any]
    ellipses: list[_Ellipse] = field(default_factory=list)
    """
    Class representing metadata for an image.

    Attributes:
        source: Source of the image.
        image_size: Dimensions of the image in (height, width).
        scale: The height of the bell is from 0.0 to 1.0, the scale is the max x value from 0.0
        sensor_dimensions: Physical dimensions of the camera sensor(width, height) in millimeters.
        focal_length: Focal length of the camera in millimeters.
        mask: Dictionary representing a mask applied to the image. Pycocotools encoding.
        ellipses: List of ellipses detected in the image.
    """

    def __post_init__(self):
        self.ellipses = [_Ellipse(**ellipse) if isinstance(ellipse, dict) else ellipse for ellipse in self.ellipses]

    def update_image_metadata(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.image_metadata, key, value)