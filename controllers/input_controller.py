import colorsys

import cv2
import numpy as np

from controllers.data_controller import (
    CDCState,
    DataController,
    Observable,
)
from shared.ellipse_utils import cart_to_pol, fit_ellipse, get_ellipse_pts


class InputController(Observable):
    def __init__(self, data_controller : DataController):
        super().__init__()

        self.data_controller = data_controller
        self.data_controller.attach(self)

        self.max_ellipses = 0
        self.loaded_ellipses_points = []
        self.bg_image = None
        self.checkers = None
        self.current = []

        self.t_line = False
        self.t_line_pts = []

        self.interactive = True

    def _generate_distinct_colors(self, n):
        colors = []
        golden_ratio_conjugate = 0.618033988749895
        for i in range(1, n + 1):
            hue = (i * golden_ratio_conjugate) % 1
            r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.95)
            colors.append((int(255 * r), int(255 * g), int(255 * b), 255))
        return colors

    def update(self, event, *args, **kwargs):
        if self.data_controller.state == CDCState.NEW_DATA or self.data_controller.state == CDCState.DATA_LOADED:
            if event == "change_of_state":
                self.loaded_ellipses_points = []
                self._update_data()
                self.draw()
        elif self.data_controller.state == CDCState.DEFAULT_SYSTEM_LOAD:
            self.max_ellipses = self.data_controller["subprocess"]["max_ellipses"]
            self.COLORS = self._generate_distinct_colors(self.max_ellipses)
            if event == "disable_edits":
                self.interactive = False
        self.notify_observers(event)

    def _update_data(self):
        self.image = self.data_controller["image"]
        self.mask = self.data_controller["mask"]
        self.contours = self.data_controller["contour"]

        if self.data_controller.get_config('show_checkers'):
            self.checkers = np.zeros_like(self.image)
            self.checkers[:,:,3] = 255
            self.checkers = self._add_checkerboard_pattern(self.checkers)

        if self.data_controller.state == CDCState.DATA_LOADED:
            for ellipse in self.data_controller["metadata"].ellipses:
                points = np.asarray([(int(x), int(y)) for x, y in zip(*ellipse.get_points())])
                params = (*ellipse.center, *ellipse.axes, ellipse.angle)
                self.loaded_ellipses_points.append((points, params))

    def draw(self, cx=None, cy=None):
        if self.checkers is not None:
            self.bg_image = self.checkers.copy()
            cv2.bitwise_and(self.image, self.image, mask=self.mask, dst=self.bg_image)
        else:
            self.bg_image = cv2.bitwise_and(self.image, self.image, mask=self.mask)

        extra = False
        if cx and cy and (cx, cy) not in self.current:
            extra = True
            self.current.append((cx, cy))

        for i, (points, _) in enumerate(self.loaded_ellipses_points):
            cv2.polylines(self.bg_image, [points], isClosed=True, color=self.COLORS[i], thickness=0)

        ci = len(self.loaded_ellipses_points)
        for x, y in self.current:
            cv2.circle(self.bg_image, (x, y), 1, self.COLORS[ci], -1)

        if self.t_line and cx is not None:
            cv2.line(self.bg_image, self.t_line_pts[0], (cx, cy), color=self.COLORS[ci], thickness=1)
        else:
            if len(self.current) >= 4:
                try:
                    coefficients = fit_ellipse(*np.asarray(self.current).T)
                    params = cart_to_pol(coefficients)
                    f = np.asarray([(int(x), int(y)) for x, y in zip(*get_ellipse_pts(params))])
                    cv2.polylines(self.bg_image, [f], isClosed=True, color=self.COLORS[ci], thickness=0)
                except Exception:
                    pass

        if extra:
            self.current.pop()

    def _is_interactive(self):
        if not self.interactive:
            return False
        
        state = self.data_controller.state
        if state == CDCState.DATA_LOADED or state == CDCState.NEW_DATA:
            return self.max_ellipses > len(self.loaded_ellipses_points)
        return False

    def on_left_click(self, x, y):
        if not self._is_interactive():
            return
        if cv2.pointPolygonTest(self.contours[:, :2], (x, y), False) == -1.0:
            return
        if self.t_line:
            point1 = np.array([self.t_line_pts[0][0], self.t_line_pts[0][1]])
            point2 = np.array([x, y])
            points = np.linspace(point1, point2, 98 + 2)[1:-1]
            for x, y in points:
                self.current.append((int(x), int(y)))
            self.t_line_pts.clear()
            self.t_line = False
        else:
            self.current.append((x, y))
        self.draw()
        self.notify_observers()

    def on_middle_click(self, x, y):
        if not self._is_interactive():
            return
        if cv2.pointPolygonTest(self.contours[:, :2], (x, y), False) == -1.0:
            return
        self.t_line = True
        self.t_line_pts = [(x, y)]
        self.draw()
        self.notify_observers()

    def on_right_click(self):
        if not self._is_interactive():
            return
        if len(self.current):
            self.current.pop()
            self.draw()
            self.notify_observers()

    def on_mouse_move(self, x, y):
        if not self._is_interactive():
            return
        if cv2.pointPolygonTest(self.contours[:, :2], (x, y), False) == -1.0:
            self.draw()
        else:
            self.draw(x, y)
        self.notify_observers()

    def on_leaving_canvas(self):
        if not self._is_interactive():
            return
        self.draw()
        self.notify_observers()
    
    def save_new_ellipse(self):
        if len(self.current) < 5:
            return
        try:
            coefficients = fit_ellipse(*np.asarray(self.current).T)
            params = cart_to_pol(coefficients)
            points = np.asarray([(int(x), int(y)) for x, y in zip(*get_ellipse_pts(params))])
            self.loaded_ellipses_points.append((points, params))
        except:
            return
        self.current.clear()
        self.draw()
        self.notify_observers(event="ellipse_change")


    def delete_ellipse(self, idx):
        self.loaded_ellipses_points.pop(idx)
        self.draw()
        self.notify_observers(event="ellipse_change")

    def make_ellipse_first(self, idx):
        self.loaded_ellipses_points.insert(0, self.loaded_ellipses_points.pop(idx))
        self.draw()
        self.notify_observers(event="ellipse_change")

    def make_ellipse_last(self, idx):
        self.loaded_ellipses_points.insert(len(self.loaded_ellipses_points) - 1, self.loaded_ellipses_points.pop(idx))
        self.draw()
        self.notify_observers(event="ellipse_change")

    def _add_checkerboard_pattern(self, image):
        height, width = image.shape[:2]
        stripe_width = 100

        x_coords = np.arange(width)
        y_coords = np.arange(height)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        checkerboard = ((x_grid // stripe_width + y_grid // stripe_width) % 2 == 0)

        image[checkerboard, :3] = (150, 150, 150)
        image[~checkerboard, :3] = (200, 200, 200)

        return image
