# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import time
from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    from vedo import Plotter

logger = logging.getLogger(__name__)


def use_headless_matplotlib() -> None:
    """Switch matplotlib to the non-interactive Agg backend.

    Vedo-based plots rasterize matplotlib figures into `vedo.Image` widgets and
    never call `plt.show()`, so they have no use for a GUI backend. With one
    selected, every figure additionally builds a Qt or Tk figure manager, which
    competes with VTK for the platform event loop. On macOS that is a native
    crash. Agg produces identical pixels without touching a GUI toolkit.

    Call this from a plot entry point, before any figure is created. It has no
    effect on plots that genuinely call `plt.show()`, since only the invoked
    plot runs in a given process.
    """
    mpl.use("Agg", force=True)


def run_interactor(
    plotter: Plotter,
    scheduler: DebounceScheduler | None = None,
    poll_sec: float = 0.01,
) -> None:
    """Process VTK events until the window closes, then shut down cleanly.

    This replaces a blocking `show(interactive=True)`. Under macOS the native
    Cocoa interactor renders the window but never delivers events to it when
    Python is not a framework build, which is the usual case under `uv` and
    `pip`, leaving the controls frozen and the window unclosable. Processing
    events explicitly keeps the window responsive, lets Ctrl-C through, and
    guarantees the plotter is closed on the way out.

    Args:
        plotter: Plotter whose interactor drives the loop. Every renderer must
            already have been shown with `interactive=False`.
        scheduler: Debounce scheduler to service on each pass, if the plot has
            one. Its due callbacks fire from this loop.
        poll_sec: Idle time in seconds between event-processing passes. Also
            bounds how late a debounced callback can fire. Lower values feel
            more responsive at the cost of more idle CPU.
    """
    interactor = plotter.interactor
    if interactor is None:
        return

    interactor.Initialize()
    interactor.Enable()

    try:
        while not interactor.GetDone():
            interactor.ProcessEvents()
            if scheduler is not None:
                scheduler.poll()
            time.sleep(poll_sec)
    except KeyboardInterrupt:
        logger.info("Visualization interrupted")
    finally:
        if scheduler is not None:
            scheduler.shutdown()
        plotter.close()


@dataclass
class Location2D:
    """2D location wrapper.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
    """

    x: float
    y: float

    def __eq__(self, other: object) -> bool:
        """Return True only if same class and all values match exactly."""
        if other.__class__ is self.__class__:
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def to_3d(self, z: float) -> Location3D:
        """Create a 3D location by adding a z coordinate.

        Args:
            z: Z coordinate value.

        Returns:
            A new Location3D with (x, y, z).
        """
        return Location3D(self.x, self.y, z)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Convert to a NumPy array in the form [x, y].

        Returns:
            A NumPy array with shape (2,).
        """
        return np.array([self.x, self.y], dtype=float)


@dataclass
class Location3D:
    """3D location wrapper.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
        z: Z coordinate.
    """

    x: float
    y: float
    z: float

    def __eq__(self, other: object) -> bool:
        """Return True only if same class and all values match exactly."""
        if other.__class__ is self.__class__:
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def to_2d(self) -> Location2D:
        """Drop the z coordinate and return a 2D location.

        Returns:
            A new Location2D with (x, y).
        """
        return Location2D(self.x, self.y)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Convert to a NumPy array in the form [x, y, z].

        Returns:
            A NumPy array with shape (3,).
        """
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class Bounds:
    """rectangle bounds described by (xmin, xmax, ymin, ymax).

    Attributes:
        xmin: Minimum x value.
        xmax: Maximum x value.
        ymin: Minimum y value.
        ymax: Maximum y value.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def width(self) -> float:
        return self.xmax - self.xmin

    def height(self) -> float:
        return self.ymax - self.ymin

    def contains(self, location: Location2D) -> bool:
        """Check whether a 2D point lies inside or on the boundary.

        Args:
            location: Location to test.

        Returns:
            True if x and y values lie within the rectangle boundary.
        """
        return (self.xmin <= location.x <= self.xmax) and (
            self.ymin <= location.y <= self.ymax
        )


@dataclass
class CoordinateMapper:
    """Map between GUI click coordinates and data coordinates, and back.

    Attributes:
        gui: Bounds for the GUI rectangle.
        data: Bounds for the data rectangle.

    Raises:
        ValueError: If any provided bounds have nonpositive width or height.
    """

    gui: Bounds
    data: Bounds

    def __post_init__(self) -> None:
        if self.gui.width() <= 0 or self.gui.height() <= 0:
            raise ValueError("GUI bounds must have positive width and height")
        if self.data.width() <= 0 or self.data.height() <= 0:
            raise ValueError("Data bounds must have positive width and height")

    def map_click_to_data_coords(self, gui_location: Location2D) -> Location2D:
        """Map a GUI click (x, y) to data coordinates (x_val, y_val).

        Args:
            gui_location: Click location in GUI coordinates.

        Returns:
            Location of the click point in data coordinates.
        """
        x_rel = (gui_location.x - self.gui.xmin) / self.gui.width()
        y_rel = (gui_location.y - self.gui.ymin) / self.gui.height()

        x_val = self.data.xmin + x_rel * self.data.width()
        y_val = self.data.ymin + y_rel * self.data.height()
        return Location2D(x_val, y_val)

    def map_data_coords_to_world(self, data_location: Location2D) -> Location2D:
        """Map data coordinates (x, y) back to GUI coordinates (x_gui, y_gui).

        Args:
            data_location: Point in data coordinates.

        Returns:
            Location of the point in GUI coordinates.
        """
        x_rel = (data_location.x - self.data.xmin) / self.data.width()
        y_rel = (data_location.y - self.data.ymin) / self.data.height()

        x_gui = self.gui.xmin + x_rel * self.gui.width()
        y_gui = self.gui.ymin + y_rel * self.gui.height()
        return Location2D(x_gui, y_gui)


class DebounceScheduler:
    """Registry of debounced callbacks, serviced by an external event loop.

    Debouncing restricts how often a function is called by waiting for a specified
    period of inactivity after an event occurs before executing the callback. This
    way we can keep changing the widget states, but it won't call the expensive
    callback until we stop changing the widget state for some time (i.e.,
    `debounce sec`) to ensure the UI stays responsive.

    The scheduler holds a registry of callbacks that are scheduled to run once at
    or after a given time, each keyed by a hashable token. It has no clock of its
    own: the owner drives it by calling `poll` regularly, which `run_interactor`
    does on every pass of its event loop.

    Attributes:
        _callbacks: Mapping from keys to callbacks.
        _due: Mapping from keys to ready times in seconds.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        self._callbacks: dict[Hashable, Callable[[], None]] = {}
        self._due: dict[Hashable, float] = {}

    def register(self, key: Hashable, callback: Callable[[], None]) -> None:
        """Register a callback under a key.

        Args:
            key: Unique hashable key for the callback.
            callback: callback function to invoke when due.
        """
        self._callbacks[key] = callback

    def schedule_once(self, key: Hashable, delay_sec: float) -> None:
        """Schedule a registered callback to run after a delay.

        Args:
            key: Key of a previously registered callback.
            delay_sec: Delay in seconds. If less than or equal to zero, schedule
                immediately.

        Raises:
            KeyError: If the key is not registered.
        """
        if key not in self._callbacks:
            raise KeyError("Key not registered with scheduler")
        now = time.perf_counter()
        self._due[key] = now if delay_sec <= 0 else now + delay_sec

    def cancel(self, key: Hashable) -> None:
        """Cancel a scheduled callback and remove it from the registry.

        Args:
            key: Key for the callback to cancel.
        """
        self._due.pop(key, None)
        self._callbacks.pop(key, None)

    def shutdown(self) -> None:
        """Clear all scheduled and registered callbacks."""
        self._due.clear()
        self._callbacks.clear()

    def poll(self) -> None:
        """Invoke every callback whose delay has elapsed.

        Safe to call as often as desired; each due callback fires once because
        its entry is removed before it runs.
        """
        if not self._due:
            return
        now = time.perf_counter()
        ready = [k for k, t in list(self._due.items()) if now >= t]
        for key in ready:
            self._due.pop(key, None)
            cb = self._callbacks.get(key)
            if cb:
                cb()


def trace_hypothesis_backward(
    ix: int, removed_ids: Iterable[int], added_ids: Iterable[int]
) -> int | None:
    """Trace a hypothesis index backward one step in time.

    This function reconstructs the index of a hypothesis at step `t-1`
    given its index at step `t`, using the bookkeeping of which indices
    were removed and which were newly added.

    If `ix` corresponds to a newly added hypothesis at step `t`,
    the hypothesis did not exist at step `t-1` -> returns `None`.
    Otherwise, reinserts the slots removed in the transition
    (t-1 -> t), which shifts the index to the right by one for each
    removed index less than or equal to it.

    Note: This assumes that `added_ids` are always appended to the end of
    the hypotheses at `t-1`, therefore cannot shift the tracked index when
    tracing backwards. This simplifies the computations as we only need to
    check if the tracked index is in the `added_ids` or not, but we do not need
    to shift the tracked index based on `added_ids`.

    Args:
        ix: Index of the hypothesis at step t.
        removed_ids: Sorted sequence of indices that were removed in
            the transition (t-1 -> t).
        added_ids: Sorted sequence of indices that were newly added at step t.

    Returns:
        The index of the hypothesis at step (t-1), or `None` if the
        hypothesis was newly added at step t.
    """
    # NOTE: This is similar to checking `if ix in added_ids` but runs faster
    # O(log n) because it makes use of the sorted list (binary search).
    added_pos = bisect_left(added_ids, ix)
    if added_pos < len(added_ids) and added_ids[added_pos] == ix:
        return None

    i_prev = ix
    for r in removed_ids:
        if r <= i_prev:
            i_prev += 1
        else:
            break
    return i_prev


def trace_hypothesis_forward(ix: int, removed_ids: Iterable[int]) -> int | None:
    """Trace a hypothesis index forward one step in time.

    This function computes the index of a hypothesis at step `t+1` given
    its index at step `t`, using the list of indices removed during the
    transition (t -> t+1).

    If the current index `ix` is removed, the hypothesis ceases to exist -> returns
    `None`. Otherwise, the index shifts left by the number of removed indices less
    than `ix`.

    Args:
        ix: Index of the hypothesis at step t.
        removed_ids: Sorted sequence of indices that were removed in the
            transition (t -> t+1).

    Returns:
        The index of the hypothesis at step (t+1), or `None` if the
        hypothesis was removed.
    """
    pos = bisect_left(removed_ids, ix)
    if pos < len(removed_ids) and removed_ids[pos] == ix:
        return None
    return ix - pos


def rotate_about_pivot(
    rotation: Rotation,
    points: npt.NDArray[np.float64],
    pivot: npt.NDArray[np.float64],
) -> np.ndarray:
    """Rotate 3D point(s) about a fixed pivot.

    Applies an active rotation around a specified pivot point. The rotation
    is performed by translating the points so that the pivot is at the origin,
    applying the rotation, then translating back.

    p_rot = pivot + R @ (p - pivot)

    Args:
        rotation: A SciPy Rotation object representing the rotation to apply.
        points: A single 3D point with shape (3,) or an array of points with
            shape (N, 3).
        pivot: The pivot point to rotate about, with shape (3,).

    Returns:
        A NumPy array of rotated point(s) with the same shape as `points`.

    Raises:
        ValueError: If `points` does not have shape (3,) or (N, 3).
    """
    points = np.asarray(points, dtype=float)
    pivot = np.asarray(pivot, dtype=float)

    if (points.shape == (3,)) or (points.ndim == 2 and points.shape[1] == 3):
        return pivot + rotation.apply(points - pivot)

    raise ValueError(
        f"`points` must have shape (3,) or (N, 3); got shape {points.shape}"
    )
