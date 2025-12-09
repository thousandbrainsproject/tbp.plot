# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class WidgetAction:
    """A scheduled animation action executed by a `WidgetAnimator`.

    `WidgetAction` represents a single time-triggered event in an animation
    sequence. When the animation reaches the specified `time` (in seconds since
    the animation started), the associated `func` is invoked. This is typically
    used to update the state of a widget (e.g., sliders, buttons) in a scripted
    and reproducible way during an interactive vedo visualization.

    Attributes:
        time: The time offset in seconds at which this action should be
            executed relative to the start of the animation.
        func: A zero-argument callable that performs the desired state update
            or side effect. This is usually a bound method like
            `widget.set_state(value)`.
    """

    time: float
    func: Callable[[], None]


class WidgetAnimator:
    """Animate widget state changes over time using a VTK/vedo timer.

    `WidgetAnimator` schedules and executes a sequence of `WidgetAction` objects
    during an interactive vedo session. Each action is invoked at a precise time
    offset relative to when the animation starts, allowing the visualization to
    behave as if a user were manually interacting with widgets (e.g., sliders,
    buttons) in a smooth, time-controlled manner. The animator uses a repeating
    VTK timer to poll elapsed time and trigger pending actions.

    Args:
        plotter: The `vedo.Plotter` instance whose interactor will host the timer
            callback. This is typically the main plotter used in an
            `InteractivePlot`.
        actions: A sequence of scheduled actions sorted by time. If unsorted,
            the constructor will sort them by `action.time`. Each action's `func`
            is invoked when its scheduled time is reached.
        timer_dt_ms: Interval in milliseconds at which the timer callback is
            invoked. Defaults to 50 ms (20 Hz).

    Attributes:
        plotter: The vedo plotter associated with this animator.
        actions: The sorted list of scheduled actions.
        timer_dt_ms: Timer interval in milliseconds.
        timer_id: The VTK timer ID returned by the plotter when the timer is created.
            Used for destroying the timer later.
        _start_time: The absolute time (in seconds) at which the animation started.
        _next_idx: Index of the next action to execute.
        _running: Whether the animation is currently active.

    """

    def __init__(
        self,
        plotter,
        actions: list[WidgetAction],
        timer_dt_ms: int = 50,
    ) -> None:
        self.plotter = plotter
        self.actions = sorted(actions, key=lambda a: a.time)
        self.timer_dt_ms = timer_dt_ms

        self.timer_id: int | None = None
        self._start_time: float | None = None
        self._next_idx: int = 0
        self._running: bool = False

    def start(self) -> None:
        """Start the animation."""
        if not self.actions:
            return

        self._start_time = time.perf_counter()
        self._next_idx = 0
        self._running = True

        # Register timer callback only once.
        self.plotter.add_callback("timer", self._on_timer)

        # Create a repeating timer if not already created.
        if self.timer_id is None:
            self.timer_id = self.plotter.timer_callback(
                "create",
                dt=self.timer_dt_ms,
            )

    def stop(self) -> None:
        """Stop the animation and destroy the timer."""
        if not self._running:
            return
        self._running = False

        if self.timer_id is not None:
            self.plotter.timer_callback("destroy", timer_id=self.timer_id)
            self.timer_id = None

    def _on_timer(self, evt) -> None:
        """Timer callback invoked by vedo."""
        if not self._running or self._start_time is None:
            return

        current_time = time.perf_counter() - self._start_time

        # Execute all actions whose scheduled time has passed.
        while (
            self._next_idx < len(self.actions)
            and current_time >= self.actions[self._next_idx].time
        ):
            action = self.actions[self._next_idx]
            action.func()
            self._next_idx += 1

        # Stop when done.
        if self._next_idx >= len(self.actions):
            self.stop()
