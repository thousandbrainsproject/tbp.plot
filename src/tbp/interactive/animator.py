# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from collections.abc import Callable, Hashable
from dataclasses import dataclass

from tbp.interactive.utils import VtkDebounceScheduler


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
    """Animate widget state changes using an existing VTK debounce scheduler.

    This class drives scripted widget interactions by scheduling `WidgetAction`
    callables through a shared `VtkDebounceScheduler`. Each action is executed
    once at a specified time offset, allowing the visualization to behave as if
    a user were manually interacting with widgets (e.g., sliders or buttons)
    in a smooth, reproducible manner.

    Args:
        scheduler: A `VtkDebounceScheduler` instance used to schedule animation
            callbacks within the VTK event loop.
        actions: A sequence of time-stamped actions to execute. Actions are sorted
            by their `time` attribute before scheduling.
        key_prefix: Prefix used to construct unique keys for scheduled callbacks
            in the scheduler. Defaults to `"widget_animator"`.

    Attributes:
        scheduler: The debounce scheduler used to schedule animation callbacks.
        actions: The sorted list of scheduled actions.
        key_prefix: Prefix used when generating scheduler keys.
        _running: Whether the animator is currently active.
    """

    def __init__(
        self,
        scheduler: VtkDebounceScheduler,
        actions: list[WidgetAction],
        key_prefix: Hashable = "widget_animator",
    ) -> None:
        self.scheduler = scheduler
        self.actions = sorted(actions, key=lambda a: a.time)
        self.key_prefix = key_prefix

        self._running: bool = False

    def start(self) -> None:
        """Start the animation by scheduling all actions.

        Each `WidgetAction` is registered with the scheduler and scheduled to
        execute once at its specified time offset.
        """
        if not self.actions:
            return

        for idx, action in enumerate(self.actions):
            key = (self.key_prefix, idx)
            self.scheduler.register(
                key,
                lambda i=idx, self=self: self._running and self.actions[i].func(),
            )
            self.scheduler.schedule_once(key, delay_sec=action.time)

        self._running = True

    def stop(self) -> None:
        """Stop the animation and cancel all scheduled actions.

        This method prevents any pending actions from executing and removes
        all associated callbacks from the scheduler.
        """
        if not self._running:
            return

        for idx in range(len(self.actions)):
            key = (self.key_prefix, idx)
            self.scheduler.cancel(key)

        self._running = False


def make_slider_step_actions_for_widget(
    *,
    widget,
    start_value: float,
    stop_value: float,
    num_steps: int,
    step_dt: float,
) -> list[WidgetAction]:
    """Generate time-scheduled slider step actions for a widget.

    This helper creates a sequence of `WidgetAction` objects that gradually set
    the state of the given widget from `start_value` to `stop_value` in uniform
    increments. Each action is scheduled at a fixed time offset, forming a smooth,
    evenly paced animation when executed by a `WidgetAnimator`.

    Args:
        widget: The widget whose `.set_state()` method will be invoked at each step.
        start_value: Initial slider value at time zero.
        stop_value: Final slider value at the last step.
        num_steps: Number of interpolation steps, including both endpoints.
            Must be >= 2. The slider values will be linearly spaced across these steps.
        step_dt: Time interval in seconds between consecutive actions.

    Returns:
        list[WidgetAction]: A list of actions, sorted by increasing time, where each
        action updates the widget's state to a specific intermediate value.
    """
    if num_steps < 2:
        return []

    delta = (stop_value - start_value) / (num_steps - 1)
    actions: list[WidgetAction] = []

    for i in range(num_steps):
        t = i * step_dt
        value = start_value + i * delta

        actions.append(
            WidgetAction(
                time=t,
                func=lambda val=value, w=widget: w.set_state(val),
            )
        )

    return actions
