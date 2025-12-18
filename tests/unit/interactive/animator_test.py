# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import unittest
from unittest.mock import patch

from vtk import vtkRenderWindowInteractor

from tbp.interactive.animator import (
    WidgetAction,
    WidgetAnimator,
    make_slider_step_actions_for_widget,
)
from tbp.interactive.utils import VtkDebounceScheduler


class FakeWidget:
    """A widget with a set_state method to capture assigned values."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def set_state(self, v: float) -> None:
        self.values.append(v)


class TestWidgetAnimator(unittest.TestCase):
    def setUp(self) -> None:
        self.iren = vtkRenderWindowInteractor()
        self.scheduler = VtkDebounceScheduler(self.iren, period_ms=33)

    def _tick_scheduler(self) -> None:
        """Manually tick the scheduler once (simulate a timer event)."""
        self.scheduler._on_timer(None, "TimerEvent")

    def test_start_no_actions_is_noop(self) -> None:
        animator = WidgetAnimator(self.scheduler, actions=[], key_prefix="k")
        animator.start()

        self.assertFalse(animator._running)
        self.assertEqual(self.scheduler._callbacks, {})
        self.assertEqual(self.scheduler._due, {})

    def test_start_registers_and_schedules_all_actions_sorted(self) -> None:
        called: list[str] = []

        actions = [
            WidgetAction(time=1.0, func=lambda: called.append("t1.0")),
            WidgetAction(time=0.2, func=lambda: called.append("t0.2")),
            WidgetAction(time=0.5, func=lambda: called.append("t0.5")),
        ]
        animator = WidgetAnimator(self.scheduler, actions=actions, key_prefix="anim")

        with patch("time.perf_counter", lambda: 0.0):
            animator.start()

        self.assertTrue(animator._running)

        # Keys must exist for all actions.
        for idx in range(3):
            self.assertIn(("anim", idx), self.scheduler._callbacks)
            self.assertIn(("anim", idx), self.scheduler._due)

        # Due times should be sorted: 0.2, 0.5, 1.0.
        due_times = [self.scheduler._due[("anim", idx)] for idx in range(3)]
        self.assertEqual(due_times, sorted(due_times))

    def test_actions_fire_only_when_time_reached(self) -> None:
        called: list[str] = []

        actions = [
            WidgetAction(time=0.10, func=lambda: called.append("a")),
            WidgetAction(time=0.30, func=lambda: called.append("b")),
            WidgetAction(time=0.30, func=lambda: called.append("c")),
        ]
        animator = WidgetAnimator(self.scheduler, actions=actions, key_prefix="anim")

        with patch("time.perf_counter", lambda: 0.0):
            animator.start()

        # Before 0.10: nothing.
        with patch("time.perf_counter", lambda: 0.09):
            self._tick_scheduler()
        self.assertEqual(called, [])

        # At 0.11: only "a"
        with patch("time.perf_counter", lambda: 0.11):
            self._tick_scheduler()
        self.assertEqual(called, ["a"])

        # At 0.31: both "b" and "c"
        with patch("time.perf_counter", lambda: 0.31):
            self._tick_scheduler()
        self.assertEqual(called, ["a", "b", "c"])

    def test_stop_cancels_all_keys_and_prevents_future_firing(self) -> None:
        called: list[str] = []
        actions = [
            WidgetAction(time=0.10, func=lambda: called.append("a")),
            WidgetAction(time=0.20, func=lambda: called.append("b")),
        ]
        animator = WidgetAnimator(self.scheduler, actions=actions, key_prefix="anim")

        with patch("time.perf_counter", lambda: 0.0):
            animator.start()

        animator.stop()

        self.assertFalse(animator._running)
        self.assertNotIn(("anim", 0), self.scheduler._callbacks)
        self.assertNotIn(("anim", 1), self.scheduler._callbacks)
        self.assertEqual(self.scheduler._due, {})

        # Even if time moves forward, nothing should fire.
        with patch("time.perf_counter", lambda: 1.0):
            self._tick_scheduler()

        self.assertEqual(called, [])

    def test_multiple_stop_does_nothing(self) -> None:
        actions = [WidgetAction(time=0.1, func=lambda: None)]
        animator = WidgetAnimator(self.scheduler, actions=actions, key_prefix="anim")

        with patch("time.perf_counter", lambda: 0.0):
            animator.start()

        animator.stop()
        animator.stop()  # should not raise
        self.assertFalse(animator._running)

    def tearDown(self) -> None:
        self.scheduler.shutdown()


class TestMakeSliderStepActions(unittest.TestCase):
    def test_num_steps_less_than_2_returns_empty(self) -> None:
        widget = FakeWidget()
        actions = make_slider_step_actions_for_widget(
            widget=widget,
            start_value=0.0,
            stop_value=1.0,
            num_steps=1,
            step_dt=0.5,
        )
        self.assertEqual(actions, [])
        self.assertEqual(widget.values, [])

    def test_generates_correct_number_of_actions(self) -> None:
        widget = FakeWidget()
        actions = make_slider_step_actions_for_widget(
            widget=widget,
            start_value=0.0,
            stop_value=1.0,
            num_steps=5,
            step_dt=0.5,
        )
        self.assertEqual(len(actions), 5)
        self.assertEqual([a.time for a in actions], [0.0, 0.5, 1.0, 1.5, 2.0])

    def test_action_values_are_linearly_spaced_and_call_set_state(self) -> None:
        widget = FakeWidget()
        actions = make_slider_step_actions_for_widget(
            widget=widget,
            start_value=10.0,
            stop_value=20.0,
            num_steps=3,
            step_dt=0.5,
        )

        # Execute the funcs directly, without animator.
        for a in actions:
            a.func()

        self.assertEqual(widget.values, [10.0, 15.0, 20.0])

    def test_integration_animator_executes_slider_actions_in_time_order(self) -> None:
        iren = vtkRenderWindowInteractor()
        scheduler = VtkDebounceScheduler(iren, period_ms=33)

        widget = FakeWidget()
        actions = make_slider_step_actions_for_widget(
            widget=widget,
            start_value=0.0,
            stop_value=1.0,
            num_steps=3,
            step_dt=0.5,
        )
        animator = WidgetAnimator(scheduler, actions=actions, key_prefix="anim")

        with patch("time.perf_counter", lambda: 0.0):
            animator.start()

        # t=0.0 -> first should fire
        with patch("time.perf_counter", lambda: 0.0):
            scheduler._on_timer(None, "TimerEvent")
        self.assertEqual(widget.values, [0.0])

        # t=0.49 -> none
        with patch("time.perf_counter", lambda: 0.49):
            scheduler._on_timer(None, "TimerEvent")
        self.assertEqual(widget.values, [0.0])

        # t=0.51 -> second
        with patch("time.perf_counter", lambda: 0.51):
            scheduler._on_timer(None, "TimerEvent")
        self.assertEqual(widget.values, [0.0, 0.5])

        # t=1.01 -> third
        with patch("time.perf_counter", lambda: 1.01):
            scheduler._on_timer(None, "TimerEvent")
        self.assertEqual(widget.values, [0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
