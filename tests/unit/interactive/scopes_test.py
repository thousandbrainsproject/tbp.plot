# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import unittest
from dataclasses import dataclass
from typing import Any

from tbp.interactive.scopes import ScopeViewer


class FakePlotter:
    """Minimal fake Plotter for ScopeViewer tests."""

    def __init__(self) -> None:
        self.callbacks: list[tuple[str, Any]] = []
        self.render_calls = 0

    def add_callback(self, event_name: str, callback: Any) -> None:
        self.callbacks.append((event_name, callback))

    def render(self) -> None:
        self.render_calls += 1


@dataclass
class FakeWidget:
    """Minimal fake Widget for ScopeViewer tests."""

    scopes: list[int]
    is_visible: bool = True

    def on(self) -> None:
        self.is_visible = True

    def off(self) -> None:
        self.is_visible = False


class FakeEvent:
    """Fake event with a `keypress` attribute."""

    def __init__(self, keypress: str | None) -> None:
        self.keypress = keypress


class ScopeViewerSetup(unittest.TestCase):
    def setUp(self) -> None:
        self.plotter = FakePlotter()
        self.widgets = {
            "a": FakeWidget(scopes=[1], is_visible=True),
            "b": FakeWidget(scopes=[1, 2], is_visible=True),
            "c": FakeWidget(scopes=[2], is_visible=True),
            "d": FakeWidget(scopes=[3], is_visible=True),
        }
        self.viewer = ScopeViewer(plotter=self.plotter, widgets=self.widgets)


class TestScopeViewerInitialization(ScopeViewerSetup):
    def test_builds_scope_map(self) -> None:
        self.assertEqual(
            self.viewer.scope_to_widgets,
            {
                1: {"a", "b"},
                2: {"b", "c"},
                3: {"d"},
            },
        )

    def test_all_scopes_active_by_default(self) -> None:
        """Tests that ScopeViewer starts with all scopes active."""
        self.assertEqual(self.viewer.active_scopes, {1, 2, 3})

    def test_registers_keypress_callback(self) -> None:
        """Tests that the viewer registers a keypress callback on the plotter."""
        self.assertIn(
            ("KeyPress", self.viewer._on_keypress),
            self.plotter.callbacks,
        )


class TestScopeViewerToggleScope(ScopeViewerSetup):
    def test_toggle_unknown_scope_is_noop(self) -> None:
        """Test that toggling unknown scopes does nothing."""
        before_scopes = set(self.viewer.active_scopes)
        before_vis = {k: w.is_visible for k, w in self.widgets.items()}

        self.viewer.toggle_scope(9)

        self.assertEqual(self.viewer.active_scopes, before_scopes)
        self.assertEqual(
            {k: w.is_visible for k, w in self.widgets.items()},
            before_vis,
        )

    def test_toggle_scope_off_hides_only_exclusive_widgets(self) -> None:
        """Tests for scopes that share widgets.

        In this test, widget "b" is in scopes "1" and "2". The user turns scope
        "1" off, but leaves scope "2" on. The correct behavior is for widget "b"
        to stay on and for the exclusive widget "a" to turn off.
        """
        self.viewer.toggle_scope(1)

        self.assertEqual(self.viewer.active_scopes, {2, 3})
        self.assertFalse(self.widgets["a"].is_visible)
        self.assertTrue(self.widgets["b"].is_visible)

    def test_toggle_scope_off_keeps_widgets_visible_if_other_scope_active(self) -> None:
        """Tests for toggle scope off does not affect out-of-scope widgets.

        Scope "3" only has widget "d" in it. So, no other widget should be affected.
        """
        self.viewer.toggle_scope(3)

        self.assertFalse(self.widgets["d"].is_visible)
        self.assertTrue(self.widgets["a"].is_visible)
        self.assertTrue(self.widgets["b"].is_visible)
        self.assertTrue(self.widgets["c"].is_visible)

    def test_toggle_scope_on_restores_visibility(self) -> None:
        """Tests the toggle functionality; on and off."""
        self.viewer.toggle_scope(3)
        self.assertFalse(self.widgets["d"].is_visible)

        self.viewer.toggle_scope(3)
        self.assertTrue(self.widgets["d"].is_visible)


class TestScopeViewerToggleAll(ScopeViewerSetup):
    def test_toggle_all_hides_all_when_all_visible(self) -> None:
        self.viewer.toggle_scope(0)

        self.assertEqual(self.viewer.active_scopes, set())
        for widget in self.widgets.values():
            self.assertFalse(widget.is_visible)

    def test_toggle_all_shows_all_when_any_hidden(self) -> None:
        self.widgets["c"].off()
        self.assertFalse(self.widgets["c"].is_visible)

        self.viewer.toggle_scope(0)

        self.assertEqual(self.viewer.active_scopes, {1, 2, 3})
        for widget in self.widgets.values():
            self.assertTrue(widget.is_visible)


class TestScopeViewerKeypress(ScopeViewerSetup):
    def test_non_digit_keypress_is_ignored(self) -> None:
        self.viewer._on_keypress(FakeEvent("x"))
        self.viewer._on_keypress(FakeEvent(None))
        self.viewer._on_keypress(FakeEvent(" "))

        self.assertEqual(self.plotter.render_calls, 0)
        self.assertEqual(self.viewer.active_scopes, {1, 2, 3})

    def test_digit_keypress_toggles_scope_and_renders(self) -> None:
        self.viewer._on_keypress(FakeEvent("1"))

        self.assertEqual(self.plotter.render_calls, 1)
        self.assertEqual(self.viewer.active_scopes, {2, 3})
        self.assertFalse(self.widgets["a"].is_visible)

    def test_zero_keypress_triggers_toggle_all_and_renders(self) -> None:
        self.viewer._on_keypress(FakeEvent("0"))

        self.assertEqual(self.plotter.render_calls, 1)
        self.assertEqual(self.viewer.active_scopes, set())
        for widget in self.widgets.values():
            self.assertFalse(widget.is_visible)


if __name__ == "__main__":
    unittest.main()
