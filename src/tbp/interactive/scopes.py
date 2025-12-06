# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


from __future__ import annotations

from typing import Any

from vedo import Plotter

from tbp.interactive.widgets import Widget


class ScopeViewer:
    """Controls widget visibility using numeric keypress scopes.

    Behavior summary:
    - Scope 0:
        * If at least one widget is hidden -> show ALL widgets.
        * Else (all visible) -> hide ALL widgets.
    - Scope k (1..9):
        * Toggle that scope on/off.
        * A widget is visible if ANY active scope includes it.
        * If no active scopes -> all widgets off.

    The widgets themselves decide how to hide/show internally using
    their .on() / .off() visibility handlers.
    """

    def __init__(self, plotter: Plotter, widgets: dict[str, Widget]):
        self.plotter = plotter
        self.widgets = widgets

        self.scope_to_widgets: dict[int, set[str]] = {}

        # Build scope map from each widget's `scopes` list.
        for name, widget in widgets.items():
            for s in widget.scopes:
                if s not in self.scope_to_widgets:
                    self.scope_to_widgets[s] = set()
                self.scope_to_widgets[s].add(name)

        self.active_scopes: set[int] = set(self.scope_to_widgets.keys())
        self.plotter.add_callback("KeyPress", self._on_keypress)

    def _on_keypress(self, event: Any) -> None:
        key = getattr(event, "keypress", None)
        if not key or not key.isdigit():
            return

        self.toggle_scope(int(key))
        self.plotter.render()

    def toggle_scope(self, scope_id: int) -> None:
        """Toggles a specific scope by its id."""
        if scope_id == 0:
            return self._toggle_all()

        if scope_id not in self.scope_to_widgets:
            return None

        if scope_id in self.active_scopes:
            self.active_scopes.remove(scope_id)
        else:
            self.active_scopes.add(scope_id)

        self._apply_scope_visibility()

    def _toggle_all(self) -> None:
        """Toggles all widgets on/off."""
        any_hidden = any(not w.is_visible for w in self.widgets.values())

        if any_hidden:
            for w in self.widgets.values():
                w.on()
            self.active_scopes = set(self.scope_to_widgets.keys())
        else:
            for w in self.widgets.values():
                w.off()
            self.active_scopes.clear()

    def _apply_scope_visibility(self) -> None:
        # If nothing is active, hide everything
        if not self.active_scopes:
            for w in self.widgets.values():
                w.off()
            return

        for widget in self.widgets.values():
            # Does this widget belong to ANY active scope?
            belongs = any(s in self.active_scopes for s in widget.scopes)

            if belongs and not widget.is_visible:
                widget.on()
            elif not belongs and widget.is_visible:
                widget.off()
