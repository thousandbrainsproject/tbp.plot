# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tbp.interactive.topics import TopicMessage
from tbp.interactive.widget_updaters import WidgetUpdaterProto

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@runtime_checkable
class WidgetOpsProto(Protocol):
    pass


@runtime_checkable
class SupportsAdd[WidgetT](Protocol):
    def add(self, callback: Callable[[WidgetT, str], None]) -> WidgetT: ...


@runtime_checkable
class SupportsRemove[WidgetT](Protocol):
    def remove(self, widget: WidgetT) -> None: ...


@runtime_checkable
class SupportsExtractState[WidgetT, StateT](Protocol):
    def extract_state(self, widget: WidgetT | None) -> StateT | None: ...


@runtime_checkable
class SupportsSetState[WidgetT, StateT](Protocol):
    def set_state(self, widget: WidgetT | None, value: StateT | None) -> None: ...


@runtime_checkable
class HasStateToMessages[StateT](Protocol):
    def state_to_messages(self, state: StateT | None) -> Iterable[TopicMessage]: ...


@runtime_checkable
class HasUpdaters[WidgetT](Protocol):
    updaters: Iterable[WidgetUpdaterProto[WidgetT]]
