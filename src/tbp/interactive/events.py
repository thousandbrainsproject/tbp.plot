# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import dataclass


@dataclass
class EventSpec:
    """Specification for an Event to be defined as a WidgetUpdater callback trigger.

    Attributes:
        trigger: Event trigger name (e.g., KeyPressed)
        name: Event name field in Vedo `event.name` (e.g., keypress).
        required: Whether this event is required for the callback trigger. If
            True, the updater will not call the callback until a message for this
            topic arrives.
    """

    trigger: str
    name: str
    required: bool = True
