# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import contextlib
import logging
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from pubsub.core import Publisher
from scipy.spatial.transform import Rotation
from vedo import Button, Image, Line, Mesh, Plotter, Slider2D, Sphere, Text2D

from tbp.interactive.colors import Palette
from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
    YCBMeshLoader,
)
from tbp.interactive.events import EventSpec
from tbp.interactive.topics import TopicMessage, TopicSpec
from tbp.interactive.utils import rotate_about_pivot
from tbp.interactive.widget_updaters import WidgetUpdater
from tbp.interactive.widgets import (
    VtkDebounceScheduler,
    Widget,
    extract_slider_state,
    set_slider_state,
)
from tbp.plot.registry import attach_args, register

logger = logging.getLogger(__name__)


FONT = "Arial"
FONT_SIZE = 30

DEFAULTS = {
    "x_percent_threshold": 20,
    "object_evidence_threshold": 1.0,
    "path_similarity_threshold": 0.1,
    "pose_similarity_threshold": 0.1,
    "symmetry_overlap_threshold": 0.9,
    "pm_std": 0.1,
    "pm_max": 0.0,
    "norm": "Raw Evidence",
}


COLOR_PALETTE = {
    "Primary": Palette.as_hex("numenta_blue"),
    "Secondary": Palette.as_hex("purple"),
    "Accent": Palette.as_hex("charcoal"),
    "Accent2": Palette.as_hex("link_water"),
    "Accent3": Palette.as_hex("rich_black"),
}


HUE_PALETTE = {
    True: COLOR_PALETTE["Primary"],
    False: COLOR_PALETTE["Secondary"],
    "x_percent": COLOR_PALETTE["Accent2"],
}


# This is the default mesh location for objects in the Habitat simulator
_DEFAULT_PIVOT = np.array([0.0, 1.5, 0.0], dtype=float)


class ClickWidgetOps:
    """Captures 3D click positions and publish them on the bus.

    This class registers plotter-level mouse callbacks. A left-click picks a 3D
    point (if available) and triggers the widget callback; a right-click
    resets the camera pose. There is no visual widget created by this class.

    Attributes:
        plotter: The `vedo.Plotter` where callbacks are installed.
        cam_dict: Dictionary for camera default specs.
        click_location: Last picked 3D location, if any.
    """

    def __init__(self, plotter: Plotter, cam_dict: dict[str, Any]) -> None:
        self.plotter = plotter
        self.cam_dict = cam_dict

    def add(self, callback: Callable) -> None:
        """Register mouse callbacks on the plotter.

        Note that this callback makes use of the `VtkDebounceScheduler`
        to publish messages. Storing the callback and triggering it, will
        simulate a UI change on e.g., a button or a slider, which schedules
        a publish. We use this callback because this event is not triggered
        by receiving topics from a `WidgetUpdater`.


        Args:
            callback: Function invoked like `(widget, event)` when a left-click
                captures a 3D location.
        """
        self.plotter.at(0).add_callback("RightButtonPress", self.on_right_click)

    def align_camera(self, cams: list[Any], cam_clicked: Any) -> None:
        """Align the camera objects."""
        for cam in cams:
            cam.SetPosition(cam_clicked.GetPosition())
            cam.SetFocalPoint(cam_clicked.GetFocalPoint())
            cam.SetViewUp(cam_clicked.GetViewUp())
            cam.SetClippingRange(cam_clicked.GetClippingRange())
            cam.SetParallelScale(cam_clicked.GetParallelScale())

    def get_cam(self, i):
        return self.plotter.renderers[i].GetActiveCamera()

    def on_right_click(self, event):
        """Handle right mouse press (reset camera pose and render).

        Notes:
            Bound to the "RightButtonPress" event in `self.add()`.
        """
        if event.at == 0:
            renderer = self.plotter.at(0).renderer
            if renderer is not None:
                cam = renderer.GetActiveCamera()
                cam.SetPosition(self.cam_dict["pos"])
                cam.SetFocalPoint(self.cam_dict["focal_point"])
                cam.SetViewUp((0, 1, 0))
                cam.SetClippingRange((0.01, 1000.01))
                self.plotter.at(0).render()
        elif 1 <= event.at <= 4:
            cam_clicked = self.get_cam(event.at)
            cam_copy = [self.get_cam(i) for i in range(1, 5) if i != event.at]
            self.align_camera(cam_copy, cam_clicked)


class EpisodeSliderWidgetOps:
    """WidgetOps implementation for an Episode slider.

    This class sets the slider's range based on the number of
    available episodes and publishes changes as `TopicMessage` items
    under the "episode_number" topic.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            json log file.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
        _locators: Data accessors keyed by name. These instruct the DataParser
            on how to retrieve the required information.
    """

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser

        self._add_kwargs = {
            "xmin": 0,
            "xmax": 10,
            "value": 0,
            "pos": [(0.1, 0.2), (0.9, 0.2)],
            "title": "Episode",
        }

        self._locators = {
            "episode": DataLocator(
                path=[
                    DataLocatorStep.key(name="episode"),
                ],
            )
        }

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        kwargs = deepcopy(self._add_kwargs)
        locator = self._locators["episode"]
        kwargs.update({"xmax": len(self.data_parser.query(locator)) - 1})
        widget = self.plotter.at(0).add_slider(callback, **kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> int:
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="episode_number", value=state)]
        return messages


class StepSliderWidgetOps:
    """WidgetOps implementation for a Step slider.

    This class listens for the current episode selection and adjusts the step
    slider range to match the number of steps in that episode. It publishes
    changes as `TopicMessage` items under the "step_number" topic.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            JSON log file.
        updaters: A list with a single `WidgetUpdater` that reacts to the
            `"episode_number"` topic and calls `update_slider_range`.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
        _locators: Data accessors keyed by name that instruct the `DataParser`
            how to retrieve the required information.
    """

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("episode_number", required=True)],
                callback=self.update_slider_range,
            )
        ]

        self._add_kwargs = {
            "xmin": 0,
            "xmax": 10,
            "value": 0,
            "pos": [(0.1, 0.1), (0.9, 0.1)],
            "title": "Step",
        }
        self._locators = self.create_locators()

    def create_locators(self) -> dict[str, DataLocator]:
        locators = {}
        locators["step"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
            ]
        )
        return locators

    def add(self, callback: Callable) -> Slider2D:
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> int:
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="step_number", value=state)]
        return messages

    def update_slider_range(
        self, widget: Slider2D, msgs: list[TopicMessage]
    ) -> tuple[Slider2D, bool]:
        """Adjust slider range based on the selected episode and reset to 0.

        Looks up the `"episode_number"` message, queries the number of steps for
        that episode, sets the slider range to `[0, num_steps - 1]`, resets the
        value to 0, and re-renders.

        Args:
            widget: The slider widget to update.
            msgs: Messages from the `WidgetUpdater`.

        Returns:
            A tuple `(widget, True)` indicating the updated widget and whether
            a publish should occur.
        """
        msgs_dict = {msg.name: msg.value for msg in msgs}

        # set widget range to the correct step number
        widget.range = [
            0,
            len(
                self.data_parser.query(
                    self._locators["step"], episode=str(msgs_dict["episode_number"])
                )
            )
            - 1,
        ]

        # set slider value back to zero
        self.set_state(widget, 0)

        return widget, True


class ResetButtonWidgetOps:
    """WidgetOps implementation for a reset button.

    Resets the terminal conditions parameters (i.e., sliders).

    Attributes:
        plotter: A `vedo.Plotter` object used to add/remove actors and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_button`.
    """

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = {
            "pos": (0.5, 0.99),
            "states": ["Reset"],
            "c": ["w"],
            "bc": [COLOR_PALETTE["Primary"]],
            "size": FONT_SIZE,
            "font": FONT,
            "bold": False,
        }

    def add(self, callback: Callable) -> Button:
        widget = self.plotter.at(0).add_button(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="reset_button", value=True)]
        return messages


class XPercentSliderWidgetOps:
    """WidgetOps implementation for the x-percent threshold slider.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = {
            "xmin": 0,
            "xmax": 50,
            "value": DEFAULTS["x_percent_threshold"],
            "pos": [(0.2, 0.7), (0.2, 1.0)],
            "title": "x percent",
            "title_size": 0.75,
        }

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("reset_button", required=True),
                ],
                callback=self.reset_value,
            )
        ]

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> int:
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        set_slider_state(widget, value)

    def reset_value(
        self, widget: Slider2D | None, msgs: list[TopicMessage]
    ) -> tuple[Slider2D | None, bool]:
        self.set_state(widget, value=DEFAULTS["x_percent_threshold"])
        return widget, True

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="x_percent_threshold", value=state)]
        return messages


class ObjectEvidenceSliderWidgetOps:
    """WidgetOps implementation for the object evidence slider.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = {
            "xmin": 0,
            "xmax": 10,
            "value": DEFAULTS["object_evidence_threshold"],
            "pos": [(0.3, 0.7), (0.3, 1.0)],
            "title": "object evidence",
            "title_size": 0.75,
        }

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("reset_button", required=True),
                ],
                callback=self.reset_value,
            )
        ]

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> float:
        return widget.GetRepresentation().GetValue()

    def set_state(self, widget: Slider2D, value: float) -> None:
        set_slider_state(widget, value)

    def reset_value(
        self, widget: Slider2D | None, msgs: list[TopicMessage]
    ) -> tuple[Slider2D | None, bool]:
        self.set_state(widget, value=DEFAULTS["object_evidence_threshold"])
        return widget, True

    def state_to_messages(self, state: float) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="object_evidence_threshold", value=state)]
        return messages


class PathSimilaritySliderWidgetOps:
    """WidgetOps implementation for the path similarity threshold slider.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = {
            "xmin": 0.0,
            "xmax": 4.0,
            "value": DEFAULTS["path_similarity_threshold"],
            "pos": [(0.4, 0.7), (0.4, 1.0)],
            "title": "path similarity",
            "title_size": 0.75,
        }

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("reset_button", required=True),
                ],
                callback=self.reset_value,
            )
        ]

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> float:
        return widget.GetRepresentation().GetValue()

    def set_state(self, widget: Slider2D, value: float) -> None:
        set_slider_state(widget, value)

    def reset_value(
        self, widget: Slider2D | None, msgs: list[TopicMessage]
    ) -> tuple[Slider2D | None, bool]:
        self.set_state(widget, value=DEFAULTS["path_similarity_threshold"])
        return widget, True

    def state_to_messages(self, state: float) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="path_similarity_threshold", value=state)]
        return messages


class PoseSimilaritySliderWidgetOps:
    """WidgetOps implementation for the pose similarity threshold slider.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = {
            "xmin": 0.0,
            "xmax": 4.0,
            "value": DEFAULTS["pose_similarity_threshold"],
            "pos": [(0.5, 0.7), (0.5, 1.0)],
            "title": "pose similarity",
            "title_size": 0.75,
        }

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("reset_button", required=True),
                ],
                callback=self.reset_value,
            )
        ]

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> float:
        return widget.GetRepresentation().GetValue()

    def set_state(self, widget: Slider2D, value: float) -> None:
        set_slider_state(widget, value)

    def reset_value(
        self, widget: Slider2D | None, msgs: list[TopicMessage]
    ) -> tuple[Slider2D | None, bool]:
        self.set_state(widget, value=DEFAULTS["pose_similarity_threshold"])
        return widget, True

    def state_to_messages(self, state: float) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="pose_similarity_threshold", value=state)]
        return messages


class SymmetryOverlapSliderWidgetOps:
    """WidgetOps implementation for the symmetry overlap threshold slider.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = {
            "xmin": 0.0,
            "xmax": 1.0,
            "value": DEFAULTS["symmetry_overlap_threshold"],
            "pos": [(0.6, 0.7), (0.6, 1.0)],
            "title": "symmetry overlap",
            "title_size": 0.75,
        }

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("reset_button", required=True),
                ],
                callback=self.reset_value,
            )
        ]

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> float:
        return widget.GetRepresentation().GetValue()

    def set_state(self, widget: Slider2D, value: float) -> None:
        set_slider_state(widget, value)

    def reset_value(
        self, widget: Slider2D | None, msgs: list[TopicMessage]
    ) -> tuple[Slider2D | None, bool]:
        self.set_state(widget, value=DEFAULTS["symmetry_overlap_threshold"])
        return widget, True

    def state_to_messages(self, state: float) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="symmetry_overlap_threshold", value=state)]
        return messages


class StdEvidenceSliderWidgetOps:
    """WidgetOps implementation for the std dev threshold of evidence slider.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = {
            "xmin": 0.0,
            "xmax": 1.0,
            "value": DEFAULTS["pm_std"],
            "pos": [(0.7, 0.7), (0.7, 1.0)],
            "title": "Evidence STD",
            "title_size": 0.75,
        }

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("reset_button", required=True),
                ],
                callback=self.reset_value,
            )
        ]

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> float:
        return widget.GetRepresentation().GetValue()

    def set_state(self, widget: Slider2D, value: float) -> None:
        set_slider_state(widget, value)

    def reset_value(
        self, widget: Slider2D | None, msgs: list[TopicMessage]
    ) -> tuple[Slider2D | None, bool]:
        self.set_state(widget, value=DEFAULTS["pm_std"])
        return widget, True

    def state_to_messages(self, state: float) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="evidence_std", value=state)]
        return messages


class MaxEvidenceSliderWidgetOps:
    """WidgetOps implementation for the max evidence threshold slider.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = {
            "xmin": -1.0,
            "xmax": 1.0,
            "value": DEFAULTS["pm_max"],
            "pos": [(0.8, 0.7), (0.8, 1.0)],
            "title": "Evidence MAX",
            "title_size": 0.75,
        }

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("reset_button", required=True),
                ],
                callback=self.reset_value,
            )
        ]

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> float:
        return widget.GetRepresentation().GetValue()

    def set_state(self, widget: Slider2D, value: float) -> None:
        set_slider_state(widget, value)

    def reset_value(
        self, widget: Slider2D | None, msgs: list[TopicMessage]
    ) -> tuple[Slider2D | None, bool]:
        self.set_state(widget, value=DEFAULTS["pm_max"])
        return widget, True

    def state_to_messages(self, state: float) -> Iterable[TopicMessage]:
        messages = [TopicMessage(name="evidence_max", value=state)]
        return messages


class GtMeshWidgetOps:
    """WidgetOps implementation for rendering the ground-truth target mesh.

    This widget is display-only. It listens for `episode_number` updates,
    loads the target object's YCB mesh, applies the episode-specific rotations,
    scales and positions it, and adds it to the plotter. It does not publish
    any messages.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        updaters: A single `WidgetUpdater` that reacts to `"episode_number"`.
        _locators: Data accessors keyed by name for the parser.
    """

    def __init__(
        self, plotter: Plotter, data_parser: DataParser, ycb_loader: YCBMeshLoader
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.ycb_loader = ycb_loader
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("episode_number", required=True)],
                callback=self.update_mesh,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                ],
                callback=self.update_agent,
            ),
            WidgetUpdater(
                topics=[
                    EventSpec("KeyPressed", "KeyPressEvent", required=True),
                ],
                callback=self.update_transparency,
            ),
        ]
        self._locators = self.create_locators()

        self.gaze_line: Line | None = None
        self.agent_sphere: Sphere | None = None
        self.mesh_transparency: float = 0.0

        self.plotter.at(1).add(Text2D(txt="Ground Truth", pos="top-center"))

    def create_locators(self) -> dict[str, DataLocator]:
        locators = {}
        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="target"),
            ]
        )

        locators["steps_mask"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="system", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="lm_processed_steps"),
            ]
        )

        locators["agent_location"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="system", value="motor_system"),
                DataLocatorStep.key(name="telemetry", value="action_sequence"),
                DataLocatorStep.index(name="sm_step"),
                DataLocatorStep.index(name="telemetry_type", value=1),
                DataLocatorStep.key(name="agent", value="agent_id_0"),
                DataLocatorStep.key(name="pose", value="position"),
            ]
        )

        locators["patch_location"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="system", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="locations"),
                DataLocatorStep.key(name="sm", value="patch"),
                DataLocatorStep.index(name="step"),
            ]
        )

        return locators

    def update_mesh(self, widget: Mesh, msgs: list[TopicMessage]) -> tuple[Mesh, bool]:
        if widget is not None:
            self.plotter.at(1).remove(widget)

        msgs_dict = {msg.name: msg.value for msg in msgs}

        locator = self._locators["target"]
        target = self.data_parser.extract(
            locator, episode=str(msgs_dict["episode_number"])
        )
        target_id = target["primary_target_object"]
        target_rot = target["primary_target_rotation_quat"]
        target_pos = target["primary_target_position"]

        try:
            widget = self.ycb_loader.create_mesh(target_id).clone(deep=True)
        except FileNotFoundError:
            return widget, False

        rot = Rotation.from_quat(np.array(target_rot), scalar_first=True)
        rot_euler = rot.as_euler("xyz", degrees=True)
        widget.rotate_x(rot_euler[0])
        widget.rotate_y(rot_euler[1])
        widget.rotate_z(rot_euler[2])
        widget.shift(*target_pos)
        widget.alpha(1.0 - self.mesh_transparency)

        self.plotter.at(1).add(widget)

        return widget, False

    def update_agent(self, widget: None, msgs: list[TopicMessage]) -> tuple[None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]

        steps_mask = self.data_parser.extract(
            self._locators["steps_mask"], episode=str(episode_number)
        )
        mapping = np.flatnonzero(steps_mask)

        agent_pos = self.data_parser.extract(
            self._locators["agent_location"],
            episode=str(episode_number),
            sm_step=max(0, int(mapping[step_number]) - 1),
        )

        patch_pos = self.data_parser.extract(
            self._locators["patch_location"],
            episode=str(episode_number),
            step=step_number,
        )

        if self.agent_sphere is None:
            self.agent_sphere = Sphere(
                pos=agent_pos, r=0.004, c=COLOR_PALETTE["Secondary"]
            )
            self.plotter.at(1).add(self.agent_sphere)
        self.agent_sphere.pos(agent_pos)

        if self.gaze_line is None:
            self.gaze_line = Line(
                agent_pos, patch_pos, c=COLOR_PALETTE["Accent3"], lw=4
            )
            self.plotter.at(1).add(self.gaze_line)
        self.gaze_line.points = [agent_pos, patch_pos]

        return widget, False

    def update_transparency(
        self, widget: None, msgs: list[TopicMessage]
    ) -> tuple[None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}

        key_event = msgs_dict.get("KeyPressEvent", None)
        if key_event is not None and getattr(key_event, "at", None) == 1:
            key = getattr(key_event, "keypress", None)

            if key == "Left":
                self.mesh_transparency -= 0.5
            elif key == "Right":
                self.mesh_transparency += 0.5

        self.mesh_transparency = float(np.clip(self.mesh_transparency, 0.0, 1.0))
        if widget is not None:
            widget.alpha(1.0 - self.mesh_transparency)

        self.updaters[2].expire_topic("KeyPressEvent")

        return widget, False


class MlhMeshWidgetOps:
    """WidgetOps implementation for rendering the MLH mesh.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        updaters: A single `WidgetUpdater` that reacts to `"episode_number"`.
        _locators: Data accessors keyed by name for the parser.
    """

    def __init__(
        self, plotter: Plotter, data_parser: DataParser, ycb_loader: YCBMeshLoader
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.ycb_loader = ycb_loader
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                ],
                callback=self.update_mesh,
            ),
            WidgetUpdater(
                topics=[
                    EventSpec("KeyPressed", "KeyPressEvent", required=True),
                ],
                callback=self.update_transparency,
            ),
        ]
        self._locators = self.create_locators()

        self.default_object_position = (0, 1.5, 0)
        self.sensor_sphere: Sphere | None = None
        self.mesh_transparency: float = 0.0

        self.plotter.at(2).add(Text2D(txt="MLH", pos="top-center"))

    def create_locators(self) -> dict[str, DataLocator]:
        locators = {}

        locators["mlh"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="current_mlh"),
                DataLocatorStep.index(name="step"),
            ]
        )

        locators["patch_location"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="system", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="locations"),
                DataLocatorStep.key(name="sm", value="patch"),
                DataLocatorStep.index(name="step"),
            ]
        )

        return locators

    def update_mesh(self, widget: Mesh, msgs: list[TopicMessage]) -> tuple[Mesh, bool]:
        if widget is not None:
            self.plotter.at(2).remove(widget)

        if self.sensor_sphere is not None:
            self.plotter.at(2).remove(self.sensor_sphere)
            self.sensor_sphere = None

        msgs_dict = {msg.name: msg.value for msg in msgs}
        mlh = self.data_parser.extract(
            self._locators["mlh"],
            episode=str(msgs_dict["episode_number"]),
            step=msgs_dict["step_number"],
        )
        patch_location = self.data_parser.extract(
            self._locators["patch_location"],
            episode=str(msgs_dict["episode_number"]),
            step=msgs_dict["step_number"],
        )

        mlh_id = mlh["graph_id"]
        mlh_rot = mlh["rotation"]
        mlh_pos = mlh["location"]

        # Add object_mesh
        try:
            widget = self.ycb_loader.create_mesh(mlh_id).clone(deep=True)
        except FileNotFoundError:
            return widget, False

        rot = Rotation.from_euler("xyz", mlh_rot, degrees=True).inv()
        rot_euler = rot.as_euler("xyz", degrees=True)
        widget.rotate_x(rot_euler[0])
        widget.rotate_y(rot_euler[1])
        widget.rotate_z(rot_euler[2])

        # Translate hypothesis location to sensor location
        hyp_loc_world_coord = rotate_about_pivot(rot, mlh_pos, _DEFAULT_PIVOT)
        widget.shift(
            self.default_object_position + (patch_location - hyp_loc_world_coord)
        )

        widget.alpha(1.0 - self.mesh_transparency)
        self.plotter.at(2).add(widget)

        self.sensor_sphere = Sphere(pos=patch_location, r=0.003).c(
            COLOR_PALETTE["Primary"]
        )
        self.plotter.at(2).add(self.sensor_sphere)

        return widget, False

    def update_transparency(
        self, widget: None, msgs: list[TopicMessage]
    ) -> tuple[None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}

        key_event = msgs_dict.get("KeyPressEvent", None)
        if key_event is not None and getattr(key_event, "at", None) == 2:
            key = getattr(key_event, "keypress", None)

            if key == "Left":
                self.mesh_transparency -= 0.5
            elif key == "Right":
                self.mesh_transparency += 0.5

        self.mesh_transparency = float(np.clip(self.mesh_transparency, 0.0, 1.0))
        if widget is not None:
            widget.alpha(1.0 - self.mesh_transparency)

        self.updaters[1].expire_topic("KeyPressEvent")

        return widget, False


class UniqueMeshWidgetOps:
    """WidgetOps implementation for rendering the Unique mesh.

    This widget is display-only. It listens for `"episode_number"` updates,
    loads the target object's YCB mesh, applies the episode-specific rotations,
    scales and positions it, and adds it to the plotter. It does not publish
    any messages.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        updaters: A single `WidgetUpdater` that reacts to `"episode_number"`.
        _locators: Data accessors keyed by name for the parser.
    """

    def __init__(
        self, plotter: Plotter, data_parser: DataParser, ycb_loader: YCBMeshLoader
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.ycb_loader = ycb_loader
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("unique_poses", required=True),
                ],
                callback=self.update_plot,
            ),
        ]

        self.default_object_position = (0, 1.5, 0)
        self.plotter.at(3).add(Text2D(txt="Unique Poses", pos="top-center"))
        self.widgets: list[Mesh] = []

    def update_plot(self, widget: Mesh, msgs: list[TopicMessage]) -> tuple[Mesh, bool]:
        """Update the target mesh when the episode changes.

        Removes any existing mesh, loads the episode's primary target object,
        applies its Euler rotations, scales and positions it, then adds it to
        the plotter.

        Args:
            widget: The currently displayed mesh, if any.
            msgs: Messages received from the `WidgetUpdater`.

        Returns:
            A tuple `(mesh, False)`. The second value is `False` to indicate
            that no publish should occur.
        """
        if len(self.widgets):
            for w in self.widgets:
                self.plotter.at(3).remove(w)
            self.widgets = []

        msgs_dict = {msg.name: msg.value for msg in msgs}
        data_df = msgs_dict["unique_poses"]

        if isinstance(data_df, pd.DataFrame):
            data_df = (
                data_df.assign(
                    evidence=pd.to_numeric(data_df["evidence"], errors="coerce")
                )
                .dropna(subset=["evidence"])
                .nlargest(5, "evidence")
            )

            for gid, rx, ry, rz in data_df[
                ["graph_id", "rot_x", "rot_y", "rot_z"]
            ].itertuples(index=False, name=None):
                widget = self.ycb_loader.create_mesh(gid).clone(deep=True)
                widget.rotate_x(rx)
                widget.rotate_y(ry)
                widget.rotate_z(rz)
                widget.shift(*self.default_object_position)
                self.plotter.at(3).add(widget)
                self.widgets.append(widget)

        self.plotter.at(3).render()
        return widget, False


class SymMeshWidgetOps:
    """WidgetOps implementation for rendering the Symmetric meshes.

    This widget is display-only. It listens for `"episode_number"` updates,
    loads the target object's YCB mesh, applies the episode-specific rotations,
    scales and positions it, and adds it to the plotter. It does not publish
    any messages.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        updaters: A single `WidgetUpdater` that reacts to `"episode_number"`.
        _locators: Data accessors keyed by name for the parser.
    """

    def __init__(
        self, plotter: Plotter, data_parser: DataParser, ycb_loader: YCBMeshLoader
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.ycb_loader = ycb_loader
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("symmetric_poses", required=True),
                ],
                callback=self.update_plot,
            ),
        ]

        self.default_object_position = (0, 1.5, 0)
        self.plotter.at(4).add(Text2D(txt="Symmetric Poses", pos="top-center"))
        self.widgets: list[Mesh] = []

    def update_plot(self, widget: Mesh, msgs: list[TopicMessage]) -> tuple[Mesh, bool]:
        """Update the target mesh when the episode changes.

        Removes any existing mesh, loads the episode's primary target object,
        applies its Euler rotations, scales and positions it, then adds it to
        the plotter.

        Args:
            widget: The currently displayed mesh, if any.
            msgs: Messages received from the `WidgetUpdater`.

        Returns:
            A tuple `(mesh, False)`. The second value is `False` to indicate
            that no publish should occur.
        """
        if len(self.widgets):
            for w in self.widgets:
                self.plotter.at(4).remove(w)
            self.widgets = []

        msgs_dict = {msg.name: msg.value for msg in msgs}
        data_df = msgs_dict["symmetric_poses"]

        if isinstance(data_df, pd.DataFrame):
            data_df = (
                data_df.assign(
                    evidence=pd.to_numeric(data_df["evidence"], errors="coerce")
                )
                .dropna(subset=["evidence"])
                .nlargest(5, "evidence")
            )

            for gid, rx, ry, rz in data_df[
                ["graph_id", "rot_x", "rot_y", "rot_z"]
            ].itertuples(index=False, name=None):
                with contextlib.suppress(FileNotFoundError):
                    widget = self.ycb_loader.create_mesh(gid).clone(deep=True)
                widget.rotate_x(rx)
                widget.rotate_y(ry)
                widget.rotate_z(rz)
                widget.shift(*self.default_object_position)
                self.plotter.at(4).add(widget)
                self.widgets.append(widget)

        self.plotter.at(4).render()
        return widget, False


class PossibleMatchesPlotWidgetOps:
    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser
        self._locators = self.create_locators()

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                    TopicSpec("evidence_max", required=True),
                    TopicSpec("evidence_std", required=True),
                    TopicSpec("x_percent_threshold", required=True),
                ],
                callback=self.update_plot,
            )
        ]

        self.prev_possible_matches: list[str] = []
        self.possible_matches: list[str] = []

        self.info_widget: Text2D | None = None

    def create_locators(self) -> dict[str, DataLocator]:
        locators = {}
        locators["channel"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="obj"),
                DataLocatorStep.key(name="channel", value="patch"),
            ],
        )

        locators["evidences"] = locators["channel"].extend(
            [DataLocatorStep.key(name="metric", value="evidence")]
        )

        locators["pose_errors"] = locators["channel"].extend(
            [DataLocatorStep.key(name="metric", value="pose_errors")]
        )

        locators["rotations"] = locators["channel"].extend(
            [DataLocatorStep.key(name="metric", value="rotations")]
        )

        return locators

    def state_to_messages(self, state: None) -> Iterable[TopicMessage]:
        messages = [
            TopicMessage(
                name="prev_possible_matches", value=self.prev_possible_matches
            ),
            TopicMessage(name="possible_matches", value=self.possible_matches),
        ]
        return messages

    def _extract_data(self, episode, step):
        available_objects = self.data_parser.query(
            self._locators["channel"], episode=str(episode), step=step
        )

        rows = []
        for graph_id in available_objects:
            evidences = self.data_parser.extract(
                self._locators["evidences"],
                episode=str(episode),
                step=step,
                obj=graph_id,
            )

            # empty hypothesis space for this graph_id
            if not len(evidences):
                continue

            pose_errors = self.data_parser.extract(
                self._locators["pose_errors"],
                episode=str(episode),
                step=step,
                obj=graph_id,
            )

            rotations = self.data_parser.extract(
                self._locators["rotations"],
                episode=str(episode),
                step=step,
                obj=graph_id,
            )

            # pick the index of the max evidence
            max_idx = np.array(evidences, dtype=float).argmax()

            rows.append(
                {
                    "graph_id": graph_id,
                    "evidence": evidences[max_idx],
                    "pose_error": pose_errors[max_idx],
                    "rot_x": rotations[max_idx][0],
                    "rot_y": rotations[max_idx][1],
                    "rot_z": rotations[max_idx][2],
                }
            )

        return pd.DataFrame(
            rows,
            columns=["graph_id", "evidence", "pose_error", "rot_x", "rot_y", "rot_z"],
        )

    def _threshold_possible_matches(
        self, data_df, max_ge_threshold, std_ge_threshold, x_percent_threshold
    ):
        evidences = data_df["evidence"].to_numpy(dtype=float)
        max_ge = float(np.max(evidences))
        std_ge = float(np.std(evidences))
        show_x_percent = False

        if (std_ge > std_ge_threshold) or (max_ge < max_ge_threshold):
            if max_ge > max_ge_threshold:
                x_percent_of_max = max_ge * x_percent_threshold / 100.0
                th = max_ge - x_percent_of_max
                show_x_percent = True

            # if all evidences <= 0, this results in no possibles
            else:
                th = max_ge_threshold

            possible_mask = evidences > th
        else:
            possible_mask = np.ones_like(evidences, dtype=bool)

        return data_df.assign(possible=possible_mask.astype(bool)), show_x_percent

    def _add_info_text(self, hyp: pd.DataFrame):
        info = (
            f"Total Objects: {len(hyp['possible'])}\n"
            + f"Possible Objects: {hyp['possible'].sum()}\n"
            + f"Evidence Std: {hyp['evidence'].max():.2f}"
        )
        self.info_widget = Text2D(txt=info, pos="top-left", font=FONT)
        self.plotter.at(0).add(self.info_widget)

    def _add_correlation_figure(
        self, data_df, show_x_percent=False, x_percent_threshold=20
    ) -> Image:
        data_df = data_df.copy()

        g = sns.JointGrid(data=data_df, x="evidence", y="pose_error", height=6)

        sns.scatterplot(
            data=data_df,
            x="evidence",
            y="pose_error",
            hue="possible",
            ax=g.ax_joint,
            s=30,
            alpha=1.0,
            linewidth=0,
            palette=HUE_PALETTE,
            legend=False,
        )

        if show_x_percent:
            max_ge = float(data_df["evidence"].max())
            x_percent_of_max = max_ge * x_percent_threshold / 100.0
            x0 = max_ge - x_percent_of_max
            x1 = max_ge
            y0, y1 = g.ax_joint.get_ylim()

            rect = Rectangle(
                (x0, y0),
                width=(x1 - x0),
                height=(y1 - y0),
                facecolor=HUE_PALETTE["x_percent"],
                alpha=1.0,
                edgecolor="none",
                zorder=0,
            )
            g.ax_joint.add_patch(rect)

        g.ax_joint.set_title("Possible Objects", pad=5)
        g.ax_joint.set_xlabel("Evidence", labelpad=10)
        g.ax_joint.set_ylabel("Pose Error", labelpad=10)
        g.figure.tight_layout()

        widget = Image(g.figure)
        widget.scale(0.7)
        widget.pos(-400, -200, 0)

        plt.close(g.figure)
        self.plotter.at(0).add(widget)

        return widget

    def _get_possible_matches(
        self, episode, step, evidence_max, evidence_std, x_percent_threshold
    ):
        data_df = self._extract_data(episode=episode, step=step)

        data_df, show_x_percent = self._threshold_possible_matches(
            data_df, evidence_max, evidence_std, x_percent_threshold
        )

        possible_matches = (
            data_df.query("possible")
            .sort_values("evidence", ascending=False)["graph_id"]
            .astype(str)
            .tolist()
        )

        return possible_matches, show_x_percent, data_df

    def update_plot(self, widget: None, msgs: list[TopicMessage]) -> tuple[None, bool]:
        if widget is not None:
            self.plotter.at(0).remove(widget)

        if self.info_widget is not None:
            self.plotter.at(0).remove(self.info_widget)
            self.info_widget = None

        msgs_dict = {msg.name: msg.value for msg in msgs}

        # extract previous possible matches for symmetry conditions
        self.prev_possible_matches = []
        if msgs_dict["step_number"] >= 0:
            self.prev_possible_matches, _, _ = self._get_possible_matches(
                episode=msgs_dict["episode_number"],
                step=msgs_dict["step_number"] - 1,
                evidence_max=msgs_dict["evidence_max"],
                evidence_std=msgs_dict["evidence_std"],
                x_percent_threshold=msgs_dict["x_percent_threshold"],
            )

        self.possible_matches, show_x_percent, data_df = self._get_possible_matches(
            episode=msgs_dict["episode_number"],
            step=msgs_dict["step_number"],
            evidence_max=msgs_dict["evidence_max"],
            evidence_std=msgs_dict["evidence_std"],
            x_percent_threshold=msgs_dict["x_percent_threshold"],
        )

        widget = self._add_correlation_figure(
            data_df, show_x_percent, msgs_dict["x_percent_threshold"]
        )

        self._add_info_text(data_df)

        return widget, True


class HypSpacePlotWidgetOps:
    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser
        self._locators = self.create_locators()

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                    TopicSpec("object_evidence_threshold", required=True),
                    TopicSpec("x_percent_threshold", required=True),
                    TopicSpec("path_similarity_threshold", required=True),
                    TopicSpec("pose_similarity_threshold", required=True),
                    TopicSpec("symmetry_overlap_threshold", required=True),
                    TopicSpec("x_percent_threshold", required=True),
                    TopicSpec("prev_possible_matches", required=True),
                    TopicSpec("possible_matches", required=True),
                ],
                callback=self.update_plot,
            )
        ]

        self.info_widget: Text2D | None = None

    def create_locators(self) -> dict[str, DataLocator]:
        locators = {}
        locators["channel"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="obj"),
                DataLocatorStep.key(name="channel", value="patch"),
            ],
        )

        locators["evidences"] = locators["channel"].extend(
            [DataLocatorStep.key(name="metric", value="evidence")]
        )

        locators["pose_errors"] = locators["channel"].extend(
            [DataLocatorStep.key(name="metric", value="pose_errors")]
        )

        locators["rotations"] = locators["channel"].extend(
            [DataLocatorStep.key(name="metric", value="rotations")]
        )

        locators["locations"] = locators["channel"].extend(
            [DataLocatorStep.key(name="metric", value="locations")]
        )

        return locators

    def state_to_messages(self, state: float) -> Iterable[TopicMessage]:
        unique_message = self.data_df if self.unique_pose else None
        symmetry_message = self.data_df if self.symmetric_poses else None
        messages = [
            TopicMessage(name="unique_poses", value=unique_message),
            TopicMessage(name="symmetric_poses", value=symmetry_message),
        ]
        return messages

    def _extract_data(self, episode, step, obj):
        evidences = np.array(
            self.data_parser.extract(
                self._locators["evidences"],
                episode=str(episode),
                step=step,
                obj=obj,
            )
        )
        if not len(evidences):
            return pd.DataFrame(
                columns=[
                    "graph_id",
                    "evidence",
                    "pose_error",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                    "loc_x",
                    "loc_y",
                    "loc_z",
                ]
            )

        pose_errors = np.array(
            self.data_parser.extract(
                self._locators["pose_errors"],
                episode=str(episode),
                step=step,
                obj=obj,
            )
        )

        rotations = np.array(
            self.data_parser.extract(
                self._locators["rotations"],
                episode=str(episode),
                step=step,
                obj=obj,
            )
        )

        locations = np.array(
            self.data_parser.extract(
                self._locators["locations"],
                episode=str(episode),
                step=step,
                obj=obj,
            )
        )

        return pd.DataFrame(
            {
                "graph_id": np.full(evidences.shape[0], obj),
                "evidence": evidences,
                "pose_error": pose_errors,
                "rot_x": rotations[:, 0],
                "rot_y": rotations[:, 1],
                "rot_z": rotations[:, 2],
                "loc_x": locations[:, 0],
                "loc_y": locations[:, 1],
                "loc_z": locations[:, 2],
            }
        )

    def _threshold_possible_ids(
        self, data_df, object_evidence_threshold, x_percent_threshold
    ):
        data_df = data_df.copy()
        ev = data_df["evidence"]

        max_ev = np.max(ev)
        if max_ev > object_evidence_threshold:
            x_of_max = max_ev / 100 * x_percent_threshold
            th = max_ev - x_of_max
            mask = ev > th
        else:
            mask = np.zeros_like(ev, dtype=bool)

        data_df["possible"] = mask.astype(bool)
        return data_df

    def _add_correlation_figure(
        self, data_df, show_x_percent=False, x_percent_threshold=20
    ) -> Image:
        data_df = data_df.copy()

        g = sns.JointGrid(data=data_df, x="evidence", y="pose_error", height=6)

        sns.scatterplot(
            data=data_df,
            x="evidence",
            y="pose_error",
            hue="possible",
            ax=g.ax_joint,
            s=30,
            alpha=1.0,
            linewidth=0,
            palette=HUE_PALETTE,
            legend=False,
        )

        if show_x_percent:
            max_ge = float(data_df["evidence"].max())
            x_percent_of_max = max_ge * x_percent_threshold / 100.0
            x0 = max_ge - x_percent_of_max
            x1 = max_ge
            y0, y1 = g.ax_joint.get_ylim()

            rect = Rectangle(
                (x0, y0),
                width=(x1 - x0),
                height=(y1 - y0),
                facecolor=HUE_PALETTE["x_percent"],
                alpha=1.0,
                edgecolor="none",
                zorder=0,
            )
            g.ax_joint.add_patch(rect)

        g.ax_joint.set_title("Possible Poses", pad=5)
        g.ax_joint.set_xlabel("Evidence", labelpad=10)
        g.ax_joint.set_ylabel("Pose Error", labelpad=10)
        g.figure.tight_layout()

        widget = Image(g.figure)
        widget.scale(0.7)
        widget.pos(0, -200, 0)

        plt.close(g.figure)
        self.plotter.at(0).add(widget)

        return widget

    def _check_unique_poses(
        self, data_df, path_similarity_threshold, pose_similarity_threshold
    ) -> tuple[bool, float, float]:
        data_df = data_df.copy()

        df_pos = data_df[data_df["possible"].astype(bool)]
        if df_pos.empty:
            return False

        # locations
        location_cols = ["loc_x", "loc_y", "loc_z"]
        locs = df_pos.loc[:, location_cols].to_numpy(dtype=float)
        center = locs.mean(axis=0)
        dists = np.linalg.norm(locs - center, axis=1)
        location_max_dist = np.max(dists)
        location_unique = location_max_dist < float(path_similarity_threshold)

        # most likely rotation
        rotation_cols = ["rot_x", "rot_y", "rot_z"]
        row_mlh = data_df.iloc[data_df["evidence"].to_numpy().argmax()]
        mlh_euler = np.array([row_mlh[c] for c in rotation_cols], dtype=float)
        most_likely_r = Rotation.from_euler("xyz", mlh_euler, degrees=True)

        # possible rotations
        eulers = df_pos.loc[:, list(rotation_cols)].to_numpy(dtype=float)
        rots = Rotation.from_euler("xyz", eulers, degrees=True)

        # Relative angles
        rel = most_likely_r.inv() * rots
        angles = np.linalg.norm(rel.as_rotvec(), axis=1)
        rotation_max_dist = np.max(angles)
        rotation_unique = rotation_max_dist <= float(pose_similarity_threshold)

        return (
            bool(location_unique and rotation_unique),
            location_max_dist,
            rotation_max_dist,
        )

    def _check_symmetry(
        self,
        data_df,
        episode_number,
        step_number,
        prev_possible_matches,
        possible_matches,
        symmetry_overlap_threshold,
        object_evidence_threshold,
        x_percent_threshold,
    ) -> tuple[bool, float | None]:
        if (
            len(prev_possible_matches) != 1
            or prev_possible_matches[0] != possible_matches[0]
        ):
            return False, None

        prev_data_df = self._extract_data(
            episode=episode_number,
            step=step_number,
            obj=prev_possible_matches[0],
        )

        prev_data_df = self._threshold_possible_ids(
            prev_data_df,
            object_evidence_threshold,
            x_percent_threshold,
        )

        prev_mask = prev_data_df["possible"].to_numpy(dtype=bool)
        curr_mask = data_df["possible"].to_numpy(dtype=bool)

        prev_possible = set(np.flatnonzero(prev_mask))
        curr_possible = set(np.flatnonzero(curr_mask))

        # If either set is empty, symmetry not possible
        if not prev_possible or not curr_possible:
            return False, None

        overlap = prev_possible.intersection(curr_possible)
        overlap_ratio = len(overlap) / len(curr_possible)
        return overlap_ratio > float(symmetry_overlap_threshold), overlap_ratio

    def _add_info_text(
        self,
        hyp: pd.DataFrame,
        path_dist: float | None,
        pose_dist: float | None,
        sym_overlap: float | None,
    ):
        info = (
            f"Total Hypotheses: {len(hyp['possible'])}\n"
            + f"Possible Hypotheses: {hyp['possible'].sum()}"
        )

        if path_dist is not None:
            info += f"\nPath distance: {path_dist:.2f}"

        if pose_dist is not None:
            info += f"\nPose distance: {pose_dist:.2f}"

        if sym_overlap is not None:
            info += f"\nSymmetry overlap: {sym_overlap:.2f}"

        self.info_widget = Text2D(txt=info, pos="top-right", font=FONT)
        self.plotter.at(0).add(self.info_widget)

    def update_plot(self, widget: None, msgs: list[TopicMessage]) -> tuple[None, bool]:
        if widget is not None:
            self.plotter.at(0).remove(widget)

        if self.info_widget is not None:
            self.plotter.at(0).remove(self.info_widget)
            self.info_widget = None

        msgs_dict = {msg.name: msg.value for msg in msgs}

        if not len(msgs_dict["possible_matches"]):
            return widget, False

        data_df = self._extract_data(
            episode=msgs_dict["episode_number"],
            step=msgs_dict["step_number"],
            obj=msgs_dict["possible_matches"][0],
        )

        data_df = self._threshold_possible_ids(
            data_df,
            msgs_dict["object_evidence_threshold"],
            msgs_dict["x_percent_threshold"],
        )

        single_match = len(msgs_dict["possible_matches"]) == 1
        pass_evidence_threshold = (
            data_df["evidence"].max() > msgs_dict["object_evidence_threshold"]
        )
        show_x_percent = single_match and pass_evidence_threshold
        if not show_x_percent:
            data_df["possible"] = False

        widget = self._add_correlation_figure(
            data_df=data_df,
            show_x_percent=show_x_percent,
            x_percent_threshold=msgs_dict["x_percent_threshold"],
        )
        self.data_df = data_df[data_df["possible"]]

        if show_x_percent:
            self.unique_pose, path_dist, pose_dist = self._check_unique_poses(
                data_df,
                path_similarity_threshold=msgs_dict["path_similarity_threshold"],
                pose_similarity_threshold=msgs_dict["pose_similarity_threshold"],
            )

            self.symmetric_poses, sym_overlap = self._check_symmetry(
                data_df,
                episode_number=str(msgs_dict["episode_number"]),
                step_number=msgs_dict["step_number"],
                prev_possible_matches=msgs_dict["prev_possible_matches"],
                possible_matches=msgs_dict["possible_matches"],
                symmetry_overlap_threshold=msgs_dict["symmetry_overlap_threshold"],
                object_evidence_threshold=msgs_dict["object_evidence_threshold"],
                x_percent_threshold=msgs_dict["x_percent_threshold"],
            )

        else:
            self.unique_pose = False
            self.symmetric_poses = False
            path_dist, pose_dist, sym_overlap = None, None, None

        self._add_info_text(data_df, path_dist, pose_dist, sym_overlap)

        return widget, True


class InteractivePlot:
    """An interactive plot for correlation of evidence slopes and pose errors.

    This visualization provides means for inspecting the resampling of hypotheses
    at every step. The main view is a scatter correlation plot where pose error is
    expected to decrease as evidence slope increases. You can click points to inspect
    the selected hypothesis and view its 3D mesh with basic stats. Additional controls
    let you switch objects and threshold by hypothesis age.

    Args:
        exp_path: Path to the experiment log consumed by `DataParser`.
        data_path: Root directory containing YCB meshes for `YCBMeshLoader`.

    Attributes:
        data_parser: Parser that reads the JSON log file and serves queries.
        ycb_loader: Loader that provides textured YCB meshes.
        event_bus: Publisher used to route `TopicMessage` events among widgets.
        plotter: Vedo `Plotter` hosting all widgets.
        scheduler: Debounce scheduler bound to the plotter interactor.
        _widgets: Mapping of widget names to their `Widget` instances. It
            includes episode and step sliders, primary/prev/next buttons, an
            age-threshold slider, the correlation plot, and mesh viewers.

    """

    def __init__(
        self,
        exp_path: str,
        data_path: str,
    ):
        renderer_areas = [
            {"bottomleft": (0.0, 0.0), "topright": (1.0, 1.0)},
            {"bottomleft": (0.01, 0.5), "topright": (0.21, 0.7)},  # Groudtruth
            {"bottomleft": (0.01, 0.3), "topright": (0.21, 0.5)},  # MLH
            {"bottomleft": (0.79, 0.5), "topright": (0.99, 0.7)},  # Unique poses
            {"bottomleft": (0.79, 0.3), "topright": (0.99, 0.5)},  # Symmetric poses
        ]

        self.axes_dict = {
            "xrange": (-0.05, 0.05),
            "yrange": (1.45, 1.55),
            "zrange": (-0.05, 0.05),
        }
        self.cam_dict = {"pos": (0, 0, 1500), "focal_point": (0, 0, 0)}

        self.data_parser = DataParser(exp_path)
        self.ycb_loader = YCBMeshLoader(data_path)
        self.event_bus = Publisher()
        self.plotter = Plotter(shape=renderer_areas, sharecam=False).render()
        self.scheduler = VtkDebounceScheduler(self.plotter.interactor, period_ms=33)

        # create and add the widgets to the plotter
        self._widgets = self.create_widgets()
        for w in self._widgets.values():
            w.add()
        self._widgets["episode_slider"].set_state(0)
        self._widgets["x_percent_slider"].set_state(DEFAULTS["x_percent_threshold"])
        self._widgets["max_evidence_slider"].set_state(DEFAULTS["pm_max"])
        self._widgets["std_evidence_slider"].set_state(DEFAULTS["pm_std"])
        self._widgets["symmetry_overlap_slider"].set_state(
            DEFAULTS["symmetry_overlap_threshold"]
        )
        self._widgets["path_similarity_slider"].set_state(
            DEFAULTS["path_similarity_threshold"]
        )
        self._widgets["pose_similarity_slider"].set_state(
            DEFAULTS["pose_similarity_threshold"]
        )
        self._widgets["object_evidence_slider"].set_state(
            DEFAULTS["object_evidence_threshold"]
        )

        self.plotter.at(0).show(
            camera=deepcopy(self.cam_dict),
            interactive=False,
            resetcam=False,
        )
        self.plotter.at(1).show(
            axes=deepcopy(self.axes_dict),
            interactive=False,
            resetcam=True,
        )

        self.plotter.at(2).show(
            axes=deepcopy(self.axes_dict),
            interactive=False,
            resetcam=True,
        )

        self.plotter.at(3).show(
            axes=deepcopy(self.axes_dict),
            interactive=False,
            resetcam=True,
        )

        self.plotter.at(4).show(
            axes=deepcopy(self.axes_dict),
            interactive=True,
            resetcam=True,
        )

    def create_widgets(self):
        widgets = {}

        widgets["click_widget"] = Widget[None, None](
            widget_ops=ClickWidgetOps(
                plotter=self.plotter, cam_dict=deepcopy(self.cam_dict)
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        widgets["episode_slider"] = Widget[Slider2D, int](
            widget_ops=EpisodeSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["step_slider"] = Widget[Slider2D, int](
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["reset_button"] = Widget[Button, str](
            widget_ops=ResetButtonWidgetOps(plotter=self.plotter),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=False,
        )

        widgets["x_percent_slider"] = Widget[Slider2D, int](
            widget_ops=XPercentSliderWidgetOps(plotter=self.plotter),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["object_evidence_slider"] = Widget[Slider2D, float](
            widget_ops=ObjectEvidenceSliderWidgetOps(plotter=self.plotter),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["path_similarity_slider"] = Widget[Slider2D, float](
            widget_ops=PathSimilaritySliderWidgetOps(plotter=self.plotter),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["pose_similarity_slider"] = Widget[Slider2D, float](
            widget_ops=PoseSimilaritySliderWidgetOps(plotter=self.plotter),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["symmetry_overlap_slider"] = Widget[Slider2D, float](
            widget_ops=SymmetryOverlapSliderWidgetOps(plotter=self.plotter),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["std_evidence_slider"] = Widget[Slider2D, float](
            widget_ops=StdEvidenceSliderWidgetOps(plotter=self.plotter),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["max_evidence_slider"] = Widget[Slider2D, float](
            widget_ops=MaxEvidenceSliderWidgetOps(plotter=self.plotter),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["primary_mesh"] = Widget[Mesh, None](
            widget_ops=GtMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["Mlh_mesh"] = Widget[Mesh, None](
            widget_ops=MlhMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["unique_mesh"] = Widget[Mesh, None](
            widget_ops=UniqueMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["symmetric_mesh"] = Widget[Mesh, None](
            widget_ops=SymMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["possible_matches_plot"] = Widget[None, None](
            widget_ops=PossibleMatchesPlotWidgetOps(
                plotter=self.plotter, data_parser=self.data_parser
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.3,
            dedupe=False,
        )

        widgets["hyp_space_plot"] = Widget[None, None](
            widget_ops=HypSpacePlotWidgetOps(
                plotter=self.plotter, data_parser=self.data_parser
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.3,
            dedupe=False,
        )

        return widgets


@register(
    "interactive_terminal_conditions",
    description="Detailed inspection of terminal conditions",
)
def main(experiment_log_dir: str, objects_mesh_dir: str) -> int:
    """Interactive visualization for inspecting the terminal conditions.

    Args:
        experiment_log_dir: Path to the experiment directory containing the detailed
            stats file.
        objects_mesh_dir: Path to the root directory of YCB object meshes.

    Returns:
        Exit code.
    """
    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    data_path = str(Path(objects_mesh_dir).expanduser())

    InteractivePlot(experiment_log_dir, data_path)

    return 0


@attach_args("interactive_terminal_conditions")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
    p.add_argument(
        "--objects_mesh_dir",
        default="~/tbp/data/habitat/objects/ycb/meshes",
        help=("The directory containing the mesh objects."),
    )
