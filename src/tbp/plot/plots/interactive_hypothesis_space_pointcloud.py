# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import bisect
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pubsub.core import Publisher
from vedo import Button, Image, Line, Mesh, Plotter, Points, Slider2D, Sphere

from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
    PretrainedModelsLoader,
    YCBMeshLoader,
)
from tbp.interactive.topics import TopicMessage, TopicSpec
from tbp.interactive.utils import (
    Location3D,
)
from tbp.interactive.widget_updaters import WidgetUpdater
from tbp.interactive.widgets import (
    VtkDebounceScheduler,
    Widget,
    extract_button_state,
    extract_slider_state,
    set_button_state,
    set_slider_state,
)
from tbp.plot.registry import attach_args, register

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)


class StepMapper:
    """Bidirectional mapping between global step indices and (episode, local_step).

    Global steps are defined as the concatenation of all local episode steps:

        episode 0: steps [0, ..., n0 - 1]
        episode 1: steps [0, ..., n1 - 1]
        ...

    Global index is:
        [0, ..., n0 - 1, n0, ..., n0 + n1 - 1, ...]
    """

    def __init__(self, data_parser: DataParser) -> None:
        self.data_parser = data_parser
        self._locators = self._create_locators()

        # number of steps in each episode
        self._episode_lengths: list[int] = self._compute_episode_lengths()

        # global offset of each episode
        self._prefix_sums: list[int] = self._compute_prefix_sums()

    def _create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used to access episode steps.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}
        locators["step"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
            ]
        )
        return locators

    def _compute_episode_lengths(self) -> list[int]:
        locator = self._locators["step"]

        episode_lengths: list[int] = []

        for episode in self.data_parser.query(locator):
            episode_lengths.append(
                len(self.data_parser.extract(locator, episode=episode))
            )

        if not episode_lengths:
            raise RuntimeError("No episodes found while computing episode lengths.")

        return episode_lengths

    def _compute_prefix_sums(self) -> list[int]:
        prefix_sums = [0]
        for length in self._episode_lengths:
            prefix_sums.append(prefix_sums[-1] + length)
        return prefix_sums

    @property
    def num_episodes(self) -> int:
        return len(self._episode_lengths)

    @property
    def total_num_steps(self) -> int:
        """Total number of steps across all episodes."""
        return self._prefix_sums[-1]

    def global_to_local(self, global_step: int) -> tuple[int, int]:
        """Convert a global step index into (episode, local_step).

        Args:
            global_step: Global step index in the range
                `[0, total_num_steps)`.

        Returns:
            A pair `(episode, local_step)` such that:
              * `episode` is the zero based episode index.
              * `local_step` is the zero based step index within that episode.

        Raises:
            IndexError: If `global_step` is negative or not less than
                `total_num_steps`.
        """
        if global_step < 0 or global_step >= self.total_num_steps:
            raise IndexError(
                f"global_step {global_step} is out of range [0, {self.total_num_steps})"
            )

        episode = bisect.bisect_right(self._prefix_sums, global_step) - 1
        local_step = global_step - self._prefix_sums[episode]
        return episode, local_step

    def local_to_global(self, episode: int, step: int) -> int:
        """Convert an (episode, local_step) pair into a global step index.

        Args:
            episode: Zero based episode index in the range
                `[0, num_episodes)`.
            step: Zero based step index within the given episode. Must be in
                `[0, number_of_steps_in_episode)`.

        Returns:
            The corresponding global step index, in the range
            `[0, total_num_steps)`.

        Raises:
            IndexError: If `episode` is out of range, or if `step` is out of
                range for the given episode.
        """
        if episode < 0 or episode >= self.num_episodes:
            raise IndexError(
                f"episode {episode} is out of range [0, {self.num_episodes})"
            )

        num_steps_in_episode = self._episode_lengths[episode]
        if step < 0 or step >= num_steps_in_episode:
            raise IndexError(
                f"step {step} is out of range [0, {num_steps_in_episode}) "
                f"for episode {episode}"
            )

        return self._prefix_sums[episode] + step


class StepSliderWidgetOps:
    """WidgetOps implementation for a Step slider.

    This class adds a slider widget for the global step. It uses the step mapper
    to retrieve information about the total number of steps and the mapping between
    global and local step indices. The published state is in the local format (i.e.,
    episode and local step values).

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            JSON log file.
        step_mapper: A mapper between local and global step indices.
        updaters: A list with a single `WidgetUpdater` that reacts to the
            `"episode_number"` topic and calls `update_slider_range`.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
        _locators: Data accessors keyed by name that instruct the `DataParser`
            how to retrieve the required information.
    """

    def __init__(
        self,
        plotter: Plotter,
        data_parser: DataParser,
        step_mapper: StepMapper,
    ) -> None:
        self.plotter = plotter
        self.data_parser = data_parser

        self._add_kwargs = {
            "xmin": 0,
            "xmax": 10,
            "value": 0,
            "pos": [(0.11, 0.06), (0.89, 0.06)],
            "title": "Step",
            "show_value": False,
        }

        self.step_mapper = step_mapper
        self.current_episode = -1

    def add(self, callback: Callable) -> Slider2D:
        kwargs = deepcopy(self._add_kwargs)
        kwargs.update({"xmax": self.step_mapper.total_num_steps - 1})
        widget = self.plotter.at(0).add_slider(callback, **kwargs)
        self.plotter.at(0).render()
        return widget

    def remove(self, widget: Slider2D) -> None:
        self.plotter.at(0).remove(widget)
        self.plotter.at(0).render()

    def extract_state(self, widget: Slider2D) -> int:
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        episode, step = self.step_mapper.global_to_local(state)

        messages = []

        # Only publish episode number if changed
        if self.current_episode != episode:
            messages.append(TopicMessage(name="episode_number", value=episode))
            self.current_episode = episode

        messages.append(TopicMessage(name="step_number", value=step))

        return messages


class GtMeshWidgetOps:
    """WidgetOps implementation for rendering the ground-truth target mesh.

    This widget listens for "episode_number" and "step_number" to update the
    ground-truth primary target mesh and the agent/patch location on the object
    at the current step. The widget also listens for buttons that show/hide the
    agent and sensor patch history locations.

    This widget does not publish any messages.
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
                    TopicSpec("transparency_value", required=True),
                ],
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
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                    TopicSpec("agent_path_button", required=True),
                ],
                callback=self.update_agent_path,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                    TopicSpec("patch_path_button", required=True),
                ],
                callback=self.update_patch_path,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("transparency_value", required=True),
                ],
                callback=self.update_transparency,
            ),
        ]
        self._locators = self.create_locators()

        self.agent_sphere: Sphere | None = None
        self.gaze_line: Line | None = None

        self.agent_path_spheres: list[Sphere] = []
        self.agent_path_line: Line | None = None

        self.patch_path_spheres: list[Sphere] = []
        self.patch_path_line: Line | None = None

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

    def remove(self, widget: Mesh) -> None:
        if widget is not None:
            self.plotter.at(1).remove(widget)
            self.plotter.at(1).render()

    def update_mesh(
        self, widget: Mesh | None, msgs: list[TopicMessage]
    ) -> tuple[Mesh | None, bool]:
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
        self.remove(widget)
        msgs_dict = {msg.name: msg.value for msg in msgs}

        locator = self._locators["target"]
        target = self.data_parser.extract(
            locator, episode=str(msgs_dict["episode_number"])
        )
        target_id = target["primary_target_object"]
        target_rot = target["primary_target_rotation_euler"]
        target_pos = target["primary_target_position"]
        widget = self.ycb_loader.create_mesh(target_id).clone(deep=True)
        widget.rotate_x(target_rot[0])
        widget.rotate_y(target_rot[1])
        widget.rotate_z(target_rot[2])
        widget.shift(*target_pos)
        widget.alpha(1.0 - msgs_dict["transparency_value"])

        self.plotter.at(1).add(widget)
        self.plotter.at(1).render()

        return widget, False

    def update_agent(
        self, widget: Mesh | None, msgs: list[TopicMessage]
    ) -> tuple[Mesh | None, bool]:
        """Update the agent and sensor patch location on the object.

        Args:
            widget: The currently displayed mesh, if any.
            msgs: Messages received from the `WidgetUpdater`.

        Returns:
            A tuple `(widget, False)`. The second value is `False` to indicate
            that no publish should occur.
        """
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
            sm_step=int(mapping[step_number]),
        )

        patch_pos = self.data_parser.extract(
            self._locators["patch_location"],
            episode=str(episode_number),
            step=step_number,
        )

        if self.agent_sphere is None:
            self.agent_sphere = Sphere(pos=agent_pos, r=0.004)
            self.plotter.at(1).add(self.agent_sphere)
        self.agent_sphere.pos(agent_pos)

        if self.gaze_line is None:
            self.gaze_line = Line(agent_pos, patch_pos, c="black", lw=4)
            self.plotter.at(1).add(self.gaze_line)
        self.gaze_line.points = [agent_pos, patch_pos]

        self.plotter.at(1).render()

        return widget, False

    def _clear_agent_path(self) -> None:
        for s in self.agent_path_spheres:
            self.plotter.at(1).remove(s)
        self.agent_path_spheres.clear()
        if self.agent_path_line is not None:
            self.plotter.at(1).remove(self.agent_path_line)
            self.agent_path_line = None

    def update_agent_path(
        self, widget: Mesh | None, msgs: list[TopicMessage]
    ) -> tuple[Mesh | None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]
        path_button = msgs_dict["agent_path_button"]

        # Clear existing path
        self._clear_agent_path()

        if path_button == "Agent Path: On":
            steps_mask = self.data_parser.extract(
                self._locators["steps_mask"], episode=str(episode_number)
            )
            mapping = np.flatnonzero(steps_mask)

            if len(mapping) == 0:
                self.plotter.at(1).render()
                return widget, False

            # Clamp step_number to valid range
            max_step_idx = min(step_number, len(mapping) - 1)

            # Collect all agent positions up to the current step
            points: list[np.ndarray] = []
            for k in range(max_step_idx + 1):
                agent_pos = self.data_parser.extract(
                    self._locators["agent_location"],
                    episode=str(episode_number),
                    sm_step=int(mapping[k]),
                )
                points.append(agent_pos)

            # Create small spheres at each position
            for p in points:
                sphere = Sphere(pos=p, r=0.002)
                self.plotter.at(1).add(sphere)
                self.agent_path_spheres.append(sphere)

            # Create a polyline connecting all points
            if len(points) >= 2:
                self.agent_path_line = Line(points, c="red", lw=1)
                self.plotter.at(1).add(self.agent_path_line)

        self.plotter.at(1).render()
        return widget, False

    def _clear_patch_path(self) -> None:
        for s in self.patch_path_spheres:
            self.plotter.at(1).remove(s)
        self.patch_path_spheres.clear()
        if self.patch_path_line is not None:
            self.plotter.at(1).remove(self.patch_path_line)
            self.patch_path_line = None

    def update_patch_path(
        self, widget: None, msgs: list[TopicMessage]
    ) -> tuple[None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]
        path_button = msgs_dict["patch_path_button"]

        # Clear existing path
        self._clear_patch_path()

        if path_button == "Patch Path: On":
            # Collect all patch positions up to the current step
            points: list[np.ndarray] = []
            max_step_idx = max(step_number, 0)

            for k in range(max_step_idx + 1):
                patch_pos = self.data_parser.extract(
                    self._locators["patch_location"],
                    episode=str(episode_number),
                    step=k,
                )
                points.append(patch_pos)

            # Create small black spheres at each patch position
            for p in points:
                sphere = Sphere(pos=p, r=0.002, c="black")
                self.plotter.at(1).add(sphere)
                self.patch_path_spheres.append(sphere)

            # Create a thin black polyline connecting all patch positions
            if len(points) >= 2:
                self.patch_path_line = Line(points, c="black", lw=1)
                self.plotter.at(1).add(self.patch_path_line)

        self.plotter.at(1).render()
        return widget, False

    def update_transparency(
        self, widget: None, msgs: list[TopicMessage]
    ) -> tuple[None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        widget.alpha(1.0 - msgs_dict["transparency_value"])
        self.plotter.at(1).render()
        return widget, False


class TransparencySliderWidgetOps:
    """WidgetOps implementation for the transparency slider.

    This widget provides a slider to control the transparency of the mesh
    object. It publishes on the topic `transparency_value` a float value between 0.0
    and 1.0.
    """

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter

        self._add_kwargs = {
            "xmin": 0.0,
            "xmax": 1.0,
            "value": 0.0,
            "pos": [(0.05, 0.0), (0.05, 0.4)],
            "title": "Mesh Transparency",
            "title_size": 2,
            "slider_width": 0.04,
            "tube_width": 0.015,
        }

    def add(self, callback: Callable) -> Slider2D:
        widget = self.plotter.at(1).add_slider(callback, **self._add_kwargs)
        widget.GetRepresentation().SetLabelHeight(0.05)
        self.plotter.at(1).render()
        return widget

    def extract_state(self, widget: Slider2D) -> float:
        return extract_slider_state(widget, round_value=False)

    def set_state(self, widget: Slider2D, value: float) -> None:
        set_slider_state(widget, value)

    def state_to_messages(self, state: float) -> Iterable[TopicMessage]:
        return [TopicMessage(name="transparency_value", value=state)]


class AgentPathButtonWidgetOps:
    """WidgetOps implementation for showing/hiding the Agent path.

    This widget provides a button to switch between showing and hiding the
    agent path. The published state here is `Agent Path: On` or `Agent Path: Off`
    and it is published on the topic `agent_path_button`.
    """

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = {
            "pos": (0.16, 0.98),
            "states": ["Agent Path: On", "Agent Path: Off"],
            "c": ["w", "w"],
            "bc": ["dg", "dr"],
            "size": 30,
            "font": "Calco",
            "bold": True,
        }

    def add(self, callback: Callable) -> Button:
        widget = self.plotter.at(0).add_button(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Button) -> str:
        return extract_button_state(widget)

    def set_state(self, widget: Button, value: str) -> None:
        set_button_state(widget, value)

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        messages = [
            TopicMessage(name="agent_path_button", value=state),
        ]
        return messages


class PatchPathButtonWidgetOps:
    """WidgetOps implementation for showing/hiding the sensor patch path.

    This widget provides a button to switch between showing and hiding the
    patch path. The published state here is `Patch Path: On` or `Patch Path: Off`
    and it is published on the topic `patch_path_button`.
    """

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = {
            "pos": (0.37, 0.98),
            "states": ["Patch Path: On", "Patch Path: Off"],
            "c": ["w", "w"],
            "bc": ["dg", "dr"],
            "size": 30,
            "font": "Calco",
            "bold": True,
        }

    def add(self, callback: Callable) -> Button:
        widget = self.plotter.at(0).add_button(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Button) -> str:
        return extract_button_state(widget)

    def set_state(self, widget: Button, value: str) -> None:
        set_button_state(widget, value)

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        messages = [
            TopicMessage(name="patch_path_button", value=state),
        ]
        return messages


class HypSpaceWidgetOps:
    """WidgetOps implementation for the hypothesis space point cloud widget.

    This widget shows the point cloud of the primary target pretrained model and
    a point cloud of the hypothesis space. The widget listens to different topics
    (e.g., `episode_number`, `step_number`, `hyp_color_button`, `hyp_scope_button`)
    to determine which point cloud to show and how to color the points.
    """

    def __init__(
        self,
        plotter: Plotter,
        data_parser: DataParser,
        models_loader: PretrainedModelsLoader,
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.models_loader = models_loader

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                    TopicSpec("model_button", required=True),
                ],
                callback=self.update_model,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                    TopicSpec("hyp_color_button", required=True),
                    TopicSpec("hyp_scope_button", required=True),
                ],
                callback=self.update_hypotheses,
            ),
        ]

        self._locators = self.create_locators()
        self.hyp_space: Points | None = None
        self.mlh_sphere: Sphere | None = None

    def create_locators(self) -> dict[str, DataLocator]:
        locators = {}
        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="target"),
            ]
        )

        locators["telemetry"] = DataLocator(
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
        return locators

    def remove(self, widget: Mesh) -> None:
        if widget is not None:
            self.plotter.at(2).remove(widget)

    def update_model(
        self, widget: Mesh | None, msgs: list[TopicMessage]
    ) -> tuple[Mesh | None, bool]:
        self.remove(widget)
        msgs_dict = {msg.name: msg.value for msg in msgs}
        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]
        model_button = msgs_dict["model_button"]

        if model_button == "Pretrained Model: On":
            locator = self._locators["target"]
            target = self.data_parser.extract(locator, episode=str(episode_number))
            target_id = target["primary_target_object"]
            target_rot = target["primary_target_rotation_euler"]
            target_pos = target["primary_target_position"]

            widget = self.models_loader.create_model(target_id).clone(deep=True)
            widget.rotate_x(target_rot[0])
            widget.rotate_y(target_rot[1])
            widget.rotate_z(target_rot[2])

            self.plotter.at(2).add(widget)

        self.plotter.at(2).render()
        return widget, False

    def _clear_hyp_space(self) -> None:
        if self.hyp_space is not None:
            self.plotter.at(2).remove(self.hyp_space)
            self.hyp_space = None

        if self.mlh_sphere is not None:
            self.plotter.at(2).remove(self.mlh_sphere)
            self.mlh_sphere = None

    def _extract_obj_telemetry(
        self,
        episode_number: int,
        step_number: int,
        object_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obj_hyps_telemetry = self.data_parser.extract(
            self._locators["telemetry"],
            episode=str(episode_number),
            step=step_number,
            obj=object_id,
        )
        evidences = obj_hyps_telemetry["evidence"]
        locations = obj_hyps_telemetry["locations"]
        pose_errors = obj_hyps_telemetry["pose_errors"]
        ages = obj_hyps_telemetry["hypotheses_updater"]["ages"]
        slopes = obj_hyps_telemetry["hypotheses_updater"]["evidence_slopes"]

        return evidences, locations, pose_errors, ages, slopes

    def _create_hyp_space(
        self,
        episode_number: int,
        step_number: int,
        hyp_color_button: str,
        hyp_scope_button: str,
    ) -> Points:
        curr_object = self.data_parser.extract(
            self._locators["target"], episode=str(episode_number)
        )["primary_target_object"]
        evidences, locations, pose_errors, ages, slopes = self._extract_obj_telemetry(
            episode_number, step_number, curr_object
        )
        pts = Points(np.array(locations), r=6, c="dg")

        if hyp_color_button == "Evidence":
            pts.cmap("viridis", evidences, vmin=0.0)
            pts.add_scalarbar(title="Evidence")
        if hyp_color_button == "MLH":
            mlh = np.zeros_like(evidences, dtype=float)
            pts.cmap("viridis", mlh, vmin=0.0, vmax=1.0)
            mlh_sphere = Sphere(
                pos=locations[int(np.argmax(evidences))], r=0.002, c="yellow"
            )
            self.mlh_sphere = mlh_sphere
            self.plotter.at(2).add(mlh_sphere)
        elif hyp_color_button == "Slope":
            pts.cmap("viridis", slopes, vmin=0.0)
            pts.add_scalarbar(title="Slope")
        elif hyp_color_button == "Ages":
            pts.cmap("viridis", ages, vmin=0.0)
            pts.add_scalarbar(title="Age")
        elif hyp_color_button == "Pose Error":
            pts.cmap("viridis", pose_errors, vmin=0.0)
            pts.add_scalarbar(title="Pose Error")

        return pts

    def update_hypotheses(
        self, widget: Points, msgs: list[TopicMessage]
    ) -> tuple[Points, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]
        hyp_color_button = msgs_dict["hyp_color_button"]
        hyp_scope_button = msgs_dict["hyp_scope_button"]

        self._clear_hyp_space()

        if hyp_scope_button != "Hypotheses: Off":
            hyp_space = self._create_hyp_space(
                episode_number, step_number, hyp_color_button, hyp_scope_button
            )
            self.plotter.at(2).add(hyp_space)
            self.hyp_space = hyp_space

        self.plotter.at(2).render()
        return widget, False


class ModelButtonWidgetOps:
    """WidgetOps implementation for showing/hiding the pretrained model point cloud."""

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = {
            "pos": (0.63, 0.98),
            "states": ["Pretrained Model: On", "Pretrained Model: Off"],
            "c": ["w", "w"],
            "bc": ["dg", "dr"],
            "size": 30,
            "font": "Calco",
            "bold": True,
        }

    def add(self, callback: Callable) -> Button:
        widget = self.plotter.at(0).add_button(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Button) -> str:
        return extract_button_state(widget)

    def set_state(self, widget: Button, value: str) -> None:
        set_button_state(widget, value)

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        messages = [
            TopicMessage(name="model_button", value=state),
        ]
        return messages


class HypScopeButtonWidgetOps:
    """WidgetOps implementation for showing/hiding the hypothesis space."""

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = {
            "pos": (0.85, 0.98),
            "states": ["Hypotheses: Off", "Hypotheses: On"],
            "c": ["w", "w"],
            "bc": ["dr", "dg"],
            "size": 30,
            "font": "Calco",
            "bold": True,
        }

    def add(self, callback: Callable) -> Button:
        widget = self.plotter.at(0).add_button(callback, **self._add_kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Button) -> str:
        return extract_button_state(widget)

    def set_state(self, widget: Button, value: str) -> None:
        set_button_state(widget, value)

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        messages = [
            TopicMessage(name="hyp_scope_button", value=state),
        ]
        return messages


class HypColorButtonWidgetOps:
    """WidgetOps implementation for selecting how to color the hypothesis space.

    Note that this widget is hidden (by moving it outside of the scene) when the
    hypothesis space is deactivated. It listens to `hyp_scope_button` to know if the
    widget should be shown or hidden.
    """

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = {
            "pos": (0.18, 0.15),
            "states": ["None", "Evidence", "MLH", "Pose Error", "Slope", "Ages"],
            "c": ["w", "w", "w", "w", "w", "w"],
            "bc": ["gray", "dg", "ds", "dm", "db", "dc"],
            "size": 30,
            "font": "Calco",
            "bold": True,
        }

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("hyp_scope_button", required=True),
                ],
                callback=self.toggle_button,
            ),
        ]

    def add(self, callback: Callable) -> Button:
        widget = self.plotter.at(2).add_button(callback, **self._add_kwargs)
        self.plotter.at(2).render()
        return widget

    def extract_state(self, widget: Button) -> str:
        return extract_button_state(widget)

    def set_state(self, widget: Button, value: str) -> None:
        set_button_state(widget, value)

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        messages = [
            TopicMessage(name="hyp_color_button", value=state),
        ]
        return messages

    def toggle_button(
        self, widget: Button, msgs: list[TopicMessage]
    ) -> tuple[Button, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}

        if msgs_dict["hyp_scope_button"] == "Hypotheses: Off":
            # Hide the button by moving it outside of the scene
            widget.SetPosition(-10.0, -10.0)
        elif msgs_dict["hyp_scope_button"] == "Hypotheses: On":
            x, y = self._add_kwargs["pos"]
            widget.SetPosition(x, y)

        self.plotter.at(2).render()

        return widget, False


class LinePlotWidgetOps:
    """WidgetOps implementation for the line plot.

    This widget shows a line plot for the max global slope and hypothesis space size
    over time. It also shows the sampling bursts locations as vertical dashed red lines.
    The current step is shown as a vertical solid black line and moves with the step
    slider control.
    """

    def __init__(
        self,
        plotter: Plotter,
        data_parser: DataParser,
        step_mapper: StepMapper,
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.step_mapper = step_mapper
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                ],
                callback=self.update_plot,
            ),
        ]
        self._locators = self.create_locators()
        self.df = self._create_df()

    def create_locators(self) -> dict[str, DataLocator]:
        locators = {}

        base_loc = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="obj"),
                DataLocatorStep.key(name="channel", value="patch"),
            ]
        )

        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="target"),
                DataLocatorStep.key(name="target_stat", value="primary_target_object"),
            ]
        )

        locators["evidence"] = base_loc.extend(
            [
                DataLocatorStep.key(name="telemetry2", value="evidence"),
            ]
        )

        locators["max_slope"] = base_loc.extend(
            [
                DataLocatorStep.key(name="telemetry2", value="hypotheses_updater"),
                DataLocatorStep.key(name="telemetry3", value="max_slope"),
            ]
        )

        locators["added_ids"] = base_loc.extend(
            [
                DataLocatorStep.key(name="telemetry2", value="hypotheses_updater"),
                DataLocatorStep.key(name="telemetry3", value="added_ids"),
            ]
        )

        return locators

    def remove(self, widget: Mesh) -> None:
        if widget is not None:
            self.plotter.at(0).remove(widget)

    def _get_bursts(self) -> list[bool]:
        bursts = []
        for episode in self.data_parser.query(self._locators["evidence"]):
            obj = self.data_parser.extract(self._locators["target"], episode=episode)
            for step in self.data_parser.query(
                self._locators["added_ids"], episode=episode
            ):
                num_added = len(
                    self.data_parser.extract(
                        self._locators["added_ids"],
                        episode=episode,
                        step=step,
                        obj=obj,
                    )
                )
                bursts.append(num_added > 0)
        return bursts

    def _get_hyp_space_sizes(self) -> list[int]:
        hyp_space_sizes = []
        for episode in self.data_parser.query(self._locators["evidence"]):
            obj = self.data_parser.extract(self._locators["target"], episode=episode)
            for step in self.data_parser.query(
                self._locators["evidence"], episode=episode
            ):
                num_hyp = len(
                    self.data_parser.extract(
                        self._locators["evidence"],
                        episode=episode,
                        step=step,
                        obj=obj,
                    )
                )
                hyp_space_sizes.append(num_hyp)
        return hyp_space_sizes

    def _get_max_slopes(self) -> list[float]:
        max_slopes = []
        for episode in self.data_parser.query(self._locators["evidence"]):
            obj = self.data_parser.extract(self._locators["target"], episode=episode)
            for step in self.data_parser.query(
                self._locators["evidence"], episode=episode
            ):
                slope = self.data_parser.extract(
                    self._locators["max_slope"],
                    episode=episode,
                    step=step,
                    obj=obj,
                )

                max_slopes.append(slope)
        return max_slopes

    def _create_df(self) -> pd.DataFrame:
        bursts = self._get_bursts()
        hyp_space_sizes = self._get_hyp_space_sizes()
        max_slopes = self._get_max_slopes()

        if not (len(bursts) == len(hyp_space_sizes) == len(max_slopes)):
            raise ValueError(
                f"Length mismatch: bursts={len(bursts)}, "
                f"sizes={len(hyp_space_sizes)}, slopes={len(max_slopes)}"
            )

        return pd.DataFrame(
            {
                "burst": bursts,
                "hyp_space_size": hyp_space_sizes,
                "max_slope": max_slopes,
            }
        )

    def _create_burst_figure(self, global_step: int) -> plt.Figure:
        df = self.df.copy()

        slopes = df["max_slope"].to_numpy(dtype=float)
        hyp_space_sizes = df["hyp_space_size"].to_numpy(dtype=float)
        bursts = df["burst"].to_numpy(dtype=bool)

        x = np.arange(len(slopes))

        fig, ax_left = plt.subplots(1, 1, figsize=(14, 3))
        ax_right = ax_left.twinx()

        valid_idx = np.where(~np.isnan(slopes))[0]
        start_idx = int(valid_idx[0]) if valid_idx.size > 0 else 0

        if start_idx < len(slopes):
            ax_left.plot(x[start_idx:], slopes[start_idx:], label="Max slope")

        ax_right.plot(x, hyp_space_sizes, linestyle="--", label="Hypothesis space size")

        ax_left.set_ylim(-1.0, 2.0)
        ax_right.set_ylim(0, 10000)

        # Burst locations (red dashed lines)
        add_idx = np.flatnonzero(bursts)
        if add_idx.size > 0:
            ymin, ymax = ax_left.get_ylim()
            ax_left.vlines(
                add_idx,
                ymin,
                ymax,
                colors="red",
                linestyles="--",
                alpha=1.0,
                linewidth=1.0,
                zorder=1,
                label="Burst",
            )

        # Current step marker (single black vertical line)
        if 0 <= global_step < len(slopes):
            ymin, ymax = ax_left.get_ylim()
            ax_left.vlines(
                global_step,
                ymin,
                ymax,
                colors="black",
                linestyles="-",
                linewidth=1.5,
                zorder=2,
                label="Current step",
            )

        ax_left.set_ylabel("Max slope")
        ax_right.set_ylabel("Hyp space size")

        # Merge legends and place above
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        all_lines = lines_left + lines_right
        all_labels = labels_left + labels_right

        if all_lines:
            ax_left.legend(
                all_lines,
                all_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.20),
                ncol=len(all_lines),
                frameon=False,
                columnspacing=1.0,
                handletextpad=0.5,
            )

        fig.tight_layout(rect=[0, 0, 1, 0.90])

        widget = Image(fig)
        plt.close(fig)
        return widget

    def update_plot(
        self, widget: Image, msgs: list[TopicMessage]
    ) -> tuple[Image, bool]:
        self.remove(widget)
        msgs_dict = {msg.name: msg.value for msg in msgs}
        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]
        global_step = self.step_mapper.local_to_global(episode_number, step_number)

        widget = self._create_burst_figure(global_step)
        widget.pos(-400, -150, 0)

        self.plotter.at(0).add(widget)
        self.plotter.at(0).render()

        return widget, False


class ClickWidgetOps:
    """Captures 3D click positions and publish them on the bus.

    This class registers plotter-level mouse callbacks. A right-click
    resets the camera pose. There is no visual widget created by this class.

    Attributes:
        plotter: The `vedo.Plotter` where callbacks are installed.
        cam_dict: Dictionary for camera default specs.
        click_location: Last picked 3D location, if any.
        _on_change_cb: The widget callback to invoke on left-click.
    """

    def __init__(self, plotter: Plotter, cam_dict: dict[str, Any]) -> None:
        self.plotter = plotter
        self.cam_dict = cam_dict
        self.click_location: Location3D
        self._on_change_cb: Callable

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
        self._on_change_cb = callback
        self.plotter.at(0).add_callback("RightButtonPress", self.on_right_click)

    def align_camera(self, cam_a: Any, cam_b: Any) -> None:
        """Align the camera objects."""
        cam_a.SetPosition(cam_b.GetPosition())
        cam_a.SetFocalPoint(cam_b.GetFocalPoint())
        cam_a.SetViewUp(cam_b.GetViewUp())
        cam_a.SetClippingRange(cam_b.GetClippingRange())
        cam_a.SetParallelScale(cam_b.GetParallelScale())

    def on_right_click(self, event) -> None:
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
        elif event.at == 1:
            cam_clicked = self.plotter.renderers[1].GetActiveCamera()
            cam_copy = self.plotter.renderers[2].GetActiveCamera()
            self.align_camera(cam_copy, cam_clicked)
        elif event.at == 2:
            cam_clicked = self.plotter.renderers[1].GetActiveCamera()
            cam_copy = self.plotter.renderers[2].GetActiveCamera()
            self.align_camera(cam_clicked, cam_copy)


class InteractivePlot:
    """An interactive plot for hypotheses and sampling bursts location.

    This visualizations features the following:
    - A plot of the primary target mesh (with mesh transparency slider) with the agent
        and patch location on the object.
    - Buttons to activate or deactivate the history of agent and/or patch locations.
    - A plot of the pretrained model for the primary target with a button to show the
        hypothesis space for this object.
    - The hypothesis space pointcloud can be coloured by different metrics, such as
        evidence, mlh, slope, pose error or age.
    - A line plot showing the maximum global slope (burst trigger), the
        location/duration of the sampling bursts and the hypothesis space size over
        time.

    Args:
        exp_path: Path to the experiment log consumed by `DataParser`.
        data_path: Root directory containing YCB meshes for `YCBMeshLoader`.
        models_path: Path to the pretrained models pt file.

    Attributes:
        data_parser: Parser that reads the JSON log file and serves queries.
        step_mapper: Mapper between global step index and local step index (with episode
            number).
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
        models_path: str,
    ):
        renderer_areas = [
            {"bottomleft": (0.0, 0.0), "topright": (1.0, 1.0)},
            {"bottomleft": (0.05, 0.5), "topright": (0.49, 0.9)},
            {"bottomleft": (0.51, 0.5), "topright": (0.95, 0.9)},
        ]

        self.axes_dict = {
            "xrange": (-0.05, 0.05),
            "yrange": (1.45, 1.55),
            "zrange": (-0.05, 0.05),
        }
        self.cam_dict = {"pos": (300, 200, 1500), "focal_point": (300, 200, 0)}

        self.data_parser = DataParser(exp_path)
        self.step_mapper = StepMapper(self.data_parser)
        self.ycb_loader = YCBMeshLoader(data_path)
        self.models_loader = PretrainedModelsLoader(models_path)
        self.event_bus = Publisher()
        self.plotter = Plotter(shape=renderer_areas, sharecam=False).render()
        self.scheduler = VtkDebounceScheduler(self.plotter.interactor, period_ms=33)

        # create and add the widgets to the plotter
        self._widgets = self.create_widgets()
        for w in self._widgets.values():
            w.add()
        self._widgets["step_slider"].set_state(0)
        self._widgets["agent_path_button"].set_state("Agent Path: Off")
        self._widgets["patch_path_button"].set_state("Patch Path: Off")
        self._widgets["transparency_slider"].set_state(0.0)
        self._widgets["model_button"].set_state("Pretrained Model: On")
        self._widgets["hyp_color_button"].set_state("None")
        self._widgets["hyp_scope_button"].set_state("Hypotheses: Off")

        self.plotter.at(0).show(
            camera=deepcopy(self.cam_dict),
            interactive=False,  # Must be set to False if not the last `show` call
            resetcam=False,
        )
        self.plotter.at(1).show(
            axes=deepcopy(self.axes_dict),
            interactive=False,  # Must be set to False if not the last `show` call
            resetcam=True,
        )

        self.plotter.at(2).show(
            axes=deepcopy(self.axes_dict),
            interactive=True,  # Must be set to True on the last `show` call
            resetcam=True,
        )

        # === No code runs after the last interactive call === #

    def create_widgets(self):
        widgets = {}

        widgets["step_slider"] = Widget[Slider2D, int](
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                step_mapper=self.step_mapper,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        widgets["primary_mesh"] = Widget[Mesh, None](
            widget_ops=GtMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["transparency_slider"] = Widget[Slider2D, float](
            widget_ops=TransparencySliderWidgetOps(
                plotter=self.plotter,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        widgets["agent_path_button"] = Widget[Button, str](
            widget_ops=AgentPathButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=True,
        )

        widgets["patch_path_button"] = Widget[Button, str](
            widget_ops=PatchPathButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=True,
        )

        widgets["hyp_space_viz"] = Widget[Mesh, None](
            widget_ops=HypSpaceWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                models_loader=self.models_loader,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["model_button"] = Widget[Button, str](
            widget_ops=ModelButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=True,
        )

        widgets["hyp_color_button"] = Widget[Button, str](
            widget_ops=HypColorButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=True,
        )

        widgets["hyp_scope_button"] = Widget[Button, str](
            widget_ops=HypScopeButtonWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=True,
        )

        widgets["line_plot"] = Widget[Image, str](
            widget_ops=LinePlotWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                step_mapper=self.step_mapper,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.2,
            dedupe=True,
        )

        widgets["click_widget"] = Widget[None, Location3D](
            widget_ops=ClickWidgetOps(
                plotter=self.plotter, cam_dict=deepcopy(self.cam_dict)
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        return widgets


@register(
    "interactive_hypothesis_space_pointcloud",
    description="Pointcloud of hypothesis space and lineplot of sampling bursts",
)
def main(
    experiment_log_dir: str, objects_mesh_dir: str, pretrained_models_file: str
) -> int:
    """Interactive visualization for inspecting the primary target hypothesis space.

    This plot also allows for inspecting the sampling bursts and their slope triggers in
    a line plot.

    Args:
        experiment_log_dir: Path to the experiment directory containing the detailed
            stats file.
        objects_mesh_dir: Path to the root directory of YCB object meshes.
        pretrained_models_file: Path to the pretrained models pt file.

    Returns:
        Exit code.
    """
    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    data_path = str(Path(objects_mesh_dir).expanduser())
    models_path = str(Path(pretrained_models_file).expanduser())

    InteractivePlot(experiment_log_dir, data_path, models_path)

    return 0


@attach_args("interactive_hypothesis_space_pointcloud")
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

    p.add_argument(
        "--pretrained_models_file",
        default="~/tbp/results/monty/pretrained_models/pretrained_ycb_v11/surf_agent_1lm_10distinctobj/pretrained/model.pt",
        help=("The file containing the pretrained models."),
    )
