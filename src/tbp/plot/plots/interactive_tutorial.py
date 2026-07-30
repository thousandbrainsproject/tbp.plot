# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import logging
from copy import deepcopy
from pathlib import Path

from pubsub.core import Publisher
from tbp.interactive.data import (
    DataParser,
    YCBMeshLoader,
    DataLocator, DataLocatorStep
)

from tbp.plot.registry import attach_args, register

from vedo import Plotter, Slider2D

from tbp.interactive.widgets import (
    VtkDebounceScheduler,
    Widget,
    extract_slider_state, set_slider_state
)
from tbp.interactive.widget_updaters import WidgetUpdater

from collections.abc import Callable, Iterable

from tbp.interactive.topics import TopicMessage, TopicSpec

logger = logging.getLogger(__name__)

class EpisodeSliderWidgetOps:
    """WidgetOps implementation for an Episode slider.

    This class sets the slider's range based on the number of
    available episodes and publishes changes as `TopicMessage` items
    under the "episode_number" topic.
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

        self._locators = self.create_locators()

    def create_locators(self) -> dict[str, DataLocator]:
        locators = {}
        locators["episode"] = DataLocator(
            path=[DataLocatorStep.key(name="episode")],
        )
        return locators

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
        return [TopicMessage(name="episode_number", value=state)]

class StepSliderWidgetOps:
    """WidgetOps implementation for a Step slider.

    This class listens for the current episode selection and adjusts the step
    slider range to match the number of steps in that episode. It publishes
    changes as `TopicMessage` items under the "step_number" topic.
    """

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser

        # NEW: subscriptions for this widget
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
                    name="telemetry", value="time"
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
        return [TopicMessage(name="step_number", value=state)]

    def update_slider_range(
        self, widget: Slider2D, msgs: list[TopicMessage]
    ) -> tuple[Slider2D, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}

        widget.range = [
            0,
            len(
                self.data_parser.query(
                    self._locators["step"], episode=str(msgs_dict["episode_number"])
                )
            )
            - 1,
        ]

        self.set_state(widget, 0)

        return widget, True


class InteractivePlot:
    """An interactive plot for a simple tutorial."""

    def __init__(
        self,
        exp_path: str,
        data_path: str,
    ):
        renderer_areas = [
            {"bottomleft": (0.0, 0.0), "topright": (1.0, 1.0)},
            {"bottomleft": (0.1, 0.3), "topright": (0.45, 0.8)},
            {"bottomleft": (0.55, 0.3), "topright": (0.9, 0.8)},
        ]

        self.axes_dict = {
            "xrange": (-0.05, 0.05),
            "yrange": (1.45, 1.55),
            "zrange": (-0.05, 0.05),
        }
        self.cam_dict = {"pos": (0, 0, 1), "focal_point": (0, 0, 0)}

        self.data_parser = DataParser(exp_path)
        self.ycb_loader = YCBMeshLoader(data_path)
        self.event_bus = Publisher()
        self.plotter = Plotter(shape=renderer_areas, sharecam=False).render()
        self.scheduler = VtkDebounceScheduler(self.plotter.interactor, period_ms=33)

        # NEW: Create and add widgets
        self._widgets = self.create_widgets()
        for w in self._widgets.values():
            w.add()

        # NEW: Set an initial value for the episode slider
        self._widgets["episode_slider"].set_state(0)

        # Show the plot renderers
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
            interactive=True,
            resetcam=True,
        )

    def create_widgets(self):
        widgets = {}

        widgets["episode_slider"] = Widget[Slider2D, int](
            widget_ops=EpisodeSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            scopes = [],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.0,
            dedupe=False,
        )

        widgets["step_slider"] = Widget[Slider2D, int](
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            scopes = [],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.3,
            dedupe=True,
        )

        return widgets

@register(
    "interactive_tutorial",
    description="Simple tutorial with four widgets",
)
def main(experiment_log_dir: str, objects_mesh_dir: str) -> int:
    """Interactive tutorial visualization.

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


@attach_args("interactive_tutorial")
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