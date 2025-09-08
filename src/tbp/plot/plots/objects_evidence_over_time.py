# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import seaborn as sns
from matplotlib import patches, transforms
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from tbp.plot.plots.stats import deserialize_json_chunks
from tbp.plot.registry import attach_args, register

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


# NOTE: Copied from tbp.monty.frameworks.environments.ycb.py.
# 10 objects that have little similarities in morphology.
DISTINCT_OBJECTS = [
    "mug",
    "bowl",
    "potted_meat_can",
    "spoon",
    "strawberry",
    "mustard_bottle",
    "dice",
    "golf_ball",
    "c_lego_duplo",
    "banana",
]


@register(
    "objects_evidence_over_time",
    description="Maximum evidence score for each object over time",
)
def main(experiment_log_dir: str) -> int:
    """Plot evidence scores for each object over time.

    This function visualizes the evidence scores for each object. The plot is produced
    over a sequence of episodes, and overlays colored rectangles highlighting when a
    particular target object is active.

    Args:
        experiment_log_dir: Path to the experiment directory containing the detailed
            stats file.

    Returns:
        Exit code.
    """
    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    # seaborn darkgrid style
    sns.set_theme(style="darkgrid")

    # load detailed stats
    json_file = Path(experiment_log_dir) / "detailed_run_stats.json"
    detailed_stats = deserialize_json_chunks(json_file)

    # fix colors for distinct objects (tab10 supports up to 10 distinct colors)
    cmap = plt.cm.tab10
    num_colors = len(DISTINCT_OBJECTS)
    ycb_colors = {obj: cmap(i / num_colors) for i, obj in enumerate(DISTINCT_OBJECTS)}

    classes = {
        k: [] for k in list(detailed_stats["0"]["LM_0"]["max_evidence"][0].keys())
    }
    target_objects = []  # Objects in each segment, e.g., ['strawberry', 'banana']
    target_transitions = []  # Transition points on the x-axis, e.g., [49, 99]

    for episode_data in detailed_stats.values():
        evidences_data = episode_data["LM_0"]["max_evidence"]

        # append evidence data to classes
        for ts in evidences_data:
            for k, v in ts.items():
                classes[k].append(v)

        # collect the target object of this episode
        target_objects.append(episode_data["target"]["primary_target_object"])

        # collect target transition point
        target_transitions.append(len(evidences_data))

    # Create the plot
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot the lines
    for obj, scores in classes.items():
        ax.plot(
            scores,
            marker="o",
            linestyle="-",
            label=obj,
            color=ycb_colors.get(obj, "gray"),
        )

    # Add colored rectangles indicating the current target object
    box_height = 0.02
    prev_x = 0
    for obj, x in zip(target_objects, target_transitions, strict=False):
        rect = patches.Rectangle(
            (prev_x, 1 - box_height),
            (x - 1),
            box_height,
            transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
            edgecolor="black",
            facecolor=ycb_colors.get(obj, "gray"),
            lw=1,
            alpha=1.0,
            clip_on=True,
        )
        ax.add_patch(rect)
        prev_x += x - 1

    # Formatting
    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Evidence Scores", fontsize=14)
    ax.set_title(
        "Evidence Scores Over Time with Resampling",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(title="Objects", fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(MultipleLocator(10))

    # Show plot
    plt.show()

    return 0


@attach_args("objects_evidence_over_time")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
