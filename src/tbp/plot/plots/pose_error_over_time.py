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

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tbp.plot.plots.stats import deserialize_json_chunks
from tbp.plot.registry import attach_args, register

if TYPE_CHECKING:
    import argparse

sns.set_theme(style="darkgrid")
logger = logging.getLogger(__name__)


@register(
    "pose_error_over_time", description="Pose error and theoretical limit over time"
)
def main(experiment_log_dir: str) -> int:
    """Plot MLH pose error and theoretical limits over time.

    This function visualizes the theoretical pose error limit vs. the actual
    pose error over time, along with a correctness indicator of whether the
    predicted object (`mlo`) matches the target object. It generates a two-row
    plot where the top subplot shows binary correctness (Correct/Wrong) of MLO
    and the bottom subplot shows pose error metrics.
    The theoretical limit is calculated by finding the minimum pose error over
    all existing hypotheses in Monty's hypothesis space. This metric conveys the
    best possible performance if Monty selects the best hypothesis as its most
    likely hypothesis (MLH).

    Args:
        experiment_log_dir: Path to the experiment directory containing detailed stats
            data.

    Returns:
        Exit code.
    """
    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    # load detailed stats
    json_file = Path(experiment_log_dir) / "detailed_run_stats.json"
    detailed_stats = deserialize_json_chunks(json_file)

    dfs = []
    for ep_data in detailed_stats.values():
        steps = len(ep_data["LM_0"]["target_object_theoretical_limit"])
        target = [ep_data["LM_0"]["target"]["object"] for _ in range(steps)]
        th_limit = ep_data["LM_0"]["target_object_theoretical_limit"]
        mlo = [ep_data["LM_0"]["current_mlh"][i]["graph_id"] for i in range(steps)]
        obj_error = ep_data["LM_0"]["target_object_pose_error"]

        dfs.append(
            pd.DataFrame(
                {
                    "target": target,
                    "th_limit": th_limit,
                    "mlo": mlo,
                    "obj_error": obj_error,
                }
            )
        )

    # Combine all episodes into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Create stacked subplots: (MLO accuracy, pose error)
    _, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [0.5, 4]}
    )

    mlo_correct = [
        1 if mlo == target else 0
        for mlo, target in zip(df["mlo"], df["target"], strict=False)
    ]
    colors = ["green" if c else "red" for c in mlo_correct]

    ax0.scatter(
        np.array(df.index), np.array(mlo_correct), c=colors, marker="o", s=30, alpha=0.8
    )
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(["wrong", "correct"])
    ax0.set_ylabel("MLO", fontsize=14)
    ax0.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax0.set_ylim(-0.5, 1.5)
    ax0.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax1.plot(
        np.array(df.index),
        np.array(df["th_limit"]),
        color="black",
        linestyle="-",
        linewidth=2,
        label="Theoretical limit",
    )
    ax1.scatter(
        np.array(df.index),
        np.array(df["obj_error"]),
        c="gray",
        marker="o",
        s=50,
        alpha=0.75,
        label="MLH of target object",
    )

    # Labels and formatting
    ax1.set_xlabel("Steps", fontsize=14)
    ax1.set_ylabel("Pose Error", fontsize=14)
    ax0.set_title(
        "MLH Pose Error vs. Theoretical Limit", fontsize=14, fontweight="bold"
    )

    # Create Legend
    black_line = mlines.Line2D(
        [], [], color="black", linewidth=2, label="Theoretical limit"
    )
    gray_dots = mlines.Line2D(
        [],
        [],
        color="gray",
        marker="o",
        linestyle="None",
        markersize=8,
        label="MLH of target object",
    )
    green_diamond = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=8,
        label="Correct MLO",
    )
    red_diamond = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        markersize=8,
        label="Wrong MLO",
    )

    ax1.legend(
        handles=[black_line, gray_dots, green_diamond, red_diamond],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()

    return 0


@attach_args("pose_error_over_time")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
