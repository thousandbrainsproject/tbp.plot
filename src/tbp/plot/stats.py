# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Load experiment statistics from an experiment for analysis.

NOTE: Copied from tbp.monty.frameworks.utils.logging_utils.py.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def load_stats(
    exp_path,
    load_train=True,
    load_eval=True,
    load_detailed=True,
    load_models=True,
    pretrained_dict=None,
):
    """Load experiment statistics from an experiment for analysis.

    Returns:
        train_stats: pandas DataFrame with training statistics
        eval_stats: pandas DataFrame with evaluation statistics
        detailed_stats: dict with detailed statistics
        lm_models: dict with loaded language models
    """
    train_stats, eval_stats, detailed_stats, lm_models = None, None, None, None
    if load_train:
        print("...loading and checking train statistics...")
        train_stats = pd.read_csv(os.path.join(exp_path, "train_stats.csv"))

    if load_eval:
        print("...loading and checking eval statistics...")
        eval_stats = pd.read_csv(os.path.join(exp_path, "eval_stats.csv"))

    if load_detailed:
        print("...loading detailed run statistics...")
        json_file = os.path.join(exp_path, "detailed_run_stats.json")
        try:
            with open(json_file, "r") as f:
                detailed_stats = json.load(f)
        except ValueError:
            detailed_stats = deserialize_json_chunks(json_file)
        f.close()

    if load_models:
        print("...loading LM models...")
        lm_models = load_models_from_dir(exp_path, pretrained_dict)

    return train_stats, eval_stats, detailed_stats, lm_models


def load_models_from_dir(exp_path, pretrained_dict=None):
    lm_models = {}

    if pretrained_dict is not None:
        lm_models["pretrained"] = {}
        state_dict = torch.load(os.path.join(pretrained_dict, "model.pt"))
        for lm_id in list(state_dict["lm_dict"].keys()):
            pretrained_models = state_dict["lm_dict"][lm_id]["graph_memory"]
            lm_models["pretrained"][lm_id] = pretrained_models

    for folder in os.listdir(exp_path):
        if folder.isnumeric():
            state_dict = torch.load(os.path.join(exp_path, folder, "model.pt"))
            for lm_id in list(state_dict["lm_dict"].keys()):
                epoch_models = state_dict["lm_dict"][lm_id]["graph_memory"]
                if folder not in lm_models.keys():
                    lm_models[folder] = {}
                lm_models[folder]["LM_" + str(lm_id)] = epoch_models
    return lm_models


def deserialize_json_chunks(json_file, start=0, stop=None, episodes=None):
    """Deserialize one episode at a time from json file.

    Only get episodes specified by arguments, which follow list / numpy like semantics.

    Note:
        assumes line counter is exactly in line with episode keys

    Args:
        json_file: full path to the json file to load
        start: int, get data starting at this episode
        stop: int, get data ending at this episode, not inclussive as usual in python
        episodes: iterable of ints with episodes to pull

    Returns:
        detailed_json: dict containing contents of file_handle
    """

    def should_get_episode(start, stop, episodes, counter):
        if episodes is not None:
            return counter in episodes
        else:
            return (counter >= start) and (counter < stop)

    detailed_json = {}
    stop = stop or np.inf
    with open(json_file, "r") as f:
        for line_counter, line in enumerate(f):
            if should_get_episode(start, stop, episodes, line_counter):
                # NOTE: json logging is only used at inference time and inference
                # episodes are independent and order does not matter. This hack fixes a
                # problem introduced from running in parallel: every episode had the
                # key 0 since it was its own experiment, so we update detailed_json with
                # line counter key instead of tmp_json key. This works for serial
                # episodes because order of execution is arbitrary, all that matters is
                # we know the parameters for that episode.
                tmp_json = json.loads(line)
                json_key = list(tmp_json.keys())[0]  # has only one key
                detailed_json[str(line_counter)] = tmp_json[json_key]
                del tmp_json

    if episodes is not None:
        str_episodes = [str(i) for i in episodes]
        if list(detailed_json.keys()) != str_episodes:
            print(
                "WARNING: episode keys did not equal json keys. This can happen if "
                "json file was not appended to in episode order. To manually load the"
                "whole file for debugging, run `deserialize_json_chunks(my_file)` with"
                "no further arguments"
            )
    return detailed_json
