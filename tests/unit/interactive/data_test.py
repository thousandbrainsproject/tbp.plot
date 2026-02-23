# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
    EpisodeStepMapper,
)


class FakeDataParser(DataParser):
    """Parser class that bypasses loading by injecting data directly."""

    def __init__(self):
        self.data = {
            "runs": [
                {
                    "episodes": [
                        {"steps": [{"value": "r0e0s0"}, {"value": "r0e0s1"}]},
                        {"steps": [{"value": "r0e1s0"}]},
                    ]
                },
                {
                    "episodes": [
                        {
                            "steps": [
                                {"value": "r1e0s0"},
                                {"value": "r1e0s1"},
                                {"value": "r1e0s2"},
                            ]
                        }
                    ]
                },
            ],
            "mlh": {"graph_id": "mug", "pose": [0, 0, 0]},
        }


class FakeRepeatedDataParser(DataParser):
    """Parser class that bypasses loading by injecting data directly.

    The data in this classes contains repeated keys in the nested dictionary.
    """

    def __init__(self):
        self.data = {
            "0": [  # < --- Key 0 is repeated
                {
                    "steps": {"0": "test1"},  # < --- Key 0 is repeated
                },
                {
                    "steps": {"0": "test2"},
                },
            ]
        }


def make_run_episode_step_locator(
    run_val: int | None = None,
    ep_val: int | None = None,
    step_val: int | None = None,
) -> DataLocator:
    """Returns a locator for: runs[run_idx].episodes[ep_idx].steps[step_idx].value."""
    return DataLocator(
        path=[
            DataLocatorStep.key("runs", "runs"),
            DataLocatorStep.index("run_idx", run_val),
            DataLocatorStep.key("episodes", "episodes"),
            DataLocatorStep.index("ep_idx", ep_val),
            DataLocatorStep.key("steps", "steps"),
            DataLocatorStep.index("step_idx", step_val),
            DataLocatorStep.key("value", "value"),
        ]
    )


class BaseWithSampleData(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = FakeDataParser()

    def test_extract_success_all_kwargs(self) -> None:
        loc = make_run_episode_step_locator()
        out = self.parser.extract(loc, run_idx=1, ep_idx=0, step_idx=2)
        self.assertEqual(out, "r1e0s2")

    def test_extract_success_mixed_fixed_and_kwargs(self) -> None:
        # Fix run=0 and ep=1 in the locator, leave step unresolved
        loc = make_run_episode_step_locator(run_val=0, ep_val=1)
        out = self.parser.extract(loc, step_idx=0)
        self.assertEqual(out, "r0e1s0")

    def test_extract_missing_required_step_raises(self) -> None:
        loc = make_run_episode_step_locator()
        with self.assertRaises(ValueError) as ctx:
            self.parser.extract(loc, run_idx=0, ep_idx=0)

    def test_extract_type_mismatch_raises(self) -> None:
        loc = make_run_episode_step_locator()
        cases = [
            {"run_idx": "0", "ep_idx": 0, "step_idx": 0},
            {"run_idx": 0, "ep_idx": "0", "step_idx": 0},
            {"run_idx": 0, "ep_idx": 0, "step_idx": "0"},
        ]
        for bad in cases:
            with self.assertRaises(ValueError) as ctx:
                self.parser.extract(loc, **bad)

    def test_query_returns_index_candidates(self) -> None:
        # first missing is run_idx (index)
        loc = make_run_episode_step_locator()
        candidates = self.parser.query(loc)
        self.assertEqual(candidates, [0, 1])

    def test_query_returns_key_candidates(self) -> None:
        loc = DataLocator(
            path=[
                DataLocatorStep.key("first", "mlh"),
                DataLocatorStep.key("second", None),
            ]
        )
        candidates = self.parser.query(loc)
        self.assertEqual(sorted(candidates), ["graph_id", "pose"])

    def test_query_raises_when_no_missing(self) -> None:
        loc = make_run_episode_step_locator(run_val=0, ep_val=0, step_val=1)
        with self.assertRaises(ValueError):
            self.parser.query(loc)

    def test_missing_steps_identifies_none_values(self) -> None:
        loc = DataLocator(
            path=[
                DataLocatorStep.key("runs", "runs"),
                DataLocatorStep.index("run_idx", None),
                DataLocatorStep.key("episodes", "episodes"),
                DataLocatorStep.index("ep_idx", None),
            ]
        )
        missing = loc.missing_steps()
        self.assertEqual([s.name for s in missing], ["run_idx", "ep_idx"])

    def test_repeating_key_in_nested_dictionary(self) -> None:
        parser = FakeRepeatedDataParser()
        loc = DataLocator(
            path=[
                DataLocatorStep.key("episode", "0"),  # <-- repeated
                DataLocatorStep.index("step", None),
                DataLocatorStep.key("type", "steps"),
                DataLocatorStep.index("time", "0"),  # <-- repeated
            ]
        )

        self.assertEqual(parser.extract(loc, step=0), "test1")
        self.assertEqual(parser.extract(loc, step=1), "test2")

    def test_extend_clones_and_appends(self) -> None:
        base = DataLocator(path=[DataLocatorStep.key("runs", "runs")])
        ext = base.extend([DataLocatorStep.index("run_idx", 1)])

        self.assertEqual(len(base.path), 1)
        self.assertEqual([s.name for s in ext.path], ["runs", "run_idx"])

        # deep copy: mutate ext and ensure base is unchanged
        ext.path[0].value = "changed"
        self.assertEqual(base.path[0].value, "runs")

    def test_classmethods(self) -> None:
        s1 = DataLocatorStep.key("episode", "0")
        s2 = DataLocatorStep.index("step", 3)

        self.assertEqual((s1.type, s1.name, s1.value), ("key", "episode", "0"))
        self.assertEqual((s2.type, s2.name, s2.value), ("index", "step", 3))

    def tearDown(self) -> None:
        self.parser = None
        self.sample_data = None


class FakeEpisodeStepMapperDataParser(DataParser):
    """Parser with data shaped for EpisodeStepMapper: 3 episodes with 3, 2, 5 steps."""

    def __init__(self) -> None:
        self.data = {
            "0": {
                "LM_0": {
                    "time": [0.0, 1.0, 2.0],
                },
            },
            "1": {
                "LM_0": {
                    "time": [0.0, 1.0],
                },
            },
            "2": {
                "LM_0": {
                    "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                },
            },
        }


class TestEpisodeStepMapper(unittest.TestCase):
    """Tests for the EpisodeStepMapper class.

    Episode layout:
        episode 0: 3 steps -> global [0, 1, 2]
        episode 1: 2 steps -> global [3, 4]
        episode 2: 5 steps -> global [5, 6, 7, 8, 9]
        total: 10 steps
    """

    def setUp(self) -> None:
        self.parser = FakeEpisodeStepMapperDataParser()
        self.mapper = EpisodeStepMapper(self.parser)

    def test_num_episodes(self) -> None:
        self.assertEqual(self.mapper.num_episodes, 3)

    def test_total_num_steps(self) -> None:
        self.assertEqual(self.mapper.total_num_steps, 10)

    def test_global_to_local_first_episode(self) -> None:
        self.assertEqual(self.mapper.global_to_local(0), (0, 0))
        self.assertEqual(self.mapper.global_to_local(1), (0, 1))
        self.assertEqual(self.mapper.global_to_local(2), (0, 2))

    def test_global_to_local_second_episode(self) -> None:
        self.assertEqual(self.mapper.global_to_local(3), (1, 0))
        self.assertEqual(self.mapper.global_to_local(4), (1, 1))

    def test_global_to_local_third_episode(self) -> None:
        self.assertEqual(self.mapper.global_to_local(5), (2, 0))
        self.assertEqual(self.mapper.global_to_local(9), (2, 4))

    def test_local_to_global(self) -> None:
        self.assertEqual(self.mapper.local_to_global(0, 0), 0)
        self.assertEqual(self.mapper.local_to_global(0, 2), 2)
        self.assertEqual(self.mapper.local_to_global(1, 0), 3)
        self.assertEqual(self.mapper.local_to_global(1, 1), 4)
        self.assertEqual(self.mapper.local_to_global(2, 0), 5)
        self.assertEqual(self.mapper.local_to_global(2, 4), 9)

    def test_roundtrip_global_to_local_to_global(self) -> None:
        for g in range(self.mapper.total_num_steps):
            ep, local = self.mapper.global_to_local(g)
            self.assertEqual(self.mapper.local_to_global(ep, local), g)

    def test_global_to_local_negative_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.mapper.global_to_local(-1)

    def test_global_to_local_out_of_range_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.mapper.global_to_local(10)

    def test_local_to_global_episode_out_of_range_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.mapper.local_to_global(3, 0)

    def test_local_to_global_negative_episode_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.mapper.local_to_global(-1, 0)

    def test_local_to_global_step_out_of_range_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.mapper.local_to_global(0, 3)

    def test_local_to_global_negative_step_raises(self) -> None:
        with self.assertRaises(IndexError):
            self.mapper.local_to_global(0, -1)


if __name__ == "__main__":
    unittest.main()
