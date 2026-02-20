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

import numpy as np

from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
    HierarchyStepMapper,
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


class FakeHierarchyDataParser(DataParser):
    """Parser with injected hierarchical LM data for HierarchyStepMapper tests."""

    def __init__(self):
        self.data = {
            "0": {
                "LM_0": {
                    "lm_processed_steps": [1, 0, 0, 1, 1, 0, 1, 1],
                },
                "LM_1": {
                    "lm_processed_steps": [0, 0, 0, 0, 1, 0, 1, 1],
                },
                "motor_system": {},
            },
        }


class TestHierarchyStepMapper(unittest.TestCase):
    """Tests for HierarchyStepMapper class."""

    def setUp(self) -> None:
        self.parser = FakeHierarchyDataParser()
        self.mapper = HierarchyStepMapper(self.parser, episode="0")

    def test_available_levels(self) -> None:
        """Verify agent and all LM levels are returned, sorted."""
        levels = self.mapper.available_levels()
        self.assertEqual(levels, ["agent", "LM_0", "LM_1"])

    def test_num_steps_agent(self) -> None:
        """Agent level should have 8 steps (length of mask)."""
        self.assertEqual(self.mapper.num_steps("agent"), 8)

    def test_num_steps_lm0(self) -> None:
        """LM_0 has 5 processed steps (5 ones in mask)."""
        self.assertEqual(self.mapper.num_steps("LM_0"), 5)

    def test_num_steps_lm1(self) -> None:
        """LM_1 has 3 processed steps (3 ones in mask)."""
        self.assertEqual(self.mapper.num_steps("LM_1"), 3)

    def test_num_steps_unknown_level_raises(self) -> None:
        """Unknown level should raise ValueError."""
        with self.assertRaises(ValueError):
            self.mapper.num_steps("LM_2")

    def test_lm0_to_agent(self) -> None:
        """Test LM_0 -> agent conversions."""
        # LM_0 mask: [1, 0, 0, 1, 1, 0, 1, 1] -> agent indices: [0, 3, 4, 6, 7]
        self.assertEqual(self.mapper.convert(0, "LM_0", "agent"), 0)
        self.assertEqual(self.mapper.convert(1, "LM_0", "agent"), 3)
        self.assertEqual(self.mapper.convert(2, "LM_0", "agent"), 4)
        self.assertEqual(self.mapper.convert(3, "LM_0", "agent"), 6)
        self.assertEqual(self.mapper.convert(4, "LM_0", "agent"), 7)

    def test_lm1_to_agent(self) -> None:
        """Test LM_1 -> agent conversions."""
        # LM_1 mask: [0, 0, 0, 0, 1, 0, 1, 1] -> agent indices: [4, 6, 7]
        self.assertEqual(self.mapper.convert(0, "LM_1", "agent"), 4)
        self.assertEqual(self.mapper.convert(1, "LM_1", "agent"), 6)
        self.assertEqual(self.mapper.convert(2, "LM_1", "agent"), 7)

    def test_agent_to_lm0(self) -> None:
        """Test agent -> LM_0 conversions (with None cases)."""
        self.assertEqual(self.mapper.convert(0, "agent", "LM_0"), 0)
        self.assertIsNone(self.mapper.convert(1, "agent", "LM_0"))
        self.assertIsNone(self.mapper.convert(2, "agent", "LM_0"))
        self.assertEqual(self.mapper.convert(3, "agent", "LM_0"), 1)
        self.assertEqual(self.mapper.convert(4, "agent", "LM_0"), 2)
        self.assertIsNone(self.mapper.convert(5, "agent", "LM_0"))
        self.assertEqual(self.mapper.convert(6, "agent", "LM_0"), 3)
        self.assertEqual(self.mapper.convert(7, "agent", "LM_0"), 4)

    def test_agent_to_lm1(self) -> None:
        """Test agent -> LM_1 conversions (with None cases)."""
        self.assertIsNone(self.mapper.convert(0, "agent", "LM_1"))
        self.assertIsNone(self.mapper.convert(1, "agent", "LM_1"))
        self.assertIsNone(self.mapper.convert(2, "agent", "LM_1"))
        self.assertIsNone(self.mapper.convert(3, "agent", "LM_1"))
        self.assertEqual(self.mapper.convert(4, "agent", "LM_1"), 0)
        self.assertIsNone(self.mapper.convert(5, "agent", "LM_1"))
        self.assertEqual(self.mapper.convert(6, "agent", "LM_1"), 1)
        self.assertEqual(self.mapper.convert(7, "agent", "LM_1"), 2)

    def test_lm0_to_lm1(self) -> None:
        """Test LM_0 -> LM_1 cross-level conversions."""
        # LM_0 step 0 (agent 0) -> LM_1: None (agent 0 not in LM_1)
        self.assertIsNone(self.mapper.convert(0, "LM_0", "LM_1"))
        # LM_0 step 1 (agent 3) -> LM_1: None (agent 3 not in LM_1)
        self.assertIsNone(self.mapper.convert(1, "LM_0", "LM_1"))
        # LM_0 step 2 (agent 4) -> LM_1 step 0
        self.assertEqual(self.mapper.convert(2, "LM_0", "LM_1"), 0)
        # LM_0 step 3 (agent 6) -> LM_1 step 1
        self.assertEqual(self.mapper.convert(3, "LM_0", "LM_1"), 1)
        # LM_0 step 4 (agent 7) -> LM_1 step 2
        self.assertEqual(self.mapper.convert(4, "LM_0", "LM_1"), 2)

    def test_lm1_to_lm0(self) -> None:
        """Test LM_1 -> LM_0 cross-level conversions."""
        # LM_1 step 0 (agent 4) -> LM_0 step 2
        self.assertEqual(self.mapper.convert(0, "LM_1", "LM_0"), 2)
        # LM_1 step 1 (agent 6) -> LM_0 step 3
        self.assertEqual(self.mapper.convert(1, "LM_1", "LM_0"), 3)
        # LM_1 step 2 (agent 7) -> LM_0 step 4
        self.assertEqual(self.mapper.convert(2, "LM_1", "LM_0"), 4)

    def test_same_level_identity(self) -> None:
        """Converting same level returns the input step."""
        self.assertEqual(self.mapper.convert(3, "agent", "agent"), 3)
        self.assertEqual(self.mapper.convert(2, "LM_0", "LM_0"), 2)
        self.assertEqual(self.mapper.convert(1, "LM_1", "LM_1"), 1)

    def test_invalid_from_level_raises(self) -> None:
        """Unknown from_level should raise ValueError."""
        with self.assertRaises(ValueError):
            self.mapper.convert(0, "LM_2", "agent")

    def test_invalid_to_level_raises(self) -> None:
        """Unknown to_level should raise ValueError."""
        with self.assertRaises(ValueError):
            self.mapper.convert(0, "agent", "unknown")

    def test_step_out_of_range_negative_raises(self) -> None:
        """Negative step index should raise ValueError."""
        with self.assertRaises(ValueError):
            self.mapper.convert(-1, "agent", "LM_0")

    def test_step_out_of_range_too_large_raises(self) -> None:
        """Step index beyond range should raise ValueError."""
        with self.assertRaises(ValueError):
            self.mapper.convert(8, "agent", "LM_0")  # max is 7
        with self.assertRaises(ValueError):
            self.mapper.convert(5, "LM_0", "agent")  # max is 4
        with self.assertRaises(ValueError):
            self.mapper.convert(3, "LM_1", "agent")  # max is 2

    def test_get_agent_indices_lm0(self) -> None:
        """get_agent_indices for LM_0 returns correct array."""
        indices = self.mapper.get_agent_indices("LM_0")
        np.testing.assert_array_equal(indices, [0, 3, 4, 6, 7])

    def test_get_agent_indices_lm1(self) -> None:
        """get_agent_indices for LM_1 returns correct array."""
        indices = self.mapper.get_agent_indices("LM_1")
        np.testing.assert_array_equal(indices, [4, 6, 7])

    def test_get_agent_indices_returns_copy(self) -> None:
        """get_agent_indices should return a copy to prevent mutation."""
        indices = self.mapper.get_agent_indices("LM_0")
        indices[0] = 999

        # Original should be unchanged
        original = self.mapper.get_agent_indices("LM_0")
        self.assertEqual(original[0], 0)

    def test_get_agent_indices_agent_level_raises(self) -> None:
        """get_agent_indices for 'agent' level should raise ValueError."""
        with self.assertRaises(ValueError):
            self.mapper.get_agent_indices("agent")

    def test_get_agent_indices_unknown_level_raises(self) -> None:
        """get_agent_indices for unknown level should raise ValueError."""
        with self.assertRaises(ValueError):
            self.mapper.get_agent_indices("LM_2")

    def test_get_union_agent_indices(self) -> None:
        """get_union_agent_indices returns union of all LM agent steps."""
        # LM_0: [0, 3, 4, 6, 7], LM_1: [4, 6, 7]
        # Union: [0, 3, 4, 6, 7] (sorted unique)
        union = self.mapper.get_union_agent_indices()
        np.testing.assert_array_equal(union, [0, 3, 4, 6, 7])

    def tearDown(self) -> None:
        self.parser = None
        self.mapper = None


if __name__ == "__main__":
    unittest.main()
