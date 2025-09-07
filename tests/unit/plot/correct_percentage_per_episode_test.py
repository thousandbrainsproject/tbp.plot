# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tbp.plot.plots.correct_percentage_per_episode import main as plot


class TestCorrectPercentagePerEpisode(unittest.TestCase):
    def test_exit_1_if_exp_path_does_not_exist(self):
        exit_code = plot("nonexistent_path", "LM_0", "banana")
        self.assertEqual(exit_code, 1)
