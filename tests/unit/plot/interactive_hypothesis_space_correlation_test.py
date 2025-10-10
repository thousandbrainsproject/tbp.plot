# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tbp.plot.plots.interactive_hypothesis_space_correlation import (
    main as interactive_plot,
)


class TestInteractiveObjectsEvidenceOverTime(unittest.TestCase):
    def test_exit_1_if_exp_path_does_not_exist(self):
        exit_code = interactive_plot("nonexistent_exp_path", "nonexistent_data_path")
        self.assertEqual(exit_code, 1)
