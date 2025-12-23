# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tbp.plot.plots.interactive_hypothesis_space_pointcloud import (
    main as interactive_plot,
)


class TestInteractiveHypothesisSpacePointcloud(unittest.TestCase):
    def test_exit_1_if_exp_path_does_not_exist(self):
        exit_code = interactive_plot(
            experiment_log_dir="nonexistent_exp_path",
            objects_mesh_dir="nonexistent_data_path",
            pretrained_models_file="nonexistent_data_path",
        )
        self.assertEqual(exit_code, 1)
