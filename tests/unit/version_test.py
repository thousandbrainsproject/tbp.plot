# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tbp.project import __version__


class VersionTest(unittest.TestCase):
    def test_version(self) -> None:
        assert __version__ == "0.0.0"
