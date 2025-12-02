# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import unittest

from tbp.interactive.colors import Palette, hex_to_rgb


class HexToRgbTests(unittest.TestCase):
    def test_rgb_without_alpha(self) -> None:
        self.assertEqual(hex_to_rgb("#ffffff"), (1.0, 1.0, 1.0))
        self.assertEqual(hex_to_rgb("#000000"), (0.0, 0.0, 0.0))
        self.assertEqual(hex_to_rgb("#ff0000"), (1.0, 0.0, 0.0))

    def test_rgb_without_hash_prefix(self) -> None:
        self.assertEqual(hex_to_rgb("00ff00"), (0.0, 1.0, 0.0))

    def test_rgb_with_alpha(self) -> None:
        result = hex_to_rgb("#336699", alpha=0.5)
        self.assertEqual(result, (0.2, 0.4, 0.6, 0.5))

    def test_invalid_hex_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            hex_to_rgb("#12345")


class PaletteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_color = "numenta_blue"

    def test_as_hex_returns_string(self) -> None:
        """Tests if the returned hex is valid hex code."""
        hex_value = Palette.as_hex(self.valid_color)
        self.assertIsInstance(hex_value, str)
        self.assertRegex(hex_value, r"#?[0-9A-Fa-f]{6}")

    def test_as_rgb_returns_rgb_tuple(self) -> None:
        rgb = Palette.as_rgb(self.valid_color)
        self.assertIsInstance(rgb, tuple)
        self.assertEqual(len(rgb), 3)
        for channel in rgb:
            self.assertGreaterEqual(channel, 0.0)
            self.assertLessEqual(channel, 1.0)

    def test_as_rgb_with_alpha_returns_rgba_tuple(self) -> None:
        rgba = Palette.as_rgb(self.valid_color, alpha=0.8)
        self.assertIsInstance(rgba, tuple)
        self.assertEqual(len(rgba), 4)
        r, g, b, a = rgba
        for channel in (r, g, b):
            self.assertGreaterEqual(channel, 0.0)
            self.assertLessEqual(channel, 1.0)
        self.assertAlmostEqual(a, 0.8)

    def test_as_hex_unknown_color_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            Palette.as_hex("banana_yellow")

    def test_as_rgb_unknown_color_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            Palette.as_rgb("not_a_real_color")

        msg = str(ctx.exception)
        self.assertIn("not_a_real_color", msg)
        self.assertIn("Available colors:", msg)

    def tearDown(self) -> None:
        self.valid_color = None


if __name__ == "__main__":
    unittest.main()
