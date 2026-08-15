"""Tests for the package version export."""

from importlib.metadata import version
import unittest

from ampav.keybert import __version__


class PackageVersionTest(unittest.TestCase):
    """Verify the public version matches installed package metadata."""

    def test_public_version_matches_distribution(self) -> None:
        self.assertEqual(__version__, version("ampav-keybertopic"))


if __name__ == "__main__":
    unittest.main()
