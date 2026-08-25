"""Tests for the package version export."""

from importlib.metadata import version
import unittest

from ampav.bertopic import __version__ as bertopic_version
from ampav.keybert import __version__ as keybert_version


class PackageVersionTest(unittest.TestCase):
    """Verify the public version matches installed package metadata."""

    def test_public_version_matches_distribution(self) -> None:
        self.assertEqual(keybert_version, version("ampav-keybertopic"))

    def test_sibling_package_reexports_canonical_version(self) -> None:
        self.assertEqual(bertopic_version, keybert_version)


if __name__ == "__main__":
    unittest.main()
