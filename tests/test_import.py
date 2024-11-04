"""Test autobook."""

import autobook


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(autobook.__name__, str)
