"""Test fusion types.py."""

from fusion37 import fusion_types


def test_enum_class() -> None:
    """Test Enum class"""
    types_len = 12
    assert len(list(fusion_types.Types)) == types_len


def test_datetime_timestamp_aliasing() -> None:
    """Test datetime timestamp aliasing"""
    datetime_type = fusion_types.Types.Datetime
    assert datetime_type.name == "Timestamp"
