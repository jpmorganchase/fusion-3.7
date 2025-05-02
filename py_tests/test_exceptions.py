from typing import Type

import pytest

from fusion.exceptions import (
    APIConnectError,
    APIRequestError,
    APIResponseError,
    CredentialError,
    FileFormatError,
    UnrecognizedFormatError,
)


def test_api_response_error_with_message_and_status() -> None:
    original = ValueError("something went wrong")
    exc = APIResponseError(original, message="Bad Request", status_code=400)
    assert isinstance(exc, APIResponseError)
    assert "Bad Request" in str(exc)
    assert "400" in str(exc)
    assert "something went wrong" in str(exc)
    assert exc.original_exception == original
    assert exc.status_code == 400


def test_api_response_error_without_message() -> None:
    original = KeyError("missing key")
    exc = APIResponseError(original, status_code=500)
    assert "missing key" in str(exc)
    assert "500" in str(exc)


def test_api_request_error_with_message() -> None:
    original = ConnectionError("timeout")
    exc = APIRequestError(original, message="Unable to reach host", status_code=408)
    assert "Unable to reach host" in str(exc)
    assert "timeout" in str(exc)
    assert "408" in str(exc)
    assert exc.original_exception == original


def test_api_request_error_without_message() -> None:
    original = RuntimeError("generic failure")
    exc = APIRequestError(original, status_code=502)
    assert "generic failure" in str(exc)
    assert "502" in str(exc)


@pytest.mark.parametrize("exc_cls", [
    APIConnectError,
    UnrecognizedFormatError,
    CredentialError,
    FileFormatError,
])
def test_simple_exceptions(exc_cls: Type[Exception]) -> None:
    with pytest.raises(exc_cls) as e:
        raise exc_cls("test error")
    assert "test error" in str(e.value)
