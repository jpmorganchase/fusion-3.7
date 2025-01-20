import pytest
import datetime
import json
from datetime import datetime
from tempfile import TemporaryDirectory
import os
from fusion37.credentials import (
    ProxyType,
    AuthToken,
    FusionCredentials,
    find_cfg_file,
    fusion_url_to_auth_url,
)
from fusion37.credentials import CredentialError

@pytest.fixture
def sample_auth_token():
    return AuthToken(token="sample_token", expires_in_secs=3600)

@pytest.fixture
def sample_credentials():
    return FusionCredentials(
        client_id="test_client_id",
        client_secret="test_client_secret",
        resource="https://example.com/resource",
        auth_url="https://auth.example.com/token",
        proxies={"https": "https://proxy.example.com"},
        grant_type="client_credentials",
        headers={"Custom-Header": "HeaderValue"}
    )
@pytest.fixture
def temp_file():
    """Fixture that creates a temporary credentials file."""
    with TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "credentials.json")
        
        # JSON content to be written to the file
        json_content = {
            "grant_type": "client_credentials",
            "client_id": "my_client_id",
            "client_secret": "my_client_secret",
            "resource": "my_resource",
            "auth_url": "my_auth_url",
        }

        # Write the JSON content to the file
        with open(temp_file_path, 'w') as f:
            json.dump(json_content, f)

        yield temp_file_path  # Yield the path to the temporary file


# Test AuthToken

def test_auth_token_expiry(sample_auth_token):
    assert sample_auth_token.is_expirable() is True
    assert sample_auth_token.expires_in_secs() > 0
    assert sample_auth_token.as_bearer_header() == ("Authorization", "Bearer sample_token")

def test_auth_token_no_expiry():
    token = AuthToken(token="no_expiry_token")
    assert token.is_expirable() is False
    assert token.expires_in_secs() is None

def test_auth_token_getnewargs():
    token = AuthToken(token="test_token", expires_in_secs=3600)
    token_str, expiry = token.token, token.expiry
    assert token_str == token.token
    assert expiry == token.expiry

def test_auth_token_is_expirable():
    token_with_expiry = AuthToken("test_token", 3600)
    assert token_with_expiry.is_expirable() is True

    token_without_expiry = AuthToken("test_token", None)
    assert token_without_expiry.is_expirable() is False

def test_auth_token_expires_in_secs():
    token = AuthToken("test_token", 3600)
    assert token.expires_in_secs() <= 3600

    token_without_expiry = AuthToken("test_token", None)
    assert token_without_expiry.expires_in_secs() is None

def test_auth_token_from_token():
    token = AuthToken.from_token("test_token", 3600)
    assert token.token == "test_token"

    token_without_expiry = AuthToken.from_token("test_token", None)
    assert token_without_expiry.token == "test_token"
    assert token_without_expiry.expires_in_secs() is None

# Test FusionCredentials instantiation

def test_fusion_credentials_initialization(sample_credentials):
    assert sample_credentials.client_id == "test_client_id"
    assert sample_credentials.client_secret == "test_client_secret"
    assert sample_credentials.resource == "https://example.com/resource"
    assert sample_credentials.auth_url == "https://auth.example.com/token"
    assert sample_credentials.proxies == {"https": "https://proxy.example.com"}
    assert sample_credentials.headers == {"Custom-Header": "HeaderValue"}
    assert sample_credentials.grant_type == "client_credentials"

def test_fusion_credentials_creation():
    creds = FusionCredentials(
        client_id="client_id",
        client_secret="client_secret",
        username="username",
        password="password",
        resource="resource",
        auth_url="auth_url",
        proxies=None,
        grant_type="grant_type",
        fusion_e2e="fusion_e2e",
        kid="kid",
        private_key="private_key"
    )

    assert creds.client_id == "client_id"
    assert creds.client_secret == "client_secret"
    assert creds.username == "username"
    assert creds.password == "password"
    assert creds.resource == "resource"
    assert creds.auth_url == "auth_url"
    assert creds.grant_type == "grant_type"
    assert creds.fusion_e2e == "fusion_e2e"

def test_fusion_credentials_creation_from_client_id():
    creds = FusionCredentials.from_client_id(
        client_id="client_id",
        client_secret="client_secret",
        resource="resource",
        auth_url="auth_url",
        proxies=None,
        fusion_e2e=None,
        kid=None,
        private_key=None
    )

    assert creds.client_id == "client_id"
    assert creds.client_secret == "client_secret"
    assert creds.resource == "resource"
    assert creds.auth_url == "auth_url"
    assert creds.grant_type == "client_credentials"

def test_fusion_credentials_creation_from_user_id():
    creds = FusionCredentials.from_user_id(
        client_id="client_id",
        username="username",
        password="password",
        resource="resource",
        auth_url="auth_url",
        proxies=None,
        fusion_e2e=None,
        kid=None,
        private_key=None
    )

    assert creds.username == "username"
    assert creds.password == "password"
    assert creds.resource == "resource"
    assert creds.auth_url == "auth_url"
    assert creds.grant_type == "password"

def test_fusion_credentials_from_bearer_token():
    expiry_date = datetime(2023, 11, 1)

    creds = FusionCredentials.from_bearer_token(
        bearer_token="token",
        bearer_token_expiry=expiry_date,
        proxies=None,
        fusion_e2e=None,
        headers=None
    )

    assert creds.resource is None
    assert creds.auth_url == "https://authe.jpmorgan.com/as/token.oauth2"
    assert creds.bearer_token.token == "token"
    assert creds.bearer_token.expires_in_secs() is not None

    new_token = "new_token"
    new_expiry_secs = 100
    creds.put_bearer_token(new_token, new_expiry_secs)

    assert creds.bearer_token.token == "new_token"
    assert creds.bearer_token.expires_in_secs() is not None

def test_put_bearer_token(sample_credentials):
    sample_credentials.put_bearer_token("new_token", expires_in_secs=1800)
    assert sample_credentials.bearer_token.token == "new_token"
    assert sample_credentials.bearer_token.expires_in_secs() > 0

def test_fusion_credentials_put_fusion_token():
    creds = FusionCredentials()
    token_key = "key"
    token_value = "token"
    expiry_secs = 3600  # 1 hour expiration
    creds.put_fusion_token(token_key, token_value, expiry_secs)

    
    assert token_key in creds.fusion_token, f"Fusion token key '{token_key}' not found."
    stored_token = creds.fusion_token[token_key]
    assert stored_token.token == token_value, f"Stored token does not match expected value. Got: {stored_token.token}, Expected: {token_value}"
    
    expiration_time = stored_token.expires_in_secs()
    assert expiration_time is not None, "Expiration time should not be None."
    assert expiration_time == expiry_secs, f"Expected expiration time of {expiry_secs}, but got {expiration_time}."

def test_fusion_credentials_from_file(temp_file):
    creds = FusionCredentials.from_file(temp_file)

    assert creds.client_id == "my_client_id", f"Expected 'my_client_id', but got {creds.client_id}"
    assert creds.client_secret == "my_client_secret", f"Expected 'my_client_secret', but got {creds.client_secret}"
    assert creds.resource == "my_resource", f"Expected 'my_resource', but got {creds.resource}"
    assert creds.auth_url == "my_auth_url", f"Expected 'my_auth_url', but got {creds.auth_url}"
    assert creds.grant_type == "client_credentials", f"Expected 'client_credentials', but got {creds.grant_type}"

def test_fusion_credentials_from_file_with_env_vars():
    os.environ["FUSION_CLIENT_ID"] = "env_client_id"
    os.environ["FUSION_CLIENT_SECRET"] = "env_client_secret"

    try:
        with TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "credentials.json")

            json_content = {
                "grant_type": "client_credentials",
                "resource": "my_resource",
                "auth_url": "my_auth_url",
            }

            with open(temp_file_path, "w") as f:
                json.dump(json_content, f)

            creds = FusionCredentials.from_file(temp_file_path)

            assert creds.client_id == "env_client_id", f"Expected 'env_client_id', but got {creds.client_id}"
            assert creds.client_secret == "env_client_secret", f"Expected 'env_client_secret', but got {creds.client_secret}"
            assert creds.resource == "my_resource", f"Expected 'my_resource', but got {creds.resource}"
            assert creds.auth_url == "my_auth_url", f"Expected 'my_auth_url', but got {creds.auth_url}"
            assert creds.grant_type == "client_credentials", f"Expected 'client_credentials', but got {creds.grant_type}"

    finally:
        # Cleanup environment variables
        os.environ.pop("FUSION_CLIENT_ID", None)
        os.environ.pop("FUSION_CLIENT_SECRET", None)

def test_proxy_type_from_str():
    assert ProxyType.from_str("http") == ProxyType.HTTP
    assert ProxyType.from_str("https") == ProxyType.HTTPS
    with pytest.raises(ValueError):
        ProxyType.from_str("invalid")

def test_find_cfg_file_found(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "client_credentials.json"
    cfg_file.write_text("{}")

    found_path = find_cfg_file(str(cfg_file))
    assert found_path == str(cfg_file)

def test_find_cfg_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_cfg_file(str(tmp_path / "missing.json"))

# Test fusion_url_to_auth_url

def test_fusion_url_to_auth_url_valid():
    url = "https://example.com/distributions/catalogs/test_catalog/datasets/test_dataset"
    result = fusion_url_to_auth_url(url)
    assert result == ("https://example.com/distributions/catalogs/test_catalog/datasets/test_dataset/authorize/token", "test_catalog", "test_dataset")

def test_fusion_url_to_auth_url_invalid_url():
    url = "not a valid url"
    with pytest.raises(CredentialError, match="Could not parse URL: not a valid url"):
        fusion_url_to_auth_url(url)

def test_fusion_url_to_auth_url_no_distribution():
    url = "https://example.com/catalogs/test_catalog"
    assert fusion_url_to_auth_url(url) is None

# Test FusionCredentials error handling

def test_missing_bearer_token():
    with pytest.raises(ValueError):
        FusionCredentials.from_bearer_token()

def test_fusion_url_to_auth_url_missing_catalogs_segment():
    url = "http://example.com/datasets/my_dataset/distributions"
    with pytest.raises(CredentialError):
        fusion_url_to_auth_url(url)