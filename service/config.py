"""Service settings from the environment (pydantic-settings, D76). Field names are the env vars
(case-insensitive): ``bundle_url`` is ``BUNDLE_URL`` and so on. ``service/.env.example`` lists them.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(ValueError):
    """The settings cannot run safely (for example no API key and auth not explicitly off)."""


class Settings(BaseSettings):
    """Every knob of the service. ``GEMINI_API_KEY`` is deliberately absent: the core reads it from
    the environment itself, so the key never sits in a settings object, a repr or a log line."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    # Bundle and boot
    bundle_url: str | None = None
    bundle_dir: str = "/var/litsense"
    boot_retries: int = Field(default=3, ge=1)
    boot_backoff_s: float = Field(default=5.0, ge=0)
    warmup: bool = True
    onnx_threads: int | None = Field(default=None, ge=1, le=16)
    # Auth and HTTP
    service_api_key: SecretStr | None = None
    allow_no_auth: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    forwarded_allow_ips: str = "*"
    cors_origins: str = ""
    max_body_bytes: int = Field(default=1_000_000, ge=1024)
    graceful_shutdown_s: int = Field(default=20, ge=0)
    log_level: str = "INFO"
    # Concurrency, limits, caches
    core_threads: int = Field(default=4, ge=1, le=64)
    recommend_concurrency: int = Field(default=2, ge=1, le=64)
    queue_timeout_s: float = Field(default=10.0, gt=0)
    rate_limit_rps: float = Field(default=20.0, ge=0)
    rate_limit_burst: int = Field(default=40, ge=1)
    cache_ttl_s: float = Field(default=300.0, ge=0)
    cache_max_entries: int = Field(default=1024, ge=0)
    # LLM
    llm_timeout_s: float = Field(default=12.0, gt=0)
    llm_cache_path: str | None = None
    llm_provider: str | None = None

    @property
    def api_key(self) -> str | None:
        """The shared secret, or None when auth is off."""
        if self.service_api_key is None:
            return None
        value = self.service_api_key.get_secret_value().strip()
        return value or None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def check(self) -> None:
        """Refuse to run without a key unless auth was switched off on purpose."""
        if self.api_key is None and not self.allow_no_auth:
            raise ConfigError(
                "SERVICE_API_KEY is not set. Set it (the Node backend sends it as X-API-Key) or "
                "set ALLOW_NO_AUTH=true for local development only."
            )
