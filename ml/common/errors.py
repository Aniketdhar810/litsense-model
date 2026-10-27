"""Exceptions raised by the offline pipeline."""


class ConfigError(ValueError):
    """ml/config.yaml is missing a key or holds an invalid value."""


class StageError(RuntimeError):
    """A stage is unknown, misdeclared, missing an input, or did not write its outputs."""


class DataError(ValueError):
    """Raw data broke an expectation the pipeline relies on (schema, units, integrity)."""


class StreamError(RuntimeError):
    """A streamed source could not be read completely (HTTP error, truncation, retries spent)."""
