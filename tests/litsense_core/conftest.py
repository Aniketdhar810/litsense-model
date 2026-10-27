"""Fixtures for the bundle tests: the tiny fixture over a local HTTP server, and every pipeline
stage registered (the bundle test runs the pipeline end to end)."""

from ml.stages import load_stages
from tests.ml.data.conftest import file_server, tiny_cfg  # noqa: F401

load_stages()
