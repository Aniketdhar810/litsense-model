"""data.download keeps the 5-core CSV byte for byte and refuses data that breaks its contract."""

import gzip
from pathlib import Path
from typing import Any

import pytest

import ml.data.download  # noqa: F401  (registers the stage)
from ml.common.errors import DataError
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_stage
from tests.ml.data.conftest import FileServer


def manifest(cfg: dict[str, Any], stage: str, sample: int | None = None) -> dict[str, Any]:
    paths = RunPaths.from_config(cfg, sample)
    result = read_manifest(paths.manifest(stage))
    assert result is not None
    return result


def test_copies_the_file_and_records_counts(tiny_cfg: dict[str, Any], tiny_dataset: Path) -> None:
    run_stage("data.download", tiny_cfg)
    out = RunPaths.from_config(tiny_cfg, None).resolve("data:ratings_5core.csv.gz")
    assert out.read_bytes() == (tiny_dataset / "raw" / "Books.csv.gz").read_bytes()
    m = manifest(tiny_cfg, "data.download")
    assert m["notes"]["counts"] == {"ratings": 600, "users": 50, "items": 80}
    assert m["notes"]["source"]["complete"] and m["notes"]["source"]["reconnects"] == 0
    assert m["notes"]["time_range"][0].startswith("2019-")
    assert set(m["notes"]["stars"]) <= {"1.0", "2.0", "3.0", "4.0", "5.0"}


def test_sample_runs_still_fetch_the_whole_file(
    tiny_cfg: dict[str, Any], tiny_dataset: Path
) -> None:
    head = Path(tiny_cfg["project"]["local_raw_root"])
    head.mkdir(parents=True)
    with gzip.open(tiny_dataset / "raw" / "Books.csv.gz", "rt") as f:
        first = "".join(next(f) for _ in range(11))
    with gzip.open(head / "Books.csv.gz", "wt") as f:  # a local head must not be used
        f.write(first)
    run_stage("data.download", tiny_cfg, sample=5)
    assert manifest(tiny_cfg, "data.download", sample=5)["notes"]["counts"]["ratings"] == 600


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("user_id,item,rating,timestamp\nu,a,5.0,1546300800000\n", "header"),
        ("user_id,parent_asin,rating,timestamp\nu,a,5.0,1546300800\n", "not in milliseconds"),
        ("user_id,parent_asin,rating,timestamp\nu,a,6.0,1546300800000\n", "malformed"),
    ],
)
def test_rejects_a_broken_contract(
    tiny_cfg: dict[str, Any], file_server: FileServer, body: str, message: str
) -> None:
    with gzip.open(file_server.root / "Books.csv.gz", "wt") as f:
        f.write(body)
    with pytest.raises(DataError, match=message):
        run_stage("data.download", tiny_cfg)
    assert not RunPaths.from_config(tiny_cfg, None).manifest("data.download").exists()


def test_lid_model_is_kept(tiny_cfg: dict[str, Any], file_server: FileServer) -> None:
    run_stage("data.lid", tiny_cfg, sample=5)  # always the whole file, even in sample runs
    out = RunPaths.from_config(tiny_cfg, 5).resolve("data:lid.176.ftz")
    assert out.read_bytes() == (file_server.root / "lid.176.ftz").read_bytes()
    assert (
        manifest(tiny_cfg, "data.lid", sample=5)["notes"]["counts"]["bytes"] == out.stat().st_size
    )


def test_lid_rejects_a_non_fasttext_file(tiny_cfg: dict[str, Any], file_server: FileServer) -> None:
    (file_server.root / "lid.176.ftz").write_bytes(b"<html>not found</html>")
    with pytest.raises(DataError, match="not a fastText model"):
        run_stage("data.lid", tiny_cfg)
