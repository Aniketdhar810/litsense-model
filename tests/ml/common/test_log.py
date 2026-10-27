"""Progress lines for long streams: rate-limited, with size, share, speed and time left."""

import logging

import pytest

from ml.common.log import Progress, get_logger


def test_progress_is_rate_limited_and_informative(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    progress = Progress(get_logger("t.progress"), "meta_Books", every_s=3600)
    progress.update(10, 100)
    assert not caplog.records  # less than every_s since the start

    progress.every_s = 0
    progress.update(2_000_000_000, 4_940_000_000, "; 5 lines")
    message = caplog.records[-1].getMessage()
    assert message.startswith("meta_Books: 2.00/4.94 GB (40.5%) at ")
    assert "MB/s, about" in message and message.endswith("min left; 5 lines")

    progress.update(1_500_000_000, None)  # size unknown
    assert caplog.records[-1].getMessage().startswith("meta_Books: 1.50 GB at ")
