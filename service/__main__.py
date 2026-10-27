"""``python -m service``: one uvicorn process (D79), JSON logs, graceful shutdown on SIGTERM."""

import logging
import sys

import uvicorn

from service.app import create_app
from service.config import ConfigError, Settings
from service.logs import configure_logging, log


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    try:
        app = create_app(settings)
    except ConfigError as exc:
        log(logging.getLogger("litsense.service"), logging.ERROR, str(exc))
        sys.exit(2)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        timeout_graceful_shutdown=settings.graceful_shutdown_s,
        server_header=False,
    )


if __name__ == "__main__":
    main()
