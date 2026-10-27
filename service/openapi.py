"""Write the OpenAPI document the Node backend builds against: ``python -m service.openapi``.

``tests/service/test_openapi.py`` fails when ``service/openapi.json`` is stale.
"""

import json
from pathlib import Path
from typing import Any

from service.app import create_app
from service.boot import BootResult
from service.config import Settings

OUT = Path(__file__).with_name("openapi.json")


def _no_boot(settings: Settings, on_event: Any) -> BootResult:
    raise RuntimeError("the OpenAPI export never boots a model")


def build() -> dict[str, Any]:
    settings = Settings(allow_no_auth=True, service_api_key=None, cors_origins="")
    return create_app(settings, loader=_no_boot).openapi()


def render() -> str:
    return json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
