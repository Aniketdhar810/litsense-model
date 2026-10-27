"""The catalogue the backend seeds its ``books`` table from (contract section 7 item 1, D76).

Built from the loaded store: ``CatalogBook`` rows (the contract's proposed schema) from the
catalogue arrays and strings. ``isbn13`` is not kept by the store, so it is read once from the
bundle's ``catalog_text.json.gz``; a bundle or test store without that file gives ``None``.
"""

import gzip
import json
import math
from pathlib import Path
from typing import Any

from litsense_core.schemas import CatalogBook

MAX_PAGE = 1000


def _read_isbn13(bundle_dir: Path, n_items: int) -> list[str | None]:
    try:
        with gzip.open(bundle_dir / "catalog_text.json.gz", "rt", encoding="utf-8") as f:
            values = json.load(f).get("isbn13")
    except (OSError, ValueError):
        return [None] * n_items
    if not isinstance(values, list) or len(values) != n_items:
        return [None] * n_items
    return [str(v) if v else None for v in values]


class CatalogView:
    def __init__(self, store: Any) -> None:
        self.store = store
        self.version = str(store.version)
        self.total = int(store.catalog.n_items)
        self.isbn13 = _read_isbn13(Path(store.bundle_dir), self.total)

    def book(self, i: int) -> CatalogBook:
        cat = self.store.catalog
        series = int(cat.series_id[i])
        pos = float(cat.series_pos[i])
        year = int(cat.year[i])
        pages = int(cat.pages[i])
        return CatalogBook(
            book_id=i,
            canonical_id=cat.canonical_id[i],
            title=cat.title[i],
            author=cat.author[i],
            genres=list(cat.genres[i]),
            primary_genre=cat.primary_genre_of(i),
            year=year if year > 0 else None,  # the export writes -1 for unknown
            pages=pages if pages > 0 else None,
            description=cat.description[i] or "",
            image=cat.image[i],
            isbn13=self.isbn13[i],
            series_id=cat.series_labels[series] if 0 <= series < len(cat.series_labels) else None,
            series_name=cat.series_name[i],
            series_pos=None if math.isnan(pos) else pos,
        )

    def page(self, offset: int, limit: int) -> list[CatalogBook]:
        stop = min(self.total, offset + min(limit, MAX_PAGE))
        return [self.book(i) for i in range(offset, stop)]
