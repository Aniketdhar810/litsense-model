# Decisions

Each entry: date, decision, reason, alternatives considered. Items marked **(needs user OK)** are defaults
taken to keep Phase 0 moving; they are easy to reverse.

## 2026-10-12: Phase 0

### D1. One set of pins for Python 3.11 and 3.12
- **Decision:** pin `numpy==2.4.6` and `scipy==1.17.1`; CI tests Python 3.11 and 3.12.
- **Reason:** CLAUDE.md fixes Python 3.11 (the backend runs the core), while Colab runs 3.12. numpy 2.5 and
  scipy 1.18 require Python ≥ 3.12, so the newest versions both interpreters share are 2.4.x and 1.17.x.
  Verified with `pip install --dry-run --only-binary=:all:` for manylinux cp311 and cp312 (all 109 packages
  resolve), and an actual install on Windows cp312 for the core set.
- **Alternatives:** move everything to 3.12 (would contradict the brief and the backend's Python); separate pin
  files per interpreter (two sets of truth).

### D2. Version pins: conservative where code exists, provisional elsewhere
- **Decision:** pins are the latest releases at least two weeks old that resolve together, except
  `pandas==2.3.3` (last 2.x) instead of 3.0. Pins for packages not imported until later phases (torch,
  sentence-transformers 6.x, transformers 5.x, spaCy, implicit, onnx, fastText) are provisional and get
  re-verified in the phase that first imports them. `optimum` is left out until Phase 3 picks the ONNX export
  path.
- **Reason:** pandas 3.0 changes defaults (copy-on-write, string dtype) and Colab images tend to ship 2.x; the
  rest has no code yet, so a pin cannot be tested.
- **Alternatives:** pandas 3.0.6 (newer but riskier); leaving later packages unpinned (violates rule 0.2.10).
- **Open:** pinning `torch` will make Colab download a multi-GB CUDA wheel if its preinstalled torch differs.
  Proposal for the Phase 1 Colab setup notebook: install `requirements-ml.txt` without torch, keep Colab's torch
  and record its version in manifests. **(needs user OK)**

### D3. fastText from `fasttext-wheel`
- **Decision:** pin `fasttext-wheel==0.9.2`.
- **Reason:** the official `fasttext` package ships no wheels (needs a C++ build) and its `predict()` breaks under
  numpy 2 (`np.array(..., copy=False)`). `fasttext-wheel` has cp311/cp312 wheels for Linux and Windows; Phase 1
  will call the low-level predict (or confirm the wrapper works) when `data.clean` is built.
- **Alternatives:** `fasttext-numpy2-wheel` (Linux only, so no local sample runs on Windows); a different
  language-ID library (contradicts CLAUDE.md §6.3).

### D4. Quoted regexes in `ml/config.yaml`
- **Decision:** the six list entries containing `?` (three `shipping_patterns`, three `boxset_patterns`) are
  single-quoted. Values are unchanged.
- **Reason:** the §5 YAML does not parse: PyYAML ends a plain scalar inside a flow sequence at `?`. Single quotes
  keep backslashes literal. A test checks the patterns still match their intended text.
- **Alternatives:** block-style lists (bigger diff from the brief).

### D5. New config key `project.local_raw_root: ./local/raw`
- **Decision:** `--sample` runs read raw files from `project.local_raw_root` and write under
  `project.local_root`; full runs use `raw_root` and `drive_root`. Relative paths resolve from the repo root.
- **Reason:** §7.0 says `--sample` writes under `local_root` but names no local raw location; `/content/raw` does
  not exist on a laptop. How files get there (download only the first N lines, or `make sample`) is decided in
  Phase 1 with `data.download`.
- **Alternatives:** CLI flags for every root (more surface); deriving it from `local_root` (implicit).

### D6. Stage fingerprint uses each stage's own code, not the repo-wide git SHA **(needs user OK)**
- **Decision:** the code part of a fingerprint is the sha256 of the stage's module plus, transitively, every
  `ml`/`litsense_core` module it imports (found by parsing imports), excluding framework plumbing
  (`ml.common.stage`, `ml.common.manifest`, `ml.common.log`, `ml.stages`). Line endings are normalized so Windows
  and Colab agree. The git SHA (with `-dirty`) is still recorded in every manifest for provenance.
- **Reason:** §7.0 puts the git SHA of `ml/` + `litsense_core/` into every fingerprint, so any commit (even to the
  evaluator or the LLM client) would invalidate every stage and rerun the multi-hour 29.5M-line review stream.
  The import closure reruns exactly the stages whose code changed, with no manual declarations to forget.
- **Alternatives:** the spec's git SHA (safe but reruns everything); manually declared code dependencies (easy to
  forget).

### D7. Input identity is the producer's recorded output hash
- **Decision:** an input produced by another stage is identified by the sha256 that stage's manifest recorded
  for that file; inputs no stage produces (raw files for now) by size + mtime.
- **Reason:** content addressing means a rerun that writes identical bytes does not cascade downstream, while
  changed output always does. Using the producer's fingerprint (the spec's wording) would also cascade on
  no-op reruns. Raw files on Colab's ephemeral disk get a new mtime every session; once `data.download` (Phase 1)
  declares them as outputs, their identity becomes the recorded sha256 and session restarts stop invalidating
  downstream stages.
- **Alternatives:** producer fingerprints (more reruns); hashing raw files on every run (minutes per GB).

### D8. Staging directory instead of `<path>.tmp`; manifest location
- **Decision:** `ctx.out(spec)` returns a path in a per-stage staging directory on the same filesystem as the
  destination; after the stage returns, outputs are moved into place with `os.replace` and the manifest is written
  last to `<root>/manifests/<data_version>/<stage>.json`. A started run deletes the stage's old manifest first; a
  failed run leaves none.
- **Reason:** same atomicity as `<path>.tmp`, but file names keep their extensions (`np.save("x.npy.tmp")` would
  write `x.npy.tmp.npy`). Several stages write into the same `data/<version>/` folder, so manifests are per stage.
- **Alternatives:** `.tmp` suffix (breaks `np.save` and some writers); one manifest per output directory (clashes).

### D9. Framework files and declared access
- **Decision:** the stage framework lives in `ml/common/stage.py` and config loading/validation in
  `ml/common/config.py` (both new files beside the §4.3 layout); `ml/stages.py` is only the CLI. `ctx.path`,
  `ctx.out` and `ctx.param` raise if a stage touches an input, output or config key it did not declare.
- **Reason:** stage modules import the decorator while the CLI imports the stage modules, so they cannot share one
  file without circular imports. Enforcing declarations guarantees the fingerprint covers everything a stage
  reads.
- **Alternatives:** framework inside `ml/stages.py` (circular imports; `python -m` double-import of the registry).

### D10. Schema deviations from §15.2 (identical JSON)
- **Decision:** `HomeRails.because_you_liked: list[BecauseYouLiked]`, `Clarification.options:
  list[ClarificationOption]`, `SessionState.history: list[ChatMessage]` instead of `list[dict]`. Added
  validation: ms timestamps in [1e11, 1e13], stars in [1, 5], `book_id ≥ 0`; `Bucket` and `Intent` literals;
  `default_factory` for mutable defaults; `ASPECT_LABELS` (aspect → UI label, §11.2 table). `Evidence` keeps its
  four `dict` fields until Phase 4 fixes their contents.
- **Reason:** typed models produce better OpenAPI types for the frontend at no JSON cost; validation catches
  backend unit bugs at the boundary. Tests check `ASPECT_LABELS` and `Bucket` against the config.
- **Alternatives:** the literal §15.2 types.

### D11. Bucket membership with fractional `n`
- **Decision:** a reader is in the last bucket whose lower bound is ≤ `n` (cold 0, warming 1, blending 5, warm 20).
- **Reason:** `n` = ratings + 0.5 × list adds can fall between `[1, 4]` and `[5, 19]` (for example 4.5).
- **Alternatives:** `floor(n)` (same results for these bounds, less explicit).

### D12. Mock behaviour choices **(needs user OK where marked)**
- **Decision:** (a) onboarding picks count as seen and are never recommended back; (b) `metrics_report()`
  returns `null` metrics plus `"is_mock": true`, not zeros; (c) chip removal = backend edits the session and
  sends an empty message; a picked clarification option is sent back as the message; (d) the mock catalogue is
  synthetic (invented titles and authors, no real books) and generated from a seeded `random.Random`;
  (e) **proposed** `catalog() -> list[CatalogBook]` so the backend can seed its `books` table. **(needs user OK)**
- **Reason:** (a) picks are books the reader says they enjoyed; (b) rule 0.2.4, zeros would read as results;
  (c) §14.7 says the backend removes a modifier and the core recomputes, but names no call; (d) real titles with
  invented descriptions would be misleading; (e) nothing in §15.1 hands the backend the catalogue.
- **Alternatives:** (e) a CSV export inside the bundle.

### D13. Tiny fixture scope
- **Decision:** Phase 0's `make_tiny.py` generates the raw dataset (5-core CSV, `meta_Books`, reviews, four
  Goodreads files) plus `truth.json` with every planted case. The tiny serving bundle (§16.1, second half) is
  added in Phase 4.
- **Reason:** the bundle's format is defined by `serve.export` (Phase 4) and its feature list by `features.py`
  (Phase 5). Writing it now means guessing both and maintaining the guess twice.
- **Assumption to verify in Phase 1:** Goodreads values are strings (`"num_pages": "352"`) and Amazon timestamps
  are milliseconds, as CLAUDE.md §6 says from memory.

### D14. CI scope
- **Decision:** CI installs `requirements-core.txt` plus the subset of `requirements-ml.txt` the fast tests need
  (`CI_ML_PACKAGES` in the Makefile, currently pandas, pyarrow, pytest, ruff), runs `ruff check`,
  `ruff format --check` and `pytest -m "not slow"` on Python 3.11 and 3.12. Markdown is excluded from ruff (it
  would reformat code blocks in CLAUDE.md).
- **Reason:** the full ML stack (torch with CUDA from PyPI) would add gigabytes per run for tests that do not
  import it yet. The subset grows when fast tests need more (the LightGCN tiny test will add CPU torch).
- **Alternatives:** a third requirements file for CI (another source of pins).

## 2026-10-12: Raw data inspection (`local/raw_sample/`)

Numbers below are measured on the local samples (first N lines of each file), not on full files; they show
what the parsers must handle, not final statistics. Full-file numbers come from the Colab runs.

### D15. Goodreads files come from the UCSD mirror
- **Decision:** new config key `sources.goodreads_base:
  https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads`; a file's URL is `<base>/<file name>`.
- **Reason:** every Google Drive link on the dataset page (and the Drive file pages) returns 404. The mirror
  serves all files over plain HTTPS with Range support: books 2.08 GB, series 28 MB, works 75 MB, spoilers
  620 MB. `gdown` is therefore unnecessary; removing it from `requirements-ml.txt` is left for the Week-2
  download work.
- **Alternatives:** asking the user to upload the files to Drive by hand (CLAUDE.md §6.2 fallback).

### D16. Local sample files
- **Decision:** `local/raw_sample/` (gitignored by the existing `local/` rule) holds the first lines of four
  files, streamed with `curl | gunzip | head` and re-gzipped with `gzip -n` under the original file names so
  they can stand in for raw files: `Books.csv.gz` (200,000 lines = header + 199,999 ratings),
  `meta_Books.jsonl.gz` (50,000), `Books.jsonl.gz` (50,000 reviews), `goodreads_reviews_spoiler.json.gz` (5,000).
- **Finding that shapes `--sample` runs:** the files are not mutually consistent. The 5-core CSV and the
  review file are both grouped by user in the same user order, so 60.2% of sampled review pairs are in the
  ratings sample. The meta file is ordered differently: only 3.6% of sampled meta items appear in the ratings
  sample (15.4% appear in the full 5-core). A useful local `data.meta` run therefore needs the full 5-core CSV
  (0.14 GB) as its item list, not its first N lines.

### Data format findings (F1–F12)
- **F1. `meta_Books` contains Kindle editions and audiobooks.** `main_category` is "Buy a Kindle" for 6.6% of
  meta records and "Audible Audiobooks" for 2.5%. Among sampled meta items that are in the 5-core, Kindle
  editions carry **21.1% of the 5-core interactions** (audiobooks 0.4%). This contradicts PRD §17.1 ("Kindle
  e-books live in a separate file"). Kindle records have `parent_asin` `B0…` (100%), no `ISBN 10`/`ISBN 13`
  (0%), page counts under `Print length` (98%), `Publication date` instead of a dated `Publisher`, and
  sometimes `Page numbers source ISBN` (17%), which points at the print edition. Consequences for
  `data.canonical`: Kindle/print merges are common and mostly heuristic (no ISBN for the Goodreads join);
  `FORMAT_KEYS` needs a Kindle/Audible format; `Page numbers source ISBN` can link a Kindle edition to its
  print ISBN. **Needs a user decision before Week 3:** keep Kindle and audiobook editions (merged into their
  works) or drop them.
- **F2. `main_category` is not always "Books":** also "" (82), null (3), "Office Products", "Toys & Games",
  "Musical Instruments". `categories` is empty for 8.9% of records; `categories[0]` can be
  "Audible Books & Originals". Genre slicing on `categories[1]` must tolerate both.
- **F3. `details["ISBN 10"]` sometimes holds a 13-digit value** (1,511 of 40,138; also a few 4/5/12-digit
  values). `norm_isbn` must validate length and route 13-digit values to ISBN-13. `ISBN 13` has a dash in
  84.7% of records.
- **F4. Author sources disagree.** `author` is null for 28.5% of records (`store` is present for 95% of
  those). One `author` dict has no `name` key. Where both exist, the `author.name` surname is missing from
  `store` 5.3% of the time (for example an editor credited in `store`). `store` lists several contributors
  for 25% of records ("Name (Author), Name (Illustrator)…"), sometimes with a trailing "Format: Kindle
  Edition", and has no "(Role)" for 203 records. Parse `author.name` first, then the first "(Author)" in
  `store`.
- **F5. `description` mixes sections.** 51.4% of records have an empty `description` (`features` is
  non-empty for 90.9%, so it is the main fallback). Non-empty descriptions often start with section labels:
  "Review" (19% of records), "About the Author" (18%), "From Publishers Weekly", "From the Back Cover". The
  §10.1 content text builder should drop author bios and editorial-review sections or prefer the blurb.
- **F6. `images` entries usually have only `large` and `variant`** (97%); `thumb`/`hi_res` appear in 30
  records. No dict-of-lists form seen.
- **F7. No JSON-string encodings seen.** `details` is always a dict and `author` a dict or null in this
  sample; the §7.2 `_maybe_json` handling stays as cheap insurance.
- **F8. Dates.** The `Publisher` "(Month D, YYYY)" pattern gives a year for 92.0% of records; Kindle records
  use `Publication date` (6.8%); `subtitle` often ends in a date ("Paperback – January 1, 2004"), a useful
  third fallback. `subtitle` is null for 4.3% and is sometimes just "Kindle Edition" or "1st Edition".
- **F9. 5-core CSV layout.** Ratings are floats ("5.0"), timestamps are milliseconds (sample range
  1997-08-14 to 2023-03-20; full file 1996-08-17 to 2023-09-12). Rows are grouped by user and time-sorted
  within a user, but users are not in alphabetical order. The first 200K lines are 10,210 complete user
  histories (5 to 3,134 ratings each).
- **F10. Reviews.** Timestamps are milliseconds. `images` entries are dicts of `small_image_url`,
  `medium_image_url`, … (unused). `asin` differs from `parent_asin` in 0.3% of reviews. The raw file has
  duplicate (user, parent_asin) pairs (11 in 50K) that the 5-core file does not. 37.9% of texts contain HTML
  tags (mostly `<br />`) and 6.0% HTML entities; 6.8% are shorter than 20 characters.
- **F11. Goodreads spoiler file.** `rating` is an int 0–5 where 0 means "no rating" (1.9% of reviews);
  `timestamp` is a date string ("2017-08-30"), not milliseconds; `review_sentences` is `[[int, str], …]` and
  `has_spoiler == any(label)` holds. Spoiler sentences are rare (1.9% of sentences, 5.9% of reviews), which
  supports the balanced sampling in §11.7. The file is grouped by user (52 users in the first 5,000 reviews),
  so its head is not a representative sample.
- **F12. Goodreads book records** (checked while planning, first 2,000 lines): every scalar is a string,
  empty string for missing; `series` is a list of id strings; real series suffixes include "(Pluto's Snitch
  #2)" (no comma) and "(Sun Wolf and Starhawk, #1-2)" (a range, correctly left unparsed by the §7.3 regex).
  The tiny fixture's assumptions (D13) hold.

## 2026-10-13: Week 2 (Phase 1 data: download, meta, Goodreads)

### D17. Kindle and audiobook records are editions of the same work (user decision on F1)
- **Decision:** keep Kindle and audiobook records from `meta_Books` and treat them as editions of the same
  work. `data.meta` keeps and tags them (`edition_kind`: print, kindle, audio, unknown; Kindle records also
  carry `page_source_isbn13`, the print ISBN from "Page numbers source ISBN"). `data.canonical` (Week 3)
  merges them with print editions by ISBN when one is present (including the page-source ISBN), otherwise
  by the title+author heuristic with the box-set and volume guards of CLAUDE.md §7.4.
- **Reason:** user decision (2026-10-13). In the 50K-line sample Kindle editions carry 21% of 5-core
  interactions overall and 68.5% of the "Mystery, Thriller & Suspense" interactions; dropping them would
  remove most of some genres.
- **Alternatives:** drop Kindle and audio records (PRD §17.1 assumed they were not in this file).

### D18. Large sources are streamed from their URLs, never stored (user rule)
- **Decision:** `meta_Books` (and in Week 3 the reviews; later the Goodreads spoiler file) are read with
  `ml/data/stream.py`: bytes are gunzipped in memory and filtered line by line. The stage framework has a
  `url:<source>` input type; `ctx.source()` returns the URL, or with `--sample` the same-named local head in
  `project.local_raw_root` (now `./local/raw_sample`) when it exists, else the URL read for N lines. The
  input's fingerprint identity is the URL (plus the local head's size and mtime in sample runs).
- **Resilience:** a dropped connection resumes with an HTTP Range request from the exact compressed byte
  reached, feeding the same decompressor (tested with a server that hangs up mid-file). A crashed process
  cannot resume a gzip stream, so `data.meta` checkpoints its output every `data.stream.checkpoint_rows`
  kept records into its work dir on Drive (`PartWriter`); a rerun re-streams from byte 0 but skips the
  lines already processed without parsing them, and counters resume from the checkpoint (tested: output
  and counts identical to an uninterrupted run). Every stream must end with a complete gzip trailer and
  deliver exactly its Content-Length.
- **Alternatives:** download then parse (needs about 11 GB of Colab disk per session for meta + reviews and
  re-downloads after every restart).

### D19. `data.download` keeps only the 5-core ratings, always whole
- **Decision:** `data.download` fetches the 5-core CSV into `data/<v>/ratings_5core.csv.gz` on Drive (not
  ephemeral raw disk), byte for byte, while checking the header, stars in {1..5} and millisecond
  timestamps (`DataError` otherwise). It reads the whole file even with `--sample` (user OK), because every
  filter keys on it and its first lines barely overlap the meta head (D16). fastText's model moves to Week 3
  with `data.clean`; the Goodreads files are streamed by `data.goodreads`.
- **Reason:** 0.14 GB, used by every later stage; keeping it on Drive avoids re-downloads after restarts.

### D20. The Goodreads book graph is optional (user rule)
- **Decision:** `data.use_goodreads_graph: false` by default. Off, `data.goodreads` makes no network request
  and writes `gr_isbn_index.parquet` and `gr_series.parquet` with zero rows and their final schema, so
  `data.canonical` has one code path. On, it streams the books, series and works files and keeps only
  books whose ISBN-13 (Goodreads ISBN-13, or its ISBN-10 converted) matches a `books_raw` ISBN: print
  ISBN-10/13 or a Kindle edition's page-source ISBN (user requirement). The user enables it after
  `data.meta` has run on Colab.
- **Consequence:** this departs from CLAUDE.md §1.5 (Goodreads ISBN join first, heuristics as fallback)
  while the flag is off: series ids and positions then come only from Amazon title patterns (Week 3), so the
  series guard and "continue series" cover fewer books.
- **Spoiler file:** the Goodreads spoiler reviews (Phase 3) are streamed independently of this flag.

### D21. Torch on Colab: use the preinstalled build (user decision; closes D2's open item)
- **Decision:** never install torch on Colab. `make install-colab` installs `requirements-core.txt` and
  `requirements-ml.txt` minus the `torch==` line; `notebooks/00_colab_setup.ipynb` calls it and prints the
  torch version actually used. `requirements-ml.txt` keeps its torch pin for laptops and CI.

### D22. `data.meta` details
- **Prefilter:** `parent_asin` is read from the raw line (`json_str_field`) and only 5-core lines are
  JSON-parsed. On the real 50K-line head this parses 7,707 of 50,000 lines; `parent_asin` extraction agreed
  with full JSON parsing on every meta and review line.
- **Parsing rules from F1–F8:** author = `author.name`, else the first "(Author)" in `store` with any
  "Format: …" suffix removed, else its first credit; ISBNs are classified by shape, so a 13-digit "ISBN 10"
  becomes the ISBN-13; pages come from any "N pages" value (Kindle `Print length` included); year from the
  Publisher date, then `Publication date`, then the subtitle; the binding is a `details` key or, for about
  a tenth of print books, the subtitle's first part ("Paperback – January 1, 1994"). With these rules the
  real head has 1.0% `unknown` editions (10.4% before the subtitle fallback).
- **Memory:** `books_raw.parquet` is written from the checkpoint parts one part at a time (duplicates
  dropped across parts), and the category counts read back only four columns. On the sample it takes
  about 2.4 KB per book on disk with zstd (D24), about 1.2 GB for the full 5-core, too much to hold as one
  pandas table on Colab's ~12 GB.
- **Columns beyond §7.2:** `subtitle`, `main_category`, `features` (kept apart from `description`, which
  often holds author bios and editorial reviews, F5), `page_source_isbn13`, `format`, `edition_kind`,
  `n_interactions`. Explicit pyarrow schemas fix column types even when a sample has only nulls.
- **Outputs for the decision:** `category_counts.csv` adds Kindle and audio interaction columns;
  `genre_slice_proposal.json` applies the §7.2 rule with its constants in config
  (`data.genre_slice_rule: {min_slices: 2, max_slices: 3, target_multiple: 4}`).
- **Stop:** the stage ends with a STOP message and `python -m ml.data.meta` reprints the report.
  `require_genre_slices(cfg)` raises while `data.genre_slices` is empty; Week-3 stages call it first.
- **Dependencies:** `gdown` removed from `requirements-ml.txt` (D15); `requests` added to the CI package
  subset because `ml.data.stream` imports it.

### D23. Tiny fixture additions
- **Decision:** the fixture now plants a Kindle edition of a print book (page-source ISBN), a Kindle
  edition whose print edition exists only in Goodreads (proves the page-source match), an audiobook, a
  13-digit "ISBN 10", an author dict without `name`, a `store` whose author is not the first credit, and a
  record with an empty title. Totals are unchanged (50 users, 80 books, 600 ratings).

### D24. zstd for `books_raw.parquet`; `features` stays (user request, 2026-10-13)
- **Decision:** `books_raw.parquet` and `data.meta`'s temporary checkpoint parts are written with zstd. On
  the 50K-line sample `books_raw` shrinks from 24.5 MB (snappy, the default) to 17.1 MB (70%) with
  identical content; about 1.2 GB instead of 1.7 GB at full scale.
- **`features` checked, kept:** the user asked to drop `features` if it just duplicates `description`. On
  the sample it does not: the two are never identical; `features` is contained in `description` for 1.5%
  of books, both exist and differ for 76.4%, and `features` is the only text for 20.2% (`description`
  empty). Spot checks show `features` is usually the publisher's blurb (back-cover copy) while
  `description` mostly holds editorial reviews ("Review", "From Publishers Weekly") and author bios (F5).
  The blurb is the spoiler-safe premise text Phase 3 needs, so the column stays unchanged.
- **Alternatives:** drop `features` (loses the only text for one book in five); blank it where it is
  contained in `description` (1.5% of rows; would push Phase 3 to the noisier review/bio text).

## 2026-10-13: First Colab run of `notebooks/01_data.ipynb`

What happened: `data.download` completed on Colab in 23 s (9,488,297 ratings, 776,370 users, 495,063
items, the same counts as locally). `data.meta` was streaming normally (the traceback shows it inside
`parse_meta`) when the cell was stopped: the stage printed nothing for minutes, and Colab's input box
under the running `!` command looked like a prompt. The second traceback (`PosixPath ... '_str'`) was a
second Ctrl+C landing in the staging cleanup; no manifest was written, so nothing downstream trusts the
partial run. The run also showed that Colab now uses **Python 3.13**.

### D25. Colab feedback and Python 3.13
- **Progress output:** long streams (`data.meta`, `data.goodreads`) log a status line about once a minute:
  GB done of the total, percent, MB/s, minutes left, lines and 5-core matches so far (`Progress` in
  `ml/common/log.py`, which is framework plumbing and so outside stage fingerprints).
- **Notebooks:** both now say that the empty box under a running `!` cell is Colab's input box, not a
  question; run cells read stdin from `/dev/null` so nothing typed there reaches the process.
- **fastText:** `fasttext-wheel==0.9.2` has no Python 3.13 wheel, so `make install-colab` could not
  install the pin set on Colab (the stages then ran on Colab's preinstalled pandas/pyarrow/requests).
  Replaced by `fasttext-predict==0.9.2.4`: the prediction-only fastText build (all that language ID in
  `data.clean` needs), same `fasttext.load_model(...).predict(...)` API, wheels for cp311-cp313 on Linux
  and Windows, and no numpy dependency, which also removes the numpy-2 `predict()` problem of D3. Checked
  with the real `lid.176.ftz` model (English, Spanish and German sentences identified correctly).
- **Pins:** the full core+ml set now resolves with binary wheels for cp311, cp312 and cp313 (104
  packages each); CI tests 3.11, 3.12 and 3.13.
- **Alternatives:** compiling `fasttext` from source on every Colab session; pinning Colab to an older
  runtime (not offered for free accounts).

## 2026-10-13: Genre slices and repository hygiene

### D26. Genre slices: Literature & Fiction; Mystery, Thriller & Suspense; Science Fiction & Fantasy
- **Decision (user):** `data.genre_slices: [Literature & Fiction, 'Mystery, Thriller & Suspense',
  Science Fiction & Fantasy]` (values of `categories[1]`).
- **Evidence (full Colab run of `data.meta`, 2026-10-13, 446 s):** 4,448,181 meta lines; all 495,063
  5-core items found; 13,408 dropped by the title blocklist and 27,647 by `min_pages`; 454,008 books in
  `books_raw`. 5-core interactions per allowlisted category: Literature & Fiction 2,143,678 (24.6%, 98,546
  books, 30.9% Kindle); Mystery, Thriller & Suspense 2,064,657 (23.7%, 56,335 books, 70.6% Kindle); Teen &
  Young Adult 383,164 (4.4%); Science Fiction & Fantasy 202,160 (2.3%, 12,939 books); Romance 122,661 (1.4%).
- **Reason:** the §7.2 rule proposed the first two (4,208,335 interactions, threshold 1,000,000). The user
  added Science Fiction & Fantasy (total 4,410,495) because the PRD's demo and persona examples are mostly
  SF and fantasy; without the slice those books would not exist in the catalogue.
- **To watch in Week 3:** SF & Fantasy is small (2.3%); the EDA reports the genre mix after k-core and
  sizing to show how much of it survives. Amazon's `categories[1]` is not always the literary genre (in the
  sample, "The Handmaid's Tale" is under Politics & Social Sciences and Harry Potter under Children's Books),
  so some well-known fiction falls outside the slices; the title lookups at the gate check this.
- **Alternatives:** the two proposed slices only; adding Teen & Young Adult (larger than SF & Fantasy but
  further from the demo examples).

### D27. Notebooks are committed without outputs
- **Decision:** `tests/test_notebooks.py` fails CI if any notebook in `notebooks/` has cell outputs or
  execution counts (no new dependency; checked against the Colab-saved notebook, which it rejects). The user
  will not use Colab's "Save in GitHub" again.
- **Why:** Colab's "Save in GitHub" committed three executed copies of `01_data.ipynb` (with outputs) to
  `claude/litsense-week2-data`. Outputs belong on Drive and in `results/`, not in git (rule 0.2.11).
- **Branch cleanup:** the Colab fix `7b3fc9e` missed PR #3's merge; it was cherry-picked onto a fresh branch
  from `main` (`claude/litsense-week2-colab-fixes`) without the three Colab commits, and
  `claude/litsense-week2-data` was deleted once nothing on it was needed.
- **Alternatives:** an `nbstripout` pre-commit hook (a new dependency, and hooks do not run when Colab
  commits directly).


## 2026-10-13: Week 3, part 1 (`data.canonical`)

### D28. Edition canonicalization rules (deviations from §7.4; thresholds await the merge audit)
- **Merge all editions first, then slice (user, rule A):** the 454,008 `books_raw` editions are merged into
  works, and a work is kept if *any* edition's `categories[1]` is a genre slice. §7.4 filtered by slice
  first. Why: Amazon files editions of one work under different categories. In the Week 2 lookup, the
  Project Hail Mary audiobook sits in SF & Fantasy and the print editions in Literature & Fiction; Mistborn
  book 1 and the box set sit in SF & Fantasy and books 2–6 in Literature & Fiction. Slicing first would
  split works and lose their Kindle ratings. `asin_to_canonical.in_slice_edition` and the report's
  `rescued_*` counts show how much rule A adds.
- **Merge rules, in order:** `isbn` (same ISBN-13, ISBN-10s converted); `page_source` (a Kindle edition's
  "Page numbers source ISBN" equals a print edition's ISBN, F1/D17, unless both have authors and they
  differ); `goodreads` (same work id, only with `use_goodreads_graph`); `title` (below). Union-find; two
  components are never joined if they hold different Goodreads works or different known volume numbers.
- **Title rule:** block by (first token of the normalized title, author surname); merge when
  `token_set_ratio >= fuzzy_threshold` (90, §5) **and** `token_sort_ratio >= subset_guard_threshold` (90,
  new key) **and** the same author key **and** the volume guard passes **and**, only for non-identical
  normalized titles, the year gap is at most `max_year_gap` (3).
  - *Subset guard:* `token_set_ratio` is 100 whenever one title's words are a subset of the other's:
    "martian" vs "martian chronicles", "alex cross" vs "alex cross s trial".
  - *Year gap only for non-identical titles:* reissues of the same title can be decades apart, and Kindle
    publication years differ from print.
  - *Volume guard:* conflict when both volumes are known and differ, or when one is unknown and the other is
    2 or more ("Mistborn" may be book 1, never book 3). Known cost: a Kindle title carrying "(Series, Book 2)"
    does not merge **by title** with a print edition whose title has no marker (it still merges by ISBN or
    page-source ISBN). The audit's `not_merged` rows with reason `volume` show how often this happens.
  - *Author required:* editions without an author never merge by title.
- **Matching key (`ml/data/titles.py`):** NFKC, lowercase, every bracketed segment dropped (series info is
  read first), a subtitle after ':' dropped when it is format noise, marketing or a volume marker
  ("A Novel", "TikTok sensation", "Book Three of Mistborn"), trailing format words and dates dropped
  ("Paperback 18 Feb 2021"), punctuation stripped, leading article dropped. New config:
  `canonical.subtitle_noise_patterns`, plus edition-noise terms (US/UK/international/revised/new edition,
  reissue, movie/media tie-in, MTI).
- **Box sets:** excluded from every rule, shared ISBNs included, and dropped (`drop_boxsets: true`).
- **Canonical record (§7.4 step 7):** title = the most-rated edition's title with format noise removed
  (series bracket kept); author = most common author key (ties go to the most-rated edition); description and
  `features` = the longest among editions (D24); year = min; pages = median; image and ISBN-13 from the
  most-rated edition that has one; also `slices`, `primary_slice` (first slice among the editions, most-rated
  first), `edition_kinds`, `n_interactions`. Series: Goodreads when enabled, else parsed from titles, with
  `series_id = "am_<author key>:<series key>"`. Ids: `gr_<work_id>` or `am_<min parent_asin>`.
- **Audit (§7.4 step 9):** `merge_audit.csv` has 100 random kept works with 2 or more editions (up to 15
  edition rows each, scored against the most-rated edition) and 50 title pairs that were not merged
  (`token_set >= 80`), spread over the reasons `below_threshold`, `subset_guard`, `volume`, `year` and
  `goodreads_work`, drawn from a seeded reservoir of 2,000 pairs per reason (all pairs are counted in the
  report). The reviewer fills `verdict` (Y = right decision, N = wrong). The file is UTF-8 with a BOM so
  Excel opens it correctly. Both audit files are under `results:canonical/`, so they reach the repo with
  `results/v0/` (user rule B: only `results/v0/` and the registry entry are copied from Drive, never data
  files).
- **Open (§0.3):** merge precision from the user's review is recorded here before `data.reviews` runs; the
  thresholds above are not accepted until then.
- **Alternatives:** slice first (§7.4 as written); `token_set_ratio` alone; `fuzz.WRatio`; no volume guard
  for one-sided volumes.

### D29. Genre labels (`ml/data/genres.py`, §7.4 step 8)
- 26 ordered keyword labels plus the fallback "Fiction (general)". A path's labels come from its **deepest**
  element that matches anything, so "Fantasy" wins over the compound top level "Science Fiction & Fantasy",
  which would match both. Audience labels (Young Adult, Children's) are added from any level. A work's
  genres are the union over its editions, most-rated first; `primary_genre` is the first non-audience label.
- **Limitation measured on the 50,000-line sample head:** 24,817 of the 69,963 in-slice interactions sit
  under "Literature & Fiction > Genre Fiction" with no deeper level, so those works get the fallback label.
  Genres are filters and ranker features, not the slice rule. If the full-run EDA shows the fallback
  dominating, Goodreads shelves (with `use_goodreads_graph`) are the obvious refinement.
- **Alternatives:** labels from the top level only (too coarse); every matching label at every level (SF &
  Fantasy books would all get both labels).

### D30. `data.lid` keeps `lid.176.ftz` in `data/` (user decision C)
- A separate stage, `data.lid` (in `ml/data/download.py`), fetches fastText's language-ID model (0.9 MB,
  always whole, even in `--sample` runs) for `data.clean` and checks the fastText magic number, so an HTML
  error page cannot pass as a model.
- **Side effect on Colab:** this change and the new config validation (`ml/common/config.py`) are in the
  code fingerprints of `data.download` and `data.meta`, so both rerun once (about 8 minutes, mostly the
  `meta_Books` stream); `require_genre_slices` now takes the slice list.

## 2026-10-14: Merge audit, first pass

### D31. Shared-ISBN merges need the same author (user decision after the audit)
- **Finding:** in the full-run `merge_audit.csv`, group G018 joined a crafts book (*Log Cabin
  Variations*, author "Annies") to *The Long Ships* because Amazon lists both under one ISBN. The `isbn`
  rule had no author check.
- **Decision:** an `isbn` merge now applies the same rule as `page_source`: the authors must match unless
  one is missing. No threshold changed. Regression test `test_shared_isbn_needs_the_same_author`.
- **Not changed (user's choice):** the subset guard stays at 90, so G054 (a sequel merged at
  `token_sort` exactly 90) stays a known false merge.
- **Audit of run 1 (verdicts model-drafted, spot-checked by the user; the user will spot-check
  again later):** 97/100 merged works correct (wrong: G018 shared-ISBN data error, fixed here; G054 sequel
  at the subset-guard boundary; G004 one two-in-one "Animal Farm: 1984" edition) and 44/50 not-merged
  pairs correct (all 6 misses were same works blocked by the subset guard, which led to D32). File:
  `results/v0/canonical/merge_audit_run1_reviewed.csv` (verdict + note columns).

### D32. "Title" merges with "Title: subtitle" when the blurbs agree (user decision)
- **Rule (`subtitle`, runs after every other rule):** an edition whose title has no subtitle joins an
  edition whose title is that text plus ": subtitle" when all of these hold:
  - the bare title's matching key equals the key of the other's text before the colon (`title_head`);
  - the author key is the same (never missing);
  - the volume guard passes and the two works hold no different Goodreads works;
  - neither is a box set or omnibus;
  - the blurbs are similar: `features`, else `description` (D24), first
    `canonical.subtitle_description_chars` = 500 characters, lowercased, punctuation collapsed; rapidfuzz
    `token_sort_ratio` >= `canonical.subtitle_description_threshold` = 70.
  An empty blurb on either side means no merge.
- **Why:** 6 of the 50 audited non-merges were the same work with a subtitle on one side ("Cold Hand in
  Mine" vs "Cold hand in mine: Strange stories"). The subset guard blocks these by design, and the blurb
  check stops a bare series name joining book 1 ("Mistborn" vs "Mistborn: The Final Empire").
- **Threshold evidence (50K-line sample head; small, rough):** 27 same-work edition pairs and 534
  different-work pairs by the same author. At 70, 41% of same-work pairs and 1.9% of different-work pairs
  pass. At 60: 44% / 3.9%. 70 favours precision; the title-head condition is the main filter.
- **Cost control:** at most 3 description comparisons per pair of works (most-rated editions first);
  blurbs are read from `books_raw` only for candidate editions.
- **Audit:** `merge_audit.csv` gains a `subtitle_merge` section: 30 random kept works joined only by
  this rule (each subtitle edge joins two works the other rules kept apart), showing the two joining
  editions with `merge_rule = subtitle` and the description similarity in `reason`. Pairs rejected for
  dissimilar blurbs are near misses with reason `subtitle_description`. `canonical_report.json` has a
  `subtitle_rule` block of counts by outcome.
- **Alternatives:** TF-IDF cosine (needs scikit-learn in the stage; no clear gain at this sample size);
  `token_set_ratio` (similar separation, more lenient on subsets); merging without a blurb check (would
  join series names to book 1).

## 2026-10-14: Run 2 of `data.canonical` (D31 + D32)

### D33. Different numbers in fuzzy-matched titles block a title merge
- **Finding (run 2):** one kept work held *NYPD Red 2, 3, 4* and *6* (17 editions, 2,649 interactions).
  "nypd red 2" vs "nypd red 3" scores 90 on both fuzzy ratios, and `volume_number` does not read a
  bare trailing number. The editions were chained by the `title` rule; the subtitle rule then added one
  more NYPD Red 6 edition. The bug existed in run 1 too; the random merged-work audit did not sample it.
- **Decision:** for non-identical normalized titles, the `title` rule refuses a merge when the titles'
  number tokens differ (digits and multi-letter roman numerals, `number_tokens`). The near-miss reason is
  `number`. Test `test_numbered_titles_never_merge`. Identical titles are unaffected.
- **Effect on the 50K sample:** one title merge removed: two editions of an out-of-slice programming book
  ("... iPhone SDK" vs "... iPhone 3.0 SDK"), which arguably are different editions anyway.
- **Alternatives:** read a trailing number as a volume (would break titles like "Catch-22" and "1984");
  raise the fuzzy thresholds (would lose real edition merges).

### Run-2 audit (subtitle section; verdicts model-drafted, spot-checked by the user)
- `subtitle_merge`: 30/30 sampled works correct (the same book with a marketing, genre or series
  subtitle). Run-2 totals (from the user's Colab output): 1,603 subtitle merges from 10,850 candidates;
  5,375 rejected for dissimilar blurbs, 555 for missing blurbs, 2,734 by the volume or work guard;
  648 kept works joined.
- The merged-work and not-merged sections of run 2 were not re-reviewed: the rules behind them changed
  only by D31 (fewer ISBN merges: 705 to 513) and now D33.

## 2026-10-14: `data.canonical` closed; `data.reviews`

- **Run 3 of `data.canonical` (after D33):** 454,008 editions became 384,579 works (50,127 with 2 or more
  editions); kept 130,960 works and 4,499,455 interactions; the NYPD Red chain is gone from the largest
  works. Near misses: number 6,852, year 1,496 (some pairs moved from year to number). Edition merging is
  accepted with the audits recorded in D31-D33 (model-drafted verdicts, spot-checked by the user).

### D34. `data.reviews` details
- **Join:** a review is kept when its `(user_id, parent_asin)` is a 5-core rating of an edition whose work
  is `kept`. Both ids are read from the raw line before JSON parsing; join on `parent_asin`, never `asin`
  (F10). Timestamps must be milliseconds or the stage fails.
- **Dedup order (§7.5 step 4):**
  1. The latest review per `(user_id, canonical_id)`. This covers both the same review on two editions
     and two editions that `data.canonical` merged.
  2. Then one per `(user_id, text_hash)`. `text_hash` is a 16-hex sha1 of the text with tags, case and
     punctuation removed. Empty texts are never text duplicates.
- **No-review ratings** are not rows here. `data.kcore_split` builds interactions from the 5-core file and
  sets `has_text` from this table; the stage logs `pairs_without_review`.
- **Memory:** the pair set holds the kept works' 5-core pairs as strings (about 4.5M on the full run).
  Parts are checkpointed every `data.stream.checkpoint_rows` matches, and dedup reads only the key
  columns before rewriting the parts.

### D35. `data.clean` details
- **Rules, in order:** clean the text (tags, entities, NFKC, whitespace; texts over `max_review_chars`
  are cut at the last sentence end); then `too_short` (< `min_review_chars`); then `non_english`;
  then `shipping` (a `shipping_patterns` match and no aspect-lexicon term from `ml/nlp/aspect_seeds.py`).
  Every review keeps its row; `text_ok` and `text_drop_reason` only gate NLP use.
- **Language ID:** fastText `lid.176` on the first 500 characters flags `non_english` only when the
  other language has probability >= `data.lid_min_confidence` (0.5, new key). Short English texts are
  often misread: "Great read!!" -> en 0.12, "meh" -> de 0.44, "Gut" -> sr 0.48, while "Muy bueno el
  libro, lo recomiendo" -> es 0.95.
- **fastText bug:** `fasttext-predict` 0.9.2.4 mis-unpacks batch `predict(list)` results (a 3-text
  list raises; a 2-text list returns labels in place of probabilities). Texts are scored one at a
  time; the loader is injectable, so CI tests use a stub because fastText is not installed there.
- **`ml/nlp/aspect_seeds.py`** (taxonomy, lexicons, prototypes, descriptor poles from §11.2) is added
  now because the shipping rule needs the lexicon.

### D36. `data.kcore_split` details
- **One interaction per (user, work):** the latest 5-core rating when a reader rated two editions of
  one work (§7.7 assumed this would not happen). `has_text` = that reader's review of the work survived
  `data.clean` with `text_ok`.
- **Sizing:** every user draws one seeded uniform number; a fraction f keeps users whose draw is
  below f, then the k-core reruns. f is found by bisection (up to 40 steps) until the size is within
  `target x (1 +/- size_tolerance)`; otherwise the stage fails and reports the closest size. f is in the
  manifest notes and in `splits.json` (`sizing`); it is copied here after the full run.
- **`splits.json`** also carries the attrition counts and the slice and genre mix after the k-core and
  after sizing (the user's "how much SF & Fantasy survives" request). `data.eda` adds the works-kept
  step from `canonical_report.json`.
- **Ids:** `id_map.json` = sorted user ids and sorted canonical ids, written only here; a rerun gives
  identical bytes (tested).
- **Framework:** `StageContext.upstream_notes(stage)` reads an upstream stage's manifest counts (for
  the EDA attrition table); the EDA declares one output of each such stage as an input.
- **CI** installs matplotlib (`CI_ML_PACKAGES`) for the EDA test.

### D37. Checkpoints survive a part lost by Google Drive
- **Finding (Colab, 2026-10-14):** a first `data.reviews` run streamed the whole file and stopped
  before finishing. On rerun, `progress.json` named 90 parts but `part-00089.parquet` was missing; the
  Drive FUSE mount had kept the later progress file and lost the earlier part.
- **Decision:** `PartWriter` records each part's line offset and counters (`history`). On resume it
  checks every recorded part and rolls back to the last one that is present and readable. Old
  checkpoints without history restart from line 0. `close()` marks the stream `finished`, so a
  rerun after a post-processing failure skips the re-stream. `data.reviews` frees its pair set
  before the dedup.
- **Cost of this fix on Colab:** `ml/data/stream.py` is in the code fingerprints of `data.meta` and
  `data.reviews`, so both rerun once, and the reviews stream starts over.

## 2026-10-14: v0 frozen (Week 3 run on Colab)

### D38. v0 sizing, and Drive flushing
- **Sizing:** the user fraction recorded by `data.kcore_split` is in `models/registry.json`
  (`datasets.v0.sizing`), and the v0 numbers are in `results/v0/eda/summary.json` and `attrition.csv`.
  Sampling users and then re-applying the 10-per-item core leaves a small catalogue (see `summary.json`);
  this is fine for development, and v1 sizing is revisited before Week 11.
- **Drive flushing:** after a run, Drive's FUSE mount can still be uploading when the session ends (a
  second runtime briefly saw a 0-byte `reviews_clean.parquet` and no later manifests). The data notebook
  now ends with `drive.flush_and_unmount()`, and one-off cells do the same.
- **Open (user decision):** the slice mix shifts away from SF & Fantasy at the k-core and sizing steps
  (`summary.json` -> `slice_mix`).

### D39. Open items for the v1 rebuild (user decision, 2026-10-14; resolve before Week 11)
v0 stays as built, and Week 4 starts on it. Before v1:
1. **Slice-aware sizing:** sample users so each genre slice keeps roughly its pre-sizing share of
   interactions.
2. **Book threshold:** compare `kcore.item` 5 vs 10. For each, report catalogue size, long-tail share
   and number of evaluable test users; the user chooses (the long tail is the thesis).
3. **Catalogue size:** target roughly 10-20K books in v1; report what each sizing option gives.
4. **Serving catalogue vs training set:** propose (do not build) serving a larger catalogue, e.g. all
   in-slice works with >= N reviews, scored by content and review features for books outside
   training, with its cost (bundle size, RSS, latency, evaluation caveats).

## 2026-10-14: Week 4 (evaluation harness and baselines)

### D40. Harness and baseline choices
- **Scorer cutoff:** each scorer records the cutoff of the `FitData` it was fit on. `evaluate` raises
  `LeakageError` when it differs from the mode's cutoff, and `FitData` refuses rows at or after its
  cutoff. Stages write the cutoff into their manifest (§3.2).
- **Degenerate scores:** a row whose finite scores all tie, or that has fewer than two finite scores,
  raises `DegenerateScoresError`; -inf outside a candidate set is allowed (§8.3).
- **Cold items:** collaborative scorers (popularity, ItemKNN, ALS) give items with no fit interactions
  -inf (§8.1); content-only scores every item.
- **Baselines are refit, not stored:** `eval.baselines_val` fits random, popularity, ItemKNN, ALS and
  content-only inside the stage (seconds to minutes each). Model artifacts with manifests start with
  LightGCN in Week 5, which serving needs.
- **Content text:** title, author, genre labels, then the blurb (`features`, else `description`)
  truncated by tokens to the 256-piece budget. Reviews and LLM text never go in.
- **Content-only profile:** the rating-weighted mean of content vectors of the user's fit items rated
  >= `relevance_min_rating` (all fit items if none), L2-normalized (§9.2).
- **Reports:** `table_main.csv` (means with 95% bootstrap CIs, beyond-accuracy metrics, catalogue size
  as the coverage denominator), `table_strata.csv`, `table_longtail.csv` (long-tail and cold-item
  truth), and `bootstrap.csv` (each model vs popularity, paired, all users and long tail), plus two
  figures. The admin `metrics.json` (§15.6) follows once LightGCN results exist.
- **Test lock:** `ml/eval/lock.py` implements §8.8 now (tested); the `eval.test` stage is Week 13.
- **ALS:** `implicit` 0.7.3 on the (users x items) matrix with confidence alpha x binary; output
  shapes are asserted, and BLAS runs single-threaded during the fit.
- **CI** installs scipy and implicit.

### D41. Edits to shared modules rerun every stage that imports them
- Adding `LeakageError` and `DegenerateScoresError` to `ml/common/errors.py` would have changed the code
  fingerprint of every data stage (all import it) and rerun the whole data pipeline on Colab, including
  the review stream. They live in `ml/eval/errors.py` instead, and `ml/common/errors.py` is back to the
  bytes Colab ran. Rule: keep finished stages' import closures stable; put new shared code in new modules.

### D42. "Up to date" now also checks output sizes
- **Finding:** on Colab, `reviews_clean.parquet` stayed on Drive as a 0-byte file while its manifest
  said complete. The next stage that read it failed (`Parquet file size is 0 bytes`).
- **Decision:** a stage is current only if every output exists with the byte size its manifest
  recorded (directories: existence). Sizes are cheap; full hashes are not. A truncated output makes the
  stage stale, so `run all` rebuilds it. `ml/common/stage.py` is framework code, outside stage
  fingerprints, so this change reruns nothing by itself.

## 2026-10-15: Week 5, LightGCN

### D43. LightGCN training, artifacts and evaluation stage
- **Stage `models.lightgcn_val`** (`ml/models/lightgcn.py`) trains on train (cutoff t1) and writes
  `artifacts/<v>/eval_val/lightgcn/`: `lgcn_user.npy`, `lgcn_item.npy`, `lgcn_item_norm.npy`,
  `item_degree.npy` and `meta.json` (cutoff, mode, params, best epoch, device, torch version), plus
  `results/<v>/training/lightgcn_val.csv` (loss and inner NDCG@20 per evaluation). The test-mode set
  (train + val, t2) reuses `build_lightgcn` in Week 13.
- **Training choices not fixed by s9.4:** one epoch = every fit positive once in shuffled batches;
  full-graph propagation every step; the L2 term counts the negatives' ego embeddings divided by
  `n_negs`; the inner split holds out max(1, floor(0.1 n)) of the latest rows of users with >= 6 fit
  interactions and is scored by the shared `build_eval_data` + `evaluate` (same masks and relevance
  rule as val). The refit trains from scratch on the whole fit window, same seed, for the best epoch
  count. GPU sparse products are not bit-reproducible; CPU runs are (tested).
- **Negative sampler:** vectorized redraws, then the rare survivors are drawn from the user's
  complement, so a positive is never returned (s16.2).
- **Scoring:** `ml/models/embedding_scorers.py` (numpy only, so evaluation never imports torch) gives
  `lightgcn_raw` (u . i) and `lightgcn_norm` (u . i/|i|); items with no fit interactions score -inf.
  A stored scorer's `fit` refuses an artifact whose cutoff, shapes or item degrees differ from the
  evaluation's fit data.
- **Stage rename:** `eval.baselines_val` became `eval.val` (`ml/eval/validation.py`, the name in
  s7.0). It refits the baselines, loads LightGCN, copies `meta.json` into each LightGCN result file,
  and writes paired bootstraps against popularity, `lightgcn_raw` and `lightgcn_norm` (s2.2 #1). The
  baseline numbers are recomputed with the same seed and code.
- **CI** installs CPU-only torch (the `requirements-ml.txt` pin) from PyTorch's CPU index; the PyPI
  wheel bundles CUDA.
- **Speed:** about 6 s per epoch on a laptop CPU at v0 size (synthetic benchmark), so Colab runs use a
  T4 GPU runtime.
- **Hyperparameters:** the s5 values for the first run; if the best epoch is the last one allowed,
  training was still improving and `lr`/`max_epochs` are tuned on val (logged here).

### D44. First LightGCN run on v0 (Colab T4, 2026-10-16): gate check and settings kept
- **Run:** `models.lightgcn_val` took 234 s on a T4 (inner search 300 epochs, best epoch 275, then the
  refit); 379 items have no train interactions. Inner NDCG@20 rose quickly to epoch ~200 and changed
  little over the last 60 epochs (`results/v0/training/lightgcn_val.csv`), so the s5 values
  (lr 1e-3, `max_epochs` 300) are kept; no tuning yet.
- **Week 5 gate item "LightGCN >= ALS on val": met.** In `results/v0/val/bootstrap.csv`, ALS minus
  `lightgcn_raw` on NDCG@10 has a 95% CI below zero; on Recall@20 the CI crosses zero. ItemKNN and
  `lightgcn_raw` are not distinguishable on either metric (CIs cross zero).
- **Normalization (s2.2 #1) shows up as expected:** `lightgcn_norm` is below `lightgcn_raw` on all
  users but above it on long-tail truth, with far higher coverage and tail share
  (`table_main.csv`, `table_longtail.csv`). Long-tail claims must compare against both.
- **Baselines reproduced:** the refit baselines match Week 4; ALS differs only in the 7th decimal
  (multithreaded BLAS float order).
- **Colab environment noise (harmless):** pip warns about google-colab's own pins, and `implicit`
  disables its GPU module (CPU ALS is what we use).

## 2026-10-16: Week 5, projection hybrid and controls

### D45. Projection hybrid, shuffled control and static fusion
- **Stage `models.projection_val`** (`ml/models/projection.py`) trains W (dim x 384, no bias,
  Xavier-uniform init, seeded) by BPR with LightGCN users and items frozen, and writes
  `artifacts/<v>/eval_val/projection/` and `.../projection_shuffled/` (`W.npy`, `proj_item.npy`,
  the LightGCN `user.npy` it scores with, `item_degree.npy`, `meta.json` with the normalize flag and
  per-epoch BPR). Training and scoring use the same function: u . e'/|e'| with `normalize_items`
  (scoring re-normalizes the stored, already normalized rows, which changes nothing).
- **New config key `models.projection.n_negs: 5`** (s9.5 says negatives come from
  `sample_negatives` but names no count; 5 matches LightGCN). Each step draws a batch of fit
  positive pairs uniformly with replacement.
- **Cold items:** e_gcn = 0 for items without train interactions, so their vector is W c only.
  They are never positives in the fit window but can be drawn as negatives. Both projection scorers
  rank cold items (`EmbeddingScorer(score_cold=True)`); LightGCN scorers still give them -inf.
- **Shuffled control** permutes the content rows with the stage RNG and is otherwise identical; the
  stage refuses `shuffled_control: false`, since s9.5 requires the control.
- **Static fusion (interpretation of s9.5):** score = alpha * cos(u, e_gcn) + (1 - alpha) *
  cos(p_u, PCA(c)), where PCA reduces content to the LightGCN dim (centered SVD, numpy) and p_u is the
  content-only profile (s9.2) in that space. It is one dot product of concatenated vectors
  [alpha * u/|u|, (1 - alpha) * p_u] . [e_gcn/|e_gcn|, PCA(c)/|PCA(c)|]. Summing
  normalize(e_gcn) + normalize(W_pca c) into one 64-d vector and scoring it with the LightGCN user
  vector (the literal reading) would mix two unaligned spaces, so the content half would be noise.
  Built inside `eval.val` (no training), one row per `models.static_fusion_alphas`; the best alpha is
  read from val only.
- **Speed:** about 6 s per 100 steps on a laptop CPU (10,000 steps per variant), so Colab runs on
  a T4.

### D46. Projection and controls on v0 val (Colab T4, 2026-10-16): a null result, reported as such
- **Run:** `models.projection_val` 75 s (both variants, 10,000 steps each); `eval.val` 35 s.
  `models.lightgcn_val` also reran (its import closure includes `embedding_scorers.py`, changed in
  D45) and reproduced the same best epoch and metrics.
- **Reading (from `results/v0/val/table_main.csv`, `table_longtail.csv`, `bootstrap.csv`):**
  - The projection hybrid is below raw LightGCN on all users (paired CI below zero) and not
    distinguishable from normalized LightGCN; on long-tail truth it is above raw LightGCN but below
    normalized LightGCN.
  - **Projection vs its shuffled control: no difference** on any headline metric. On v0 the
    projection's gain over raw LightGCN on the long tail is a normalization effect, not a semantic
    one (exactly the s2.2 #1 confound).
  - Static fusion (best alpha 0.7 on val) is the strongest fusion row: it beats normalized LightGCN
    on all users and on the long tail (paired CIs above zero), ties raw LightGCN on all users and
    beats it on the long tail.
  - **No fusion row recovers cold items** (cold-item NDCG@10 is zero): cold items are never positives
    in training, and in static fusion their score has only the content half, so warm items outrank
    them. Content-only remains the only model that ranks cold items.
- **Kept as is:** no tuning of the projection on val yet. It stays the long-tail candidate source
  and the `proj_hybrid_score` feature (s13), where the ranker decides how much it helps; static
  fusion stays a control. Possible later val-only experiments (not scheduled): higher projection
  `lr`/`epochs`, W initialized at zero.

### D47. LightGCN training budget, sweep, paper-style projection, embedding diagnostics (2026-10-17)
- **Why:** reading `results/v0/training/lightgcn_val.csv` against the paper's setup. v0 has 93K fit
  interactions, so at batch 4096 one epoch is 23 steps and 300 epochs were about 6.9K gradient steps;
  the paper's MovieLens-1M run (715K interactions, 200 epochs) took about 35K. The inner metric was
  still rising at epoch 275 (0.0688) and early stopping never fired; BPR fell to 0.014 at `reg` 1e-5,
  i.e. memorization. Per-user val results also showed LightGCN and content-only hitting almost
  disjoint users (hit@10: 179 vs 155 users, 36 in common; union 300 of 2,578), so the fusion headroom
  is real even though D46 was null.
- **Decision (config only, same model code):** `models.lightgcn` `batch_size` 4096 -> 2048,
  `max_epochs` 300 -> 1500, `reg` 1e-5 -> 1e-4. Early stopping (`patience` 10 x `eval_every` 5)
  decides the real epoch count; 1500 is a cap matching the paper's step budget.
- **Sweep:** new stage `models.lightgcn_sweep` (`ml/models/lightgcn_sweep.py`), grid
  `models.lightgcn_sweep` = dims {64, 128} x regs {1e-4, 1e-3}, scored on the inner split only,
  one row per point in `results/<v>/training/lightgcn_sweep.csv`. It writes nothing to `artifacts/`
  and registers last, so `run all --until eval.val` never runs it. The user reads the table and edits
  `models.lightgcn`; the sweep is not an automatic selector (no val leakage into the choice).
- **Projection, paper-style:** `normalize_items` true -> false (the paper's additive fusion has no
  normalization, and D46 showed normalization alone explained the projection's shape); new
  `w_init: zero` (training starts exactly at raw LightGCN and learns only the text correction) and
  `w_reg: 1e-4` (0.5 * w_reg * |W|^2 per step). Both keys default to the old behaviour (xavier, 0)
  when absent, so older configs and tests still run. `meta.json` gains `w_frobenius` (0 means W never
  moved). The shuffled control is unchanged and remains the test of semantic gain.
- **Diagnostics:** `embedding_health()` in `lightgcn.py` (mean pairwise cosine over up to 2,000 warm
  items via the sum-of-unit-vectors identity, plus warm row-norm min/median/max) is written to the
  LightGCN and projection `meta.json`, logged, and warned on when mean cosine > 0.5 (s2.2 #9).
- **Not done:** rating-weighted or 4+-only edges, hard negatives, a warm-start evaluation stratum
  (cold val items removed from truth, for a like-for-like comparison with the paper). Candidates for
  the next val-only round if the gain from this round is small.
- **Alternatives considered:** keeping batch 4096 and only raising epochs (fewer, noisier steps);
  selecting the sweep winner automatically (rejected: keeps the human in the loop and the val split
  untouched); learning a bias with W (the paper has none; kept bias-free).

### D48. D47 results on v0 val; dim 128; temporal inner split; projection e_gcn dropout (2026-10-17)
- **Runs (Colab T4, the user's own account and Drive, pipeline rebuilt from scratch and byte-identical
  to the frozen v0: 109,454 / 10,987 / 4,190):** three `run all --until eval.val` passes and the
  6-point sweep (regs widened to {1e-5, 3e-5, 1e-4} before running, since 1e-4 lowered raw val).
- **Val NDCG@10 (2,578 users):**

  | model | D44 (dim 64, reg 1e-5, 300 ep) | dim 64, reg 1e-4 (best ep 320) | dim 128, reg 1e-4 (best ep 400) |
  |---|---|---|---|
  | lightgcn_raw | 0.0216 | 0.0203 | 0.0209 |
  | lightgcn_norm | 0.0160 | 0.0185 | 0.0199 |
  | static_fusion_0.7 | 0.0187 | 0.0198 | 0.0214 |
  | projection | 0.0146 | 0.0204 | 0.0208 |
  | projection_shuffled | 0.0151 | 0.0205 | 0.0209 |
  | itemknn | 0.0227 | 0.0227 | 0.0227 |

  Sweep (inner NDCG@20, per-user split): dim 64 -> 0.0690 / 0.0691 / 0.0706; dim 128 -> 0.0730 /
  0.0735 / 0.0759 for regs 1e-5 / 3e-5 / 1e-4. Embedding health: mean pairwise cosine 0.010 to 0.017,
  warm item norms 1.8 to 9.9 (popularity lives in the norm, which is why raw and norm differ).
- **Readings.** (1) Raw LightGCN sits at 0.020 to 0.022 on v0 under every config; the differences
  are inside the D44 bootstrap CI (about +/-0.004) and it ties ItemKNN. Treated as the model's ceiling
  on this sparse catalogue, not as a bug. (2) Early stopping now fires (best epoch 320 to 400), so
  the D47 "under-trained" hypothesis is closed. (3) The per-user inner split moved the wrong way:
  inner rose 2.6% to 7.5% while outer raw fell or stayed flat. The outer val is a global time cutoff
  with 11.6% cold items; the per-user split shares items and period with training. Model selection
  on it is not trustworthy. (4) Projection equals its shuffled control for the third time; its BPR
  ends at 0.006 because the frozen LightGCN already separates almost every fit pair, so W receives
  no gradient. This is structural in the paper's method, not a tuning matter. (5) Normalized LightGCN
  (+24%) and static fusion 0.7 (+14%) improved consistently; those rows carry coverage and tail
  exposure, so the config change is kept even though raw is flat.
- **Decisions.** `models.lightgcn.dim` 64 -> 128 (kept: equal on raw, better on norm and fusion,
  early stopping healthy). New `inner_split: temporal` (default `user` keeps the old behaviour):
  hold out the newest `inner_val_fraction` of the fit window by a global timestamp quantile, so
  early stopping and the sweep see the same temporal shift and cold items as the outer val;
  early-stopping `metric` ndcg@20 -> ndcg@10 to match the headline. New
  `models.projection.gcn_dropout: 0.3`: at each step e_gcn is zeroed for a seeded random 30% of
  items (scoring and the saved matrix use the full e_gcn), so W is trained on exactly the case it
  exists for. The shuffled control runs with the same dropout and stays the test of semantic gain.
  `lightgcn_sweep.regs` widened to {1e-5, 3e-5, 1e-4}.
- **Stop rule.** This is the last LightGCN/projection tuning round on v0. Whatever the next run
  shows is reported as final for v0; effort moves to the Week 5 gate (`recommend`, `serve.export`)
  and then the Phase 5 ranker, where the per-user oracle (D47: union hit@10 11.6% vs 6.9%) says the
  real gain is.
- **Not done:** a warm-start stratum (cold items removed from truth) for a like-for-like comparison
  with the paper's protocol; rating-weighted edges; hard negatives. Results files for these runs are
  on the user's Drive (`results_v0.zip`, `results_v0_dim128.zip`) and are committed when received.

### D49. Final v0 LightGCN and projection numbers (D48 config, Colab T4, 2026-10-17); tuning closed
- **Run:** `run all --until eval.val` on the user's Drive. The data stages `data.meta`, `data.canonical`,
  `data.reviews`, `data.clean` reran (about 40 min) although their fingerprints were unchanged: their
  outputs on Drive failed the D42 size check after the "Google Drive error" popups of the earlier
  session, i.e. an incomplete flush. They regenerated byte-identical counts. `kcore_split`, `eda`,
  `nlp.content` were up to date. This is the results set committed in `results/v0/val/` and
  `results/v0/training/lightgcn_val.csv`.
- **Temporal inner split works as intended.** Inner NDCG@10 peaks at 0.0250 at epoch 80 and declines
  after (early stop at 130), on 3,002 inner users; the outer val raw is 0.0212. The per-user split
  had reported 0.07 and kept rising to epoch 400. The proxy now moves with the outer metric.
- **Val NDCG@10, final v0 numbers:**

  | model | NDCG@10 [95% CI] | recall@20 | coverage@10 | tail share@10 | long-tail NDCG@10 | cold-item NDCG@10 |
  |---|---|---|---|---|---|---|
  | itemknn | 0.0227 [0.0189, 0.0270] | 0.0541 | 0.686 | 0.410 | 0.0123 | 0 |
  | content | 0.0191 [0.0154, 0.0225] | 0.0463 | 0.569 | 0.734 | 0.0203 | **0.0230** |
  | lightgcn_raw | 0.0212 [0.0176, 0.0250] | 0.0553 | 0.425 | 0.146 | 0.0057 | 0 |
  | lightgcn_norm | 0.0168 [0.0139, 0.0202] | 0.0465 | 0.822 | 0.588 | 0.0134 | 0 |
  | projection | 0.0207 [0.0170, 0.0246] | 0.0535 | 0.422 | 0.165 | 0.0059 | 0 |
  | projection_shuffled | 0.0206 [0.0170, 0.0245] | 0.0575 | 0.394 | 0.133 | 0.0053 | 0 |
  | static_fusion_0.7 | 0.0189 [0.0155, 0.0225] | 0.0492 | 0.813 | 0.586 | 0.0161 | 0 |

  Paired bootstraps (NDCG@10): itemknn vs lightgcn_raw +0.0015 [-0.0025, +0.0053] (tie);
  projection vs lightgcn_raw -0.0005 [-0.0022, +0.0012]; projection vs shuffled: 89 users better,
  88 worse, 2,401 equal. Long tail: content beats lightgcn_raw by +0.0145 [+0.0106, +0.0186].
- **Projection with e_gcn dropout: W learns, ranking does not move.** Training BPR rose from 0.006 to
  0.235, so W now receives a gradient; the shuffled control ends at 0.309 and a smaller W (Frobenius
  17.3 vs 21.4), so real text fits the fit pairs better than permuted text. That is the first
  in-training evidence of a semantic signal. On val nothing changes: projection ties raw LightGCN and
  its shuffled control on every stratum, and **cold-item NDCG@10 is exactly 0 for the projection**
  (1,380 users with cold truth; content gets 0.0230). Reason: a cold item's vector is W c_i alone,
  while warm items carry e_gcn with norms 1.8 to 9.0; the cold items never enter anyone's top 20.
  Additive fusion in the LightGCN space cannot recover cold items unless the two halves are put on a
  common scale, which is what the ranker (s13) does with separate features and what static fusion
  did with per-half normalization (and lost raw accuracy doing so).
- **Decisions.** (1) v0 LightGCN/projection tuning is closed, per the D48 stop rule; these are the
  final v0 numbers for the report. (2) The config stays as D48 (dim 128, reg 1e-4, batch 2048,
  temporal inner split, ndcg@10, gcn_dropout 0.3). (3) The report states the honest result: on a
  sparse 4,190-book catalogue, LightGCN ties ItemKNN; the paper's projection hybrid, tested with a
  shuffled-text control, gives no ranking gain and zero cold-item recall; content-only is the only
  model that ranks cold items and wins the long tail; the per-user oracle (D47) shows the headroom
  that a feature-level ranker should capture. (4) The projection artifact is kept as the
  `proj_hybrid_score` ranker feature and the training-loss gap vs shuffled is logged as its
  justification; it is not a candidate source for cold items on v0.
- **Not done (deliberately):** scale-matched fusion variants, warm-start stratum, rating-weighted
  edges. Next: Week 5 gate (`recommend` on these artifacts, `serve.export`), then Phase 5.
## 2026-10-17: Phase 3, review NLP (branch `nlp-phase3`, parallel session)

D49 is left to the LightGCN/projection session (its progress note reserves it for the final v0
numbers); this block starts at D50.

### D50. Phase 3 layout: stages, per-cutoff artifact sets, new shared modules
- **Stages** (`ml/stages.py`, registered after `eval.val` so `run all --until eval.val` never runs
  them): `nlp.sentences` (s11.1), `nlp.aspects` (s11.2-11.3), `nlp.sentiment` (s11.4),
  `nlp.centroids_val` and `nlp.centroids_serve` (s11.6). `models.lightgcn_sweep` stays last (D47).
- **Per-cutoff artifacts follow the D43 pattern:** one stage per artifact set sharing one builder,
  so each manifest carries its own `cutoff`. `eval_val` (t1) and `serve` are built now; `eval_test`
  (t2) is registered in Week 13 with the other test-mode artifacts. Folder:
  `artifacts/<v>/<set>/review_centroids/` with `meta.json`. The `serve` cutoff is `max(ts) + 1`
  of the sentence table, so the leakage check is uniform and the manifest records a number.
- **`ml/nlp/aspect_seeds.py` is not edited.** `data.clean` imports it, so any change (even a
  comment) would rerun the review stream on Colab (D41). It already held the s11.2 taxonomy, lexicon,
  prototypes and poles; the detector in `aspects.py` reads them. Lexicon growth during validation
  will need a deliberate rerun decision or a split of the lexicon into a new module.
- **New shared modules** instead of edits to finished ones: `ml/nlp/cutoffs.py` (artifact-set
  cutoffs, `rows_before`, `check_cutoff` raising `LeakageError` from `ml/eval/errors.py`) and
  `ml/nlp/parts.py` (`PartStore`: checkpointed `.npy` parts validated on resume, D37 style).
- **New config keys** under `nlp` (no finished stage declares `nlp`, so nothing reruns):
  `embedding_dim: 384` (asserted against every sentence artifact and the encoder output, s2.2 #8;
  kept out of `content` so `nlp.content`'s fingerprint is untouched), `chunk_rows: 200000`
  (embedding checkpoint size and the aspect/sentiment chunk, the s11.1 and s11.3 numbers),
  `sentiment_max_length: 128` (s11.4). Stages declare the exact `nlp.*` keys they read, so a
  sentiment setting never reruns the sentence embedding.
- **Alternatives considered:** one stage writing all three cutoff sets (rejected: one manifest
  cannot carry three cutoffs, and the evaluator checks the manifest); embedding to a memmap on Drive
  with periodic flushes (rejected: FUSE flushes are what lost outputs in D37/D38; parts are whole
  files); hashing the lexicon out of `aspect_seeds.py` now (deferred until validation changes it).

### D51. `nlp.sentences` details
- `pos` is the sentence's index among every sentence spaCy produced for its review and
  `n_in_review` that count, both **before** the 4-80 token filter, so the spoiler rule "last 30%
  of the review" (s11.7) refers to the review, not to the kept subset.
- `sent_id` equals the row index of `sentences.parquet` and of `sent_emb.f16.npy`; the stage
  asserts it, and every reader indexes the memmap by `sent_id`.
- The sentence table itself is a checkpoint in the work dir (with its counts), so a Colab
  disconnect during embedding resumes at the last good part without re-splitting.
- Embeddings are the content model's `encode()` (mean-pooled, normalized; the same loader as
  `nlp.content`, so tests inject the same stub) cast to float16 only at rest; readers cast chunks
  back to float32 before any dot product (s18.1).
- Zero usable sentences writes an empty `(0, dim)` matrix instead of failing (sample runs).

### D52. `nlp.aspects` details
- Detection is exactly s11.3: `hit = sim >= tau[a] or lexicon[a]`, `sim` the max cosine over the
  aspect's 8 prototypes, `tau` from `aspects.tau` with `aspects.tau_default` as fallback. The stage
  asserts `aspects.names` and `aspects.descriptors` equal the seed module's order, which is the bit
  and column index everywhere.
- Stored per sentence: `aspect_mask` and also `lexicon_mask` (uint16), so validation (s11.5) can
  measure the prototype branch and the lexicon branch separately without recomputing; `top_sim`;
  `darkness`, `pace`, `complexity_level` = cos(mean plus pole) - cos(mean minus pole), NaN unless
  the linked aspect hit.
- Prototype and pole embeddings are stage outputs (`artifacts/<v>/nlp/aspect_prototypes.npy`
  (12, 8, 384) and `descriptor_poles.npy` (3, 2, 384)), not recomputed by consumers: the sentiment
  stage declares them as inputs, and `serve.export` copies them into the bundle (s4.6). They do not
  depend on a cutoff, so they live beside `content/`, not in an artifact set.
- `results/<v>/nlp/aspect_checks.json`: hits, lexicon hits and prototype-only hits per aspect, the
  share of sentences with any hit, and the tau vector used. Descriptive counts only.

### D53. `nlp.sentiment` details
- Polarity model and inputs as s11.4 (DistilBERT-SST2, `p_pos - p_neg`, max length 128, only
  sentences with a hit). The model loader is a module function (`load_polarity_model`) so tests
  inject a stub, like `embed.load_encoder`.
- **Clause rule:** a sentence with >= 2 aspects is split on `nlp.clause_split_markers`
  (case-insensitive, markers escaped). Each clause is embedded, run through the same detector
  (prototypes + lexicon) and scored. An aspect of the sentence takes the mean polarity of the
  clauses that mention it; when no clause mentions it (or there is no marker), it keeps the
  whole-sentence polarity. `from_clause` records which path produced the value.
- **Output is one long table**, `aspect_polarity.parquet` (`sent_id`, `aspect` int8, `polarity`,
  `from_clause`), one row per (sentence, aspect) hit. Profiles (s11.8) and snippets (s11.9) need
  per-aspect polarity; a per-sentence column cannot hold the clause results.
- `results/<v>/nlp/sentiment_checks.json` reports mentions, mean polarity and the shares above
  `pos_threshold` / below `neg_threshold` per aspect, so the SST-2 overconfidence pitfall (s18.1)
  is visible before profiles are built. Scoring is checkpointed in parts of `nlp.chunk_rows`
  sentences.

### D54. Review centroids details
- Rating floor = `data.relevance_min_rating` (4, the s11.6 value) rather than a literal, so the
  centroid and the relevance definition cannot drift apart.
- Sums are accumulated in float64 through sparse indicator products (one float32 copy of each
  embedding chunk, no Python loop); rows are L2-normalized, zero when the item or user has no
  qualifying sentence. `meta.json` records the cutoff, floor, dim and the counts.
- `build_centroids` raises `LeakageError` on any sentence at or after its cutoff and `ValueError`
  when ids exceed `id_map.json` (s2.2 #2).

### D55. Annotation export is its own small stage; the guide is written before the data run
- **Stage `nlp.annotation_export`** (`ml/nlp/annotate.py`) writes
  `results/<v>/nlp/annotation/aspects_300.csv`: a seeded sample of `aspects.annotation_sample.total`
  sentences, `with_hit` of them with a predicted hit (new config key, s11.5's 300 / 200). Stage
  outputs live under the Drive roots, so the user copies the file to `docs/annotation/` and commits
  it; the brief's path is the repo copy. Predictions are not in the file (labellers must not see
  them); `nlp.validate_aspects` joins labels back to `sentence_aspects.parquet` by `sent_id`.
- `docs/annotation_guides/aspects.md` (s11.5 step 2) is written now so labelling can start as soon
  as the CSV exists: definitions, two examples per aspect (the prototype sentences), confusions
  (ending vs emotional_impact, pacing vs plot, complexity vs characters, tone vs humor) and a
  labelling process. Tone and complexity labels carry the opinion; intensity comes from the
  descriptor model, which the guide says explicitly.
- **Not built yet:** `nlp.validate_aspects` steps 4-5 (tau tuning and the P/R/F1 report) wait for
  the labelled file; the spoiler classifier (s11.7) and `nlp.onnx` (s10.3) are the next pieces.

### D56. Week 5 gate: serve artifact set, bundle export v0, real `litsense_core` MVP (2026-10-17)
- **Why now:** the gate item "MVP blend `recommend` on real artifacts; bundle export v0" was the only
  Week 5 item still open, and the teammate's backend still runs on `litsense_core.mock`. v0 tuning is
  closed (D49), so the artifacts to serve are fixed.
- **Serve artifact set (s3):** new stage `models.lightgcn_serve` (`ml/serve/export.py`) fits LightGCN
  on train + val + test for exactly the epoch count early stopping chose on `eval_val`
  (`artifacts/<v>/eval_val/lightgcn/meta.json: best_epoch`), cutoff = newest timestamp + 1 ms, same
  `LIGHTGCN_FILES` plus `embedding_health` in `meta.json`. No search on the full data: there is
  nothing to validate against (D43 pattern). `nlp.content` is time-independent and reused as is.
- **Bundle (s4.6, s12.7):** stage `serve.export` writes `artifacts/<v>/serve_bundle/` and
  `artifacts/<v>/serve_bundle.tar.gz`: `catalog.npz` (year, pages, series id/pos, genre CSR,
  primary genre, Bayesian rating, counts, rating std, log popularity, trending, is_longtail,
  n_reviews, author id), `catalog_text.json.gz`, `lgcn_item.f16.npy`, `lgcn_item_norm.f16.npy`,
  `content_item.f16.npy`, `item_nn.npy`, `trending_top.npy` (200), `author_index.json`,
  `series_index.json`, `genre_index.json`, `corpus_users.npz` (the 300 most active readers with
  their exact serve LightGCN vectors), `metrics.json` (s15.6, built from `results/<v>/val/`) and
  `manifest.json` (bundle version `<v>-<date>`, cutoff, shapes, sha256 per file, the ranking and
  serving config slice, and an explicit `absent` list). Files that need later phases (projection,
  review centroids, aspects, snippets, prototypes, spoiler model, ONNX, ranker, faiss index) are
  listed as absent and `litsense_core` treats them as optional. Both stages register after
  `eval.val` in `STAGE_MODULES`, so `run all --until eval.val` is unchanged; `run all --until
  serve.export` builds the bundle (about 4 minutes on a T4 beyond eval.val).
- **`litsense_core` (s12, s15):** `store.py` (`read_store`, singleton `load_store`, `health`,
  `download_bundle` over httpx with a `serve_bundle/`-only tar filter), `vector_index.py` (faiss
  when importable, numpy otherwise, same ids), `profile.py` (bucket from `ranking.buckets`, exact
  corpus CF vector or pseudo-CF from liked/picked/list items at weight 0.5, semantic vector from
  liked items' content, genre and author affinities with the s11.8 rating weights and a 730-day
  half life, series progress, seen set), `candidates.py` (cf, semantic, item_item, popularity,
  author, longtail-by-content; round-robin union; s13.2 filters incl. the series guard and the
  next-volume rule), `rank.py` (the s12.3 blend with pseudo-CF at half weight, then muted-genre
  override, MMR over `mmr_pool`, genre cap, exploration slot), `reasons.py` (s12.5 templates 1, 3,
  4, 5, 6; template 2 waits for aspects), `recommend.py` (every s15.1 function). Degradation
  (s15.3): `recommend(use_ranker=True)` returns `degraded=True` because no ranker is in the bundle;
  `semantic_search` matches titles and authors only until the ONNX encoder ships; `book_insights`
  returns counts without bars or snippets; `analyze_user_review` returns an empty analysis;
  `chat_turn` is rule-based (like <title>, by <author>, general ask). `ValueError` only for unknown
  ids or malformed state. The package still imports only numpy, pydantic and httpx
  (`tests/litsense_core/test_imports.py`).
- **`dev/dev_server.py`:** FastAPI wrapper with `/health`, `/recommend`, `/rails`, `/explain`,
  `/similar`, `/search`, `/chat`, `/onboarding/grid`, `/metrics`; fastapi is not a core
  requirement, install it ad hoc.
- **Tests:** `tests/litsense_core/test_bundle.py` runs the pipeline through `serve.export` on the
  tiny fixture with stubbed models, checks the manifest (sizes, cutoff, serve epochs = val
  best_epoch, metrics), then exercises the real package: cold reader, warming reader, exclusions,
  genre filter, bad id, corpus demo user, rails, similar, grid, search, entities, explain, chat,
  metrics, insights. Plus unit tests for `VectorIndex`, the round-robin union, rank scores and
  buckets.
- **Not done / decisions for later:** review centroids and the projection are not in the bundle yet
  (centroids join when aspects and snippets do; the projection only as the ranker feature of D49);
  RSS is reported by `health()` instead of measured in a subprocess at export; `serve_bundle.tar.gz`
  carries the date in the manifest, not the file name, so it can be a declared stage output;
  `onboarding_vec` and `aspect_vec` are present on `Profile` but None until Phase 3 lands in
  the bundle. The 512 MB serving budget is unaffected: v0 bundle is a few MB.- **Fix after the first real run (same day):** a corpus demo reader has no ratings in the
  ``UserState``, so ``n`` was 0, the bucket ``cold`` and the cf quota 0; the exact LightGCN vector
  was never used and every pick read "Popular with readers right now". Now ``corpus_users.npz``'s
  ``n_interactions`` counts toward the bucket when the exact vector is used, and a CF-sourced item
  with no similar liked title gets "Readers with a taste like yours loved this" (a template that is
  not in s12.5; added so the collaborative source has its own line). First real run: bundle
  v0-20261017, 13 files, 6.4 MB tar, serve LightGCN 80 epochs on 109,454 interactions, load RSS
  155 MB, faiss backend, `recommend` latency 3.9 ms.

## 2026-10-18: Phase 3, second block (branch `nlp-phase3b`): profiles, snippets, spoiler, ONNX

### D57. Item and user aspect profiles per cutoff (`ml/nlp/profiles_offline.py`)
- **Stages `nlp.profiles_val` and `nlp.profiles_serve`** (D43/D50 pattern, `eval_test` in Week 13),
  folder `artifacts/<v>/<set>/aspects/`: `item_aspects.npy`, `item_mention_rate.npy`,
  `item_pos_rate.npy`, `item_neg_rate.npy` (n_items, 12) float32, `item_descriptors.npy`
  (n_items, 3), `aspect_means.npy` (12,), `user_aspects.npy`, `user_aspect_conf.npy`
  (n_users, 12), `user_review_aspects.parquet` (u, i, review_row, ts, aspect, polarity,
  helpful_vote), `meta.json`; the serve set adds `aspect_topitems.json`.
- **s11.8 as written**, with these readings: `n_ia` counts distinct reviews (a review with two
  sentences on plot counts once); `mu_a` is the helpful-vote-weighted global mean over every
  mention before the cutoff, and a cell with no mention sits exactly at `mu_a` (pure shrinkage);
  `n_reviews_i` for the mention rate = reviews of the item with at least one kept sentence
  (reviews with no usable text cannot mention anything); `pos_rate`/`neg_rate` are judged on the
  review's mean polarity for the aspect, not per sentence; descriptors are shrunk toward 0 with
  the same `k` (sum / (n + k)) and clipped.
- **User explicit vectors** use time decay from the cutoff (the offline "now") and the new
  `aspects.explicit_conf_k: 3` (s11.8's constant). `aspect_topitems` uses the new
  `aspects.topitems: {n: 300, min_reviews: 5}` (s13.1's numbers).
- **Alternatives considered:** weighting `n_ia` by helpful votes too (rejected: s11.8 uses votes
  only inside `raw`); mention rate over all interactions of the item (rejected: it would punish
  books whose readers rate without writing).

### D58. Shared aspect math lives in `litsense_core/aspects.py`; `profile.py` untouched
- `shrink`, `decay`, `implicit_aspect_vec`, `explicit_aspect_vec`, `combine` are numpy-only
  functions used by the offline stage (tested to agree with the vectorized offline path) and ready
  for `build_profile`. `profile.py` was not edited (the LightGCN session owns it); wiring is one
  block in `build_profile` once the store loads `item_aspects.npy` and `aspect_means.npy`:
  implicit = `implicit_aspect_vec(store.item_aspects, store.aspect_means, rated ids,
  rating weight x decay)`, explicit = `explicit_aspect_vec` over `state.review_analyses`
  (polarity matrix with NaN for absent aspects, decay from `now`), then `combine(..., 
  explicit_weight=0.6, positive/negative from onboarding, pinned from overrides)`.
  `combine` scales the explicit weight by the per-aspect confidence (`0.6 * conf`) so one review
  cannot dominate; s11.8 leaves that interaction open.

### D59. Evidence snippets read the spoiler flag, not the regex
- `nlp.snippets` (`ml/nlp/snippets.py`, serve only) takes `sentence_spoiler.parquet` as an input
  and drops `spoiler_flag` sentences. With the regex-only spoiler model this equals the brief's
  interim regex rule; when the classifier is trained the stage reruns automatically (its input
  hash changes). Output `artifacts/<v>/serve/snippets/evidence_snippets.json.gz`:
  `{item: {aspect: {"pos": [...], "neg": [...]}}}`, each entry `text`, `review_row`,
  `helpful_votes`, `polarity`; limits from the new `aspects.snippets` key (s11.9's 3 / 2 / 40 /
  220). Duplicate texts within an (item, aspect, sign) are kept once.

### D60. Spoiler classifier follows the D20 opt-in pattern
- **`spoiler.train_on_goodreads: false` (default)** writes a regex-only `spoiler_lr.npz` (empty
  coefficients, threshold = `regex_boost`), so `nlp.spoiler_flags`, `nlp.snippets` and the bundle
  have one code path and nothing waits on a 1.3 GB stream. The user flips it to true to approve
  the download; the stage then streams `goodreads_reviews_spoiler.json.gz` as a `url:` input
  (never stored, D18) and trains as s11.7 says.
- **Sampling:** a seeded, class-balanced reservoir (`sample_size / 2` per class) over the whole
  stream, sentences filtered with `nlp.min/max_sentence_tokens` so the training distribution
  matches the Amazon sentence table; the holdout is `holdout_fraction` of *books*.
- **Threshold** is chosen on the classifier probability (s11.7 step 4); the regex and
  late-position boosts (new `spoiler.late_fraction: 0.3`, s11.7's "last 30%") are added at
  scoring time, so they only ever raise recall. `litsense_core/spoiler.py` holds the scorer used
  both offline (`nlp.spoiler_flags` -> `sentence_spoiler.parquet`) and online.
- **Not verified yet:** the real P/R/F1/AUC (needs the Goodreads run). On the fixture only the
  mechanics run (stub embeddings).

### D61. ONNX export details (`ml/nlp/onnx_export.py`, `litsense_core/encoder.py`)
- `torch.onnx.export(..., dynamo=False)`: the legacy exporter with `dynamic_axes`, opset 17,
  output `last_hidden_state` only, on a real `nn.Module` wrapper with gradients disabled (a plain
  callable makes the exporter treat the weights as constants and fail); `optimum` is not a
  dependency. Files: `artifacts/<v>/serve/onnx/{minilm.onnx, minilm.int8.onnx, tokenizer.json}`.
- **int8 is quantized per channel** (`quantize_dynamic(..., per_channel=True)`). In the container
  check on the 96 prototype sentences, per-tensor scales (the default) fell below both int8
  thresholds while per-channel scales passed them with the same file size and latency; the real
  numbers for the report come from `results/<v>/nlp/onnx_parity.json` after the Colab run.
- **Parity texts** (new `nlp.onnx_parity`): the 96 prototype sentences plus a seeded sample of
  real review sentences up to 200 (so `data:sentences.parquet` is the stage's input). fp32 below
  0.9999 is an error; int8 below its thresholds is logged with the instruction to set
  `serving.onnx_quantized: false` (a stage cannot edit config), both files are still written.
  One-thread latency of a short query is recorded against `max_latency_ms`.
- `MiniLMEncoder` keeps the s10.4 code plus `count_tokens`/`truncate`, so it satisfies the same
  `TextEncoder` protocol as the offline encoder. `onnx` joins the CI package list so the
  hand-built tiny-model test runs in CI (onnxruntime and tokenizers are core requirements).

### D62. First Colab run of the second block (CPU runtime, 2026-10-17): ship fp32 ONNX
- **Drive lost `sentence_aspects.parquet`** (0 bytes, the D42 failure mode); `nlp.aspects` was rerun
  on CPU in 93 s. Its hit count moved by 8 sentences out of 552,265 (fp32 CPU vs fp16 GPU prototype
  embeddings at the tau boundary), so `nlp.sentiment` is now stale and is rerun on the next GPU
  session; profiles and snippets refresh automatically after it.
- **int8 parity failed on real sentences:** on the 200 parity texts (96 prototypes + 104 sampled
  review sentences) the per-channel int8 model reached mean cosine 0.9894 and min 0.9517 against
  thresholds 0.99 / 0.98 (`results/v0/nlp/onnx_parity.json`). The brief's rule applies: ship fp32,
  `serving.onnx_quantized: false`. Cost: the fp32 file is about 90 MB against 23 MB and a short
  query takes about twice as long (still a few ms); the serve bundle's load RSS was 155 MB, so the
  350 MB budget holds. Both files stay in the artifact set; the bundle copies the fp32 one.
- Regex-only spoiler flags: 12,298 of 552,265 sentences (2.2%); snippets: 192,720 over all 4,190
  items after dropping 19,765 flagged mentions. Every aspect has at least 300 eligible top items.
### D63. Phase 3 artifacts wired into the bundle and `litsense_core` (2026-10-18)
- **Merge:** branch `nlp-phase3b` (D57 to D62) merged into main with no conflicts.
- **`serve.export` inputs (new, all declared, so the bundle reruns when any of them changes):**
  `serve/aspects/{item_aspects, item_mention_rate, item_pos_rate, item_neg_rate,
  item_descriptors, aspect_means}.npy` and `aspect_topitems.json`,
  `serve/snippets/evidence_snippets.json.gz`, `serve/spoiler/spoiler_lr.npz`,
  `serve/review_centroids/review_centroid_item.npy` (cast to float16),
  `nlp/aspect_prototypes.npy`, `nlp/descriptor_poles.npy`, and the ONNX files. Only the ONNX
  variant selected by `serving.onnx_quantized` is copied, always as `minilm.onnx`, with
  `manifest.onnx = {quantized, source, max_seq_length}` (D62: fp32 ships). The `absent` list is now
  `proj_item.f16.npy`, `ranker.txt`, `feature_list.json`, `content.faiss` and the unused ONNX
  variant. The config slice gains `aspects` and `ranking.overrides`. Because the export now depends
  on every Phase 3 stage, `run all --until serve.export` runs them (they are up to date on Drive
  except `nlp.sentiment`, which D62 left stale; it and its dependants rerun first).
- **`litsense_core.store`:** optional loading of every file above (`_load_phase3`), `aspect_names`
  and the per-aspect `aspect_tau` from the config slice, `SpoilerScorer.from_npz`, and the
  `MiniLMEncoder` over the bundled ONNX (a load failure leaves `encoder=None`, s15.3). `health()`
  reports `aspects`, `snippets`, `spoiler`, `encoder`.
- **`profile.py` (s11.8 user side, via `litsense_core.aspects`, D58 as written):** implicit vector
  from rated items' shrunk profiles with the same rating weight x 730-day decay, explicit vector
  and confidence from `state.review_analyses`, onboarding positive/negative aspects +-0.5, pinned
  aspects +0.5, clipped; `descriptor_pref` from rated items' descriptors plus onboarding
  descriptors; `onboarding_vec` from the encoder with the s12.2 blend 0.6 picks + 0.4 text.
- **`candidates.py`:** the `aspect` source (top-2 positive aspects' `aspect_topitems`, intersected
  with preferred genres when that keeps at least half the quota, interleaved); the `semantic`
  quota is split with `onboarding_vec` when present. All seven sources of s13.1 now exist.
- **`rank.py`:** `aspect_match` = cosine between the reader's aspect vector and the item's centered
  shrunk profile, each dimension weighted by 0.5 + 0.5 x explicit confidence (so implicit-only
  readers still get a signal); pinned aspects add `ranking.overrides.pinned_aspect_boost` x item
  aspect in post-processing.
- **`reasons.py`:** template 2 is live: an aspect the reader values (> 0.3) that the item is also
  praised for (> 0.3) gives "Readers praise its {label} ({pos_pct}% of mentions), something you
  value" with `pos_pct` from `item_pos_rate`; the evidence carries the item's aspect profile,
  mention and positive rates, the reader's top positive/negative aspects, and the best positive
  safe snippet for the matched aspect (else the item's most praised aspect). Reason order is
  otherwise the s12.5 order plus the D56 CF line.
- **`recommend.py`:** `semantic_search` adds `matched="semantic"` hits from the encoder after the
  title/author hits; `book_insights` returns real aspect bars and up to five snippets;
  `analyze_user_review` splits sentences with a regex, embeds with the encoder, matches prototypes
  with tau per aspect, takes VADER compound as polarity and counts spoiler flags with the bundled
  scorer; a free-text chat ask steers the semantic vector (s14.7 step 6).
- **Tests:** the bundle fixture now runs the data stages, `eval.val`, every Phase 3 stage with the
  NLP stubs and `nlp.onnx` with a hand-built 16-d Gather model standing in for MiniLM (the parity
  reference is that model, so the stage's own checks run unchanged), then `serve.export`; the
  real-package test adds onboarding text, pinned aspects, review analyses, semantic hits, insights
  bars and a spoiler count. `tests/litsense_core/conftest.py` registers every stage.
- **Not done:** `nlp.validate_aspects` steps 4 to 5 and the per-aspect tau (waits for the labelled
  CSV); the Goodreads-trained spoiler model (opt-in, D60); `eval_test` artifact sets (Week 13);
  the aspect source's offline recall check (`ml/ranking/candidate_recall.py`, Phase 5). With the
  real bundle the search and review analysis quality depends on the fp32 ONNX parity recorded in
  D62, which held.
### D64. First Phase 3 bundle run: memory over budget, demo readers without aspects (2026-10-18)
- **Run (Colab T4):** `nlp.sentiment` reran (215 s) after D62, then profiles, spoiler flags,
  snippets and `serve.export`: bundle v0-20261018, 27 files, 93.7 MB tar (the fp32 ONNX is 90 MB
  of it). `health()`: encoder, aspects, snippets and spoiler present; `recommend` 68 ms for a demo
  reader (aspect source and evidence added work; still under the 80 ms budget); `book_insights`
  bars and snippets real; `semantic_search` returned title hits first for a query whose words
  appear in many titles; `analyze_user_review` found plot and pacing and one spoiler sentence.
- **Problem 1, RSS 394 MB against the 350 MB budget** (D56's bundle loaded at 155 MB). The
  200K-entry snippets JSON parsed into Python dicts is the bulk of the increase. Fix: the export
  writes `evidence_snippets.jsonl` (line k = item k) plus `snippets_offsets.npy`; the store keeps
  only the offsets and reads one item's line with a seek (`SnippetFile.for_item`). The ONNX session
  also runs with `enable_cpu_mem_arena` and `enable_mem_pattern` off (one short query at a time, the
  arena only adds resident memory). To be re-measured on the next Colab run; if still above 350 MB
  the next lever is the int8 ONNX (23 MB) despite its D62 parity miss, as a user decision.
- **Problem 2, demo readers got no aspect reasons:** a corpus reader's ratings live in the corpus,
  not in the `UserState`, so the online implicit vector was empty. Fix: `corpus_users.npz` now
  carries each demo reader's offline explicit aspect vector and confidence from
  `serve/aspects/user_aspects.npy` and `user_aspect_conf.npy`; `build_profile` uses them when the
  state has no review analyses of its own (same D56 principle as the corpus LightGCN vector).
- **Noted, not changed:** `semantic_search` ranks title/author matches ahead of embedding hits by
  design (s12.6 mock parity); at k = 5 that can hide the semantic hits. Revisit with the ranker.
### D65. Second Phase 3 bundle run: RSS must be measured in a fresh process; aspect reasons centered (2026-10-18)
- **RSS:** the second load in the same Colab kernel read 516 MB, but that kernel still held the
  first store (394 MB with the eager snippets); CPython does not return freed arenas to the OS, so
  in-kernel readings after a reload are not a measurement. The s12.7 rule stands: measure in a
  fresh subprocess (`python -c "import litsense_core; load_store(...); health()"`), which the next
  Colab cell does. `recommend` latency fell to 24 ms with the lazy snippets.
- **Aspect reasons fired everywhere and said "humor" for thrillers.** Two causes. (1) The explicit
  vector (a reader's own review polarities, incl. the corpus vectors) is raw mean polarity, and v0
  polarity is mostly positive (aspect means 0.42 to 0.56), so every mentioned aspect looked "valued"
  at about 0.5; the implicit side is centered on the aspect means, so the two were not on the same
  scale. Now the explicit side is centered too, only where the reader mentioned the aspect
  (`conf > 0`). (2) Template 2 compared raw item values with the brief's 0.3, which almost every
  item exceeds. It now needs the reader's centered value > 0.15 and the item's centered shrunk value
  > 0.08 (about the 75th percentile of centered item values on v0; retune after the labelled
  validation). The `pos_pct` shown stays the raw positive-mention share, which is what the sentence
  claims.
- **Noted:** the humor snippet chosen for a thriller was a generic praise sentence; that is the
  prototype-only false-positive rate flagged in D55, fixed by the tau tuning after labelling, not
  by the serving side.- **Measured after D64 + D65 (Colab, fresh subprocess, 2026-10-18):** RSS 229 MB after `load_store`
  and 234 MB after 30 `recommend` calls (budget 350; D56's bundle was 155 MB, so the Phase 3 files
  and the fp32 ONNX cost about 75 MB). `recommend` 24 ms. Three demo readers now get thriller-shaped
  aspects (pacing 0.45, dialogue 0.37, ending 0.30 for the first) and reasons such as "Readers
  praise its pacing (62% of mentions), something you value", mixed with the CF line where no aspect
  clears the thresholds. Bundle v0-20261018: 28 files, 93.7 MB tar.
### D66. Phase 5 ranker: val bundle, feature table, LambdaRank, val ablation, ranker in the serve bundle (2026-10-19)
- **Why now:** D47 showed LightGCN and content hit disjoint users (oracle union hit@10 11.6% vs 6.9%
  for either alone) and D49 closed LightGCN tuning; the ranker is the s13 answer to that headroom.
  All seven candidate sources and the aspect artifacts existed, so nothing blocked it.
- **Val bundle (`serve.export_val`, new):** ranker training must run the real `litsense_core` path
  (profile -> candidates -> filters -> features) on artifacts at t1, not on the serve set. The
  bundle writer is now one function (`write_bundle`) used by both stages; the val bundle takes the
  eval_val LightGCN, aspect profiles and review centroids, catalogue statistics from train rows only,
  no snippets / spoiler / encoder / ranker (listed as absent), and every reader with a train
  interaction as a corpus user (exact LightGCN vector). `nlp.profiles_val` now writes
  `aspect_topitems.json` too, so the aspect source exists at t1 (its serve lists would leak).
- **Features (`litsense_core/features.py`):** the 27 features + 7 source flags of s13.3 in the
  brief's order, NaN where a signal is missing. Three interpretations the brief left open:
  `semantic_recent_sim` uses the mean content vector of the item_item anchors (last 3 liked);
  `aspect_match_recent` uses the implicit aspect vector of those anchors; `review_centroid_sim`
  uses the reader's offline review centroid when they are a corpus reader (now shipped in
  `corpus_users.npz`) and otherwise the rating-weighted mean of their liked items' centroids, a
  proxy until readers have reviews of their own. `proj_hybrid_score` is NaN on v0 (projection
  absent, D49). `assemble(..., expected_cutoff)` is the leakage guard the brief asks for.
- **Training data (`ranking.build_training`):** s13.4 as written, with one addition: a seeded half
  of the users (`ranking.exact_cf_share: 0.5`) are passed as corpus readers (exact vector,
  `cf_available` 1), the rest as new readers (pseudo-CF, 0.5). Serving sees both regimes (300 demo
  accounts vs everyone else) and the ranker should not learn one calibration only. Held-out users
  (20%) keep every candidate so the ranker and the blend are compared on identical lists; train
  users keep positives + 30 sampled negatives each. The candidate-recall diagnostic of s13.1
  (`results/<v>/ranker/candidate_recall.csv`: recall and unique contribution per source, union
  before and after filters) is written by the same pass.
- **Training (`ranking.train`):** the s13.5 recipe from `ranking.lgbm`, early stopping on the
  held-out users; saves `ranker.txt` + `feature_list.json`, gain importances (CSV + PNG), and
  `ranker_vs_blend.csv` (ranker vs blend vs cf-score-only on the held-out users, with the paired
  bootstrap). Metrics use each user's true number of relevant val items, so positives the
  candidates missed count as misses for every row.
- **Val ablation (`ranking.ablation_val`):** M4..M8b of s13.7 as feature-group masks, same rows,
  same held-out users, CIs and paired bootstraps against M4 and the blend. Post-processing is not
  in this offline table (it reorders for diversity; the evaluator's beyond-accuracy metrics measure
  it). Not in `run all --until serve.export`; the Colab cell runs `--until ranking.ablation_val`,
  which includes the bundle.
- **Serving:** `serve.export` now takes `ranker.txt` + `feature_list.json` as inputs and ships them;
  `ArtifactStore` loads the booster only when the shipped feature list equals `features.FEATURES`
  (a column mismatch would score silently wrong). `recommend(use_ranker=True)` scores candidates
  with the ranker, falls back to the blend on any failure and reports `degraded=True` only then;
  `evidence.signals.ranked_by` says which one ranked. `chat_turn` and `home_rails` use the same
  path. lightgbm was already a core requirement.
- **Gate (Week 9/10):** the ranker must beat the blend on held-out val NDCG@10 and the ablation must
  show M2 vs M4 vs M6 with CIs; both are produced by the stages above and are read off
  `results/v0/ranker/` after the Colab run. If the ranker loses to the blend, the brief's rule holds:
  report it and keep the blend as default (that is one config switch away: drop the ranker inputs).- **Measured (2026-10-19, Colab CPU):** held-out ranker NDCG@10 0.140 vs blend 0.032 (paired +0.108,
  CI [0.071, 0.147]); M6 vs M4 NDCG -0.002 (CI includes 0), recall@20 +0.045 (CI above 0); candidate
  recall union 0.288, 0.168 after filters. Serving: 63.7 ms, RSS 358.8 MB on Colab (lightgbm imports
  pandas and sklearn when installed; about 260 MB expected on the core-only backend image). Details and
  open items in `docs/progress.md`.
### D67. Ranker evaluation protocol: cross-fitting and no books from the future (2026-10-19)
- **What the first `ranking.eval_val` run showed:** ranker NDCG@10 0.0728 on all 2,578 val users vs
  LightGCN 0.0212, blend 0.0124. Not trusted, for two reasons found while reading it.
- **Leak 1, training users scored by their own ranker.** `ranking.train` fits on the val labels of
  the 80% train-split users; the first eval scored every val user with that one ranker, so 591 of
  the 2,578 users were scored by a model that had seen their positives. Fix: `ranking.eval_val` now
  splits users into `ranking.eval_folds` (5) seeded folds and scores each fold with a ranker fit on
  the other folds' training rows. The shipped `ranker.txt` is unchanged (it may use every user).
- **Leak 2, val-window releases in the candidates.** The catalogue is static, so a book first
  published after t1 reaches the candidates through content, author and aspect sources, and
  `item_age_years` (cutoff year minus publication year) goes negative only for exactly those books;
  the ranker learned it (`item_age_years` was the top feature by gain). Serving never sees a
  negative age. Fix: offline candidate lists drop books dated after the cutoff year (training and
  evaluation), and the feature is clipped at 0 everywhere. Same-year releases after t1 remain (year
  granularity); noted as a residual.
- **Consequence:** `results/v0/ranker/` from the first run is superseded; the numbers in
  `docs/progress.md` under D66 are re-measured by the rerun. `ranking.build_training` and
  everything after it rerun (their code changed).
## 2026-10-19: Phase 6 GenAI layer (branch `genai-phase6`; D68 to D71)

### D68. LLM client: config source, result object, availability rule, cache of rejected text (2026-10-19)
- **Config source.** `litsense_core` never imports `ml`, and the backend installs the package without the repo,
  so the `llm:` block reaches the client three ways, in priority order: the backend calls
  `litsense_core.configure_llm(mapping, on_event)`; else `load_store` installs a client from the bundle
  manifest's `config.llm` slice (`serve.export` now copies the block, never the key); else, for bundles older
  than this phase, `ml/config.yaml` is read by path when the file exists; else the defaults (LLM off).
  The client is a process-wide singleton like the store (`litsense_core.llm.get_client()`).
- **Result object instead of `BaseModel | None`.** `client.text()` and `client.json()` return
  `LLMResult(text, data, cached, fallback_reason, latency_ms, attempts)`. The brief's bare `None` cannot say
  why (`disabled`, `not_configured`, `no_api_key`, `budget`, `rate_limited`, `timeout`, `provider_error`,
  `invalid_json`), and the callers need that to decide `degraded` (below), the backend needs it for logs, and
  the audit scripts count it. Nothing raises for model trouble; the one exception is `PrivacyError` on a user
  identifier in the variables (a programming error, s16.2 asks for the raise).
- **Availability.** `health()["llm"]` is true only when `llm.enabled`, the model id is not `SET-ME`, and the
  environment variable named by `llm.api_key_env` is set (the `fake` provider needs no key). A client that is
  not available never constructs an adapter, so no SDK import and no network. The key is read from the
  environment at call time and never stored, logged or written.
- **Rate limiter does not sleep online.** The token bucket (`rpm_limit`) returns `rate_limited` at once on the
  serving path and the template answers; the offline synthesis stage switches the client to a blocking
  acquire. The brief's backoff (min(8, 2^attempt) + jitter) applies to 429 and 5xx responses only.
- **LLM failures are not `degraded`.** `explain()` and `chat_turn()` report `method`; the template answer is a valid result, so the LLM never sets `degraded` (s15.3 degradation stays with the ranker). `ReasonResult.failed` separates "the LLM was available and produced nothing accepted" (provider error, budget, timeout, rejection) from "off, unconfigured or disabled by the reader" for the audit counts only.
- **Cache of raw text.** The sqlite cache stores the provider's raw answer before validation (keyed by prompt
  id, prompt version, model, temperature, max tokens and the canonical JSON of the variables, TTL
  `cache_ttl_hours`). A rejected answer therefore stays rejected until the TTL expires or the evidence changes;
  caching after validation would hide the rejection rate the audit measures.
- **`openai_compat` is a stub** that reports `provider_error` ("not implemented"). Wiring a local Ollama
  endpoint needs the user's decision on the endpoint; httpx is already in core, so no new dependency.
- **Prompt files** are the brief's v1 texts verbatim (`litsense_core/llm/prompts/*.v1.md`), parsed once.
  The daily request counter lives in the same sqlite file (table `budget`), or in a callback the backend
  passes (`budget_callback`) for multi-process deployments.
- **Stage order fix found on the way:** `nlp.validate_aspects` (last commit on main) declares the labelled
  CSV as a `raw:` input, so `run all --until ranking.eval_val` (the bundle test) failed with "missing input"
  before this branch. It is now registered last in `STAGE_MODULES`, after `lightgcn_sweep`, so no `--until`
  target reaches it; it is run by name as its docstring says.

### D69. Explanations, onboarding, entities and chat on top of the LLM-off code (2026-10-19)
- **Module names clash with public functions.** The brief names the modules `explain.py`, `chat.py`,
  `onboarding.py`, `entities.py` while `litsense_core.explain` etc. are the public functions. The package
  imports the modules first and binds the functions last, and `recommend.py` reaches the modules through
  `importlib.import_module`; nothing in the public surface changed.
- **Template paths unchanged.** With the LLM off, `explain`, `parse_onboarding` (now `onboarding.keyword_parse`)
  and `resolve_entities` return exactly what they returned before; the Phase 4 bundle test asserts the same
  strings. `chat_turn`'s rule path keeps the Phase 4 behaviour for the tested messages and adds the brief's
  cases ("why the second one?", "something else", "more by that author", modifier words, year and length
  filters, chips, clarification on an ambiguous title, the empty message as a plain re-run per the contract).
- **Validators** (`litsense_core/validators.py`): one case-insensitive alternation over catalogue titles with
  two or more words or seven or more characters (so "It" and "Dune" are skipped as common words, which the
  brief allows); numbers must appear in the evidence JSON (ints, floats to three decimals, and rates in
  [0, 1] as percentages); spoilers through the bundled scorer (classifier plus regex when an encoder is
  loaded, regex alone otherwise); word limit; markdown lists, emphasis, headings and URLs; a quoted span of
  twelve or more characters must be a substring of the evidence snippet. Rejection codes are returned
  (`Explanation.rejections`) for the backend's logs.
- **The main feed never calls the LLM.** A `use_llm` keyword on `recommend()` was in the first plan; the product owner rejected it (s15.1 says the feed makes zero LLM calls and the backend relies on the 80 ms budget). `RecItem.reason` stays the template; LLM text comes only from `explain()` per item on demand and from `chat_turn`.
- **Entities.** rapidfuzz with the brief's thresholds (WRatio >= 88 with a 5-point margin for titles,
  token_sort_ratio >= 85 for authors; several authors of one title use the hint, else popularity) serves the
  chat and onboarding paths; "like Andy Weir" resolves the author after a title miss. The public
  `resolve_entities` keeps its Phase 4 difflib behaviour so author-mode search does not change under the
  backend.
- **LLM intent parse is guarded.** Anchors the LLM returns must appear in the message (normalized substring);
  a similar_to answer whose anchors were all invented is discarded and the rule parser decides. Deltas are
  clipped and filtered to the vocabularies, genres to the catalogue labels, `target_index` to the shown list.
- **Chat results take one path.** Author anchors return the author's unseen books in series order (as Phase 4
  did); every other intent runs the shared candidates -> filters -> ranker/blend -> post-processing path with
  the session applied to the profile: anchors into the item_item source and the semantic vector, aspect
  deltas at 0.5 weight, descriptor deltas added, the free-text query blended into the semantic vector when
  the bundle has an encoder, "shorter"/"longer" as page limits from the anchor's pages (300 when unknown).
- **Chat reply validation.** The LLM reply is used only when its `book_id`s equal the results in order and
  the intro, the follow-up and every reason pass the validators with `chat_max_words`; a reason may name the
  reader's similar liked books but not the item's own title (the app renders it). Otherwise the template intro
  and template reasons answer and `method` says `template`.
- **Onboarding.** The LLM profile is filtered to catalogue genre labels (case-insensitive), the twelve aspect
  names and the three descriptors (clipped), themes capped at six; named titles and authors resolve through
  `entities` and resolved titles join the picks. `named_authors` keeps the canonical author label; the profile
  builder does not use it yet (favourite authors still come from ratings), noted as a follow-up.
- **Privacy.** Chat sends the message, a session summary (titles, deltas, filters) and the shown titles; never
  ids, history or anything from `UserState` beyond the profile's evidence. A `PrivacyError` anywhere (an email
  inside a snippet, say) falls back to the rule or template path.

### D70. Offline review synthesis and the bundle (2026-10-19)
- **Stage `genai.synthesis`** (`ml/genai/synthesis.py`) reads the serve aspect rates, the safe snippets and
  `books.parquet`, selects the `genai.synthesis.top_n` books by review count, builds the brief's praised and
  criticised lists (s14.8 thresholds, in `genai.synthesis`), calls `review_synthesis` on `llm.model_quality`
  through a client with the blocking limiter, validates each summary (only the book's own title; numbers
  from the prompt; regex spoiler scorer; `max_words`) and writes the stats template otherwise. Rows accumulate
  in the stage's work dir, so a run stopped by the daily budget fails loudly (no manifest) and the next run
  resumes. Output `artifacts/<v>/serve/genai/review_summaries.jsonl` + `meta.json`.
- **Skippable.** With `llm.enabled: false`, no key, or `model_quality: SET-ME` the stage writes an empty file
  and records the reason; `serve.export` then keeps `review_summary: null` for every book, which is what the
  v0 bundle will carry until the user sets the model id and key.
- **`serve.export`** takes the JSONL as an input and fills `catalog_text.review_summary`; the manifest records
  `review_summaries` (count) and the `llm` config slice (D68). Its fingerprint changes, so the next Colab
  `run all --until serve.export` reruns `genai.synthesis` (seconds when skipped) and the export (fast).
  `ArtifactStore.catalog.review_summary` serves it through `book_insights`. The val bundle has none.
- **Never a feature** (s1.5): summaries live only in `catalog_text`.

### D72. Gemini model ids and timeout from the first real call (2026-10-19)
- **Model ids.** `models.list` with the user's key (2026-10-19) shows `gemini-3.5-flash-lite` as the newest
  Flash-Lite and `gemini-3.5-flash` to `gemini-3.8-flash` as Flash (no Lite above 3.5), plus the moving
  aliases `gemini-flash-lite-latest` / `gemini-flash-latest`. Pinned: `llm.model: gemini-3.5-flash-lite`
  (online roles). Aliases are not used because the cache key and the audit results must name a fixed model.
- **Synthesis model.** The brief wants Flash for offline synthesis. On the free tier `gemini-flash-latest`
  answered 429 three times and `gemini-3.5-flash` returned no valid JSON within `max_output_tokens: 200`
  on three attempts (its JSON failed the `Summary` schema each time; not diagnosed further to save the
  free-tier quota, a thinking model spending output tokens is the likely cause), while
  `gemini-3.5-flash-lite` answered correctly first time. `llm.model_quality` is therefore Flash-Lite too
  until the user wants to pay for Flash or raise the synthesis token budget; one config line to change.
- **Timeout.** The API refuses `http_options.timeout` under 10 s ("Minimum allowed deadline is 10s"), so
  `llm.timeout_s` is 10 instead of the brief's 8; the s15.4 "typical" budgets still hold (explain 2.9 s
  cold, 0.5 ms cached; chat turn two calls, 2.4 s to 4.2 s).
- **First Colab synthesis run (2026-10-20, 00:00 IST):** flash-lite answered 503 (overloaded) and 504
  (our 10 s deadline expired server-side) for the first books, each book then burned three retries and
  the stage showed no progress for 50 minutes. Changes: 504 is classified as a timeout (no backoff);
  the stage uses `genai.synthesis.timeout_s: 40` (offline can wait; the online 10 s stays); rows are
  flushed per book and a progress line is logged every 50 books. The run was interrupted and restarted.
- **Second Colab run (CPU runtime, 20:00 UTC):** 4 LLM summaries, then a persistent 429 from book 5 on
  (the free-tier quota for flash-lite), and two rejections `title:Mystery` because the catalogue has a
  book called "Mystery". Fixes: the stage stops on `rate_limited` like on `budget` instead of writing
  template rows that a resume would never retry; rows that fell back for a transient reason are dropped
  on resume; five consecutive timeouts or provider errors stop the run; the title matcher skips one-word
  titles under ten characters (two-word titles still match). Whether the quota is per day or per minute
  is read off AI Studio's Rate Limit tab; the paid tier (about $0.40 for the 1000 summaries at the
  listed flash-lite prices) is the fallback if the free tier cannot carry the batch.
- **Quota facts (third and fourth runs):** the free tier gives flash-lite 15 requests per minute and
  500 per day per project (the first project's 500 went on retries); the token bucket bursts up to
  twice `rpm_limit` in the first minute, so the stage runs at `genai.synthesis.rpm_limit: 7`. A 429
  whose quota id says `PerDay` is reported as `quota` and stops the run; a plain 429 makes the stage
  wait 65 s and retry the same book. A second Google account's project carries a second 500.
- **Validator false positives seen in the first 114 real summaries:** `title:The Plot` (a catalogue
  book called "The Plot" matched the phrase "the plot" in about a third of the summaries) and
  `spoiler` from the weak regex patterns "the ending" / "in the end" (a summary saying "some
  criticise the ending" reveals nothing). Fixes: the title matcher drops a leading article before the
  length rule, so "The Plot", "The Help" and "The Martian" are skipped like one-word titles; generated
  text is scored without the four weak patterns (`validators.WEAK_SPOILER_PATTERNS`), the classifier
  unchanged. Rows now carry the raw LLM answer so a resume re-validates earlier rejections without a
  call; rejected rows without it are retried.
- **Smoke test** (real key, fake 12-book store, Docker): `health()["llm"]` true; `explain`, onboarding and
  the three-turn chat demo all `method="llm"` with no validator rejections; the deadline error before the
  fix degraded every call cleanly to the templates (`fallback_reason="provider_error"`), as designed. The
  key was passed to the container as an environment variable only and is being rotated by the user.

### D73. Intent gold set: a plain labelling sheet, and the rule parser extended from it (2026-10-20)
- **Sheet instead of JSON.** The brief has the user write the gold JSON by hand; the labellers are
  non-technical, so `eval_intent --draft` writes a sheet with one plain column per question (intent
  word, book or author named, points back to a result, which one, shorter/longer, mood words, years,
  genres, "something else", titles shown before) and `--from-sheet` converts it to the gold CSV,
  naming any row with an unknown word. Guide: `docs/annotation_guides/intent.md`. The labels stay
  human: no model wrote or suggested a gold value beyond the prefilled intent word they had to
  confirm or correct (five were corrected).
- **The set.** 69 of 70 candidates kept (`docs/annotation/intent_candidates.csv` filled,
  `docs/annotation/intent_50.csv` generated). Labeller notes flag the known ambiguities: a bare
  year constraint after results ("nothing older than 2010") is `refine` in the guide but `filter`
  for "published before 1980"; page limits have no sheet column and were marked "shorter".
- **Rule parser.** Scored on the hand-built test store, the Phase 4 rule parser got 57% of intents
  (out_of_scope for every filter and mixed-opinion message, "more literary" taken as an author).
  Its regex families were extended for the message shapes the set contains: genre words and
  synonyms with only/no/exclude, page and year ranges, "filter only" detection, mixed opinion
  (a catalogue title plus praise and fault clauses, deltas from the clauses), "what else did X
  write", "author of the Nth one", "in the vein of", "X vibes", "loved X, what next". Result on the
  same set and store: 97% intent, 0.97 field accuracy. That number is in-sample for the rule
  parser (the patterns were written looking at these messages) and is reported as such; the LLM
  parser's number from the real bundle is the fair one for the report, and the test store lacks a
  Horror genre, so one filter row cannot resolve there.

### D74. Backend boot check: lightgbm needs libgomp, the ranker failure is now visible (2026-10-20)
- **Check.** `dev/dev_server.py` booted in a fresh `python:3.12-slim` container with only
  `requirements-core.txt` (plus fastapi/uvicorn) from an extracted bundle (the tiny pipeline bundle;
  the real one is on Drive): import plus `load_store` 1.9 s, every route answered, RSS 150 MB with
  the tiny bundle (the v0 bundle is expected at 230 to 260 MB, D66).
- **Found.** `health()["ranker"]` was false there: lightgbm's shared library needs the OpenMP
  runtime `libgomp.so.1`, which the slim image lacks, and `_load_ranker` swallowed the `OSError`
  (s15.3 says degrade, not raise), so the blend ranked and nothing said why. With
  `apt-get install libgomp1` the ranker loads and `degraded` is false.
- **Changes.** `ArtifactStore.ranker_error` and `health()["ranker_error"]` carry the reason (missing
  files, feature-list mismatch, or the exception), and a `litsense_core` logger warns once at load.
  The contract's new section 1b documents the GitHub Release publishing steps, the boot code, the
  core-only environment, the libgomp line for Dockerfiles, and the `GEMINI_API_KEY` switch. Test
  images and Colab have libgomp, which is why the suite never saw this. scipy is pulled in by
  lightgbm's own metadata; the core still imports nothing outside its allow-list.

### D71. GenAI evaluation scripts (2026-10-19)
- **Intent gold set** (`ml/genai/eval_intent.py`): CSV `id, message, session_json, gold_json`; sessions name
  titles, resolved on the scoring bundle, so the file does not depend on item ids. Metrics as s14.9: intent
  accuracy, anchor resolution accuracy after `entities` (an author for `more_by_author`, none for
  `explain_recommendation`), filter exact match, sign accuracy of aspect and descriptor deltas, and the mean
  over eight fields. Both parsers are scored; `llm_fallbacks` counts rows where the LLM answer was discarded.
  `--draft` writes 70 hand-written candidate messages (no LLM involved) for the user to choose 50 from and
  label by hand; `docs/annotation/intent_example.csv` shows the format on six rows.
- **Explanation audit** (`ml/genai/audit.py`): `export` samples explanations for the bundle's corpus demo
  readers, half LLM and half template when the LLM is available, into `docs/annotation/explain_audit.csv`
  with blank rater columns; `score` computes rates per method by majority vote and Fleiss' kappa per yes/no
  question over the three filled copies; `rejections` generates N LLM explanations and reports validator
  rejection rates by type (the automated hallucination proxy). Demo readers stand in for "evaluation users"
  because the serve bundle carries no other histories; the Week 13 run can pass val-user states instead.

### D75. Bug fix in litsense_core: the star in the similar-books reason (2026-10-20)
- `similar_books` built its reason with a double-encoded star (the three UTF-8 bytes of the star
  read as cp1252 and encoded again, so readers saw three junk characters): the character was
  correct in the Week 5 commit (5bb75e3) and came back mangled from a later Phase 6 commit, the
  Windows `Get-Content`/`Set-Content` round trip the environment notes warn about. A scan of the
  whole repo (outside `reference/`) for double-encoded sequences and U+FFFD found no other case.
  Restored to `★`, the same character `reasons.py` and the mock use. One character, no behaviour
  change; `tests/service/test_routes.py` asserts the reason ends in `rated 5★`. This is the only
  change to `litsense_core` on the `serve-api` branch.

### D76. The model service: one route per contract function (2026-10-20)
- **Package.** `service/` at the repo root (FastAPI), separate from `litsense_core` so the core
  keeps its dependency allow-list and its import test. `dev/dev_server.py` stays the bare
  reference; the service is what deploys.
- **HTTP mapping.** `/v1/<function name in kebab case>`. Functions that take a `UserState`,
  filters or lists are POST with a JSON body whose fields are the function's arguments
  (`{state, k, filters, exclude, use_ranker}` for `recommend`); argument-free or id-only reads are
  GET (`onboarding-grid?n=`, `book-insights/{id}`, `metrics-report`, `catalog`). GET with the
  state in the query string was rejected: a power reader's state is tens of kilobytes. Responses
  are the `litsense_core.schemas` classes as the functions return them, never redefined; only
  the request wrappers, the catalogue page and the error envelope are new models. Wrappers
  forbid unknown fields, so a misspelt key is a 400 instead of a silent default.
- **Catalogue (contract section 7 item 1).** `GET /v1/catalog?offset&limit` (limit at most 1000)
  returns `{bundle_version, total, offset, limit, items: CatalogBook[]}` in `book_id` order and
  `GET /v1/catalog/{id}` one row, built from the loaded store with the contract's proposed
  `CatalogBook`. The store does not keep `isbn13`, so the service reads it once from the bundle's
  `catalog_text.json.gz`; the export's `-1` for unknown year or pages becomes `null`; `series_id`
  is the bundle's series label. `bundle_version` is the backend's reseed signal. No `catalog()`
  function was added to the core (the brief keeps the core untouched).
- **Errors.** One envelope, `{"error": {"code", "message", "request_id", "details"?}}`, for every
  non-2xx. Validation errors are 400 (`validation_error`), not FastAPI's 422, as the brief asks;
  `details` keeps `loc`, `msg` and `type` and drops the submitted input. Every `ValueError` from
  the core is 400 `bad_request` (contract section 5), including pydantic's own `ValidationError`,
  which is a `ValueError` subclass; anything else is 500 `internal` with a generic message and
  the request id, the traceback only in the log. An unknown id in `/v1/catalog/{id}` is a 404
  (a resource lookup), while an unknown `book_id` passed to a contract function stays a 400.

### D77. Auth, request ids and logs (2026-10-20)
- **Auth.** One shared secret, `SERVICE_API_KEY`, sent as `X-API-Key` and compared with
  `hmac.compare_digest`. Required on `/v1/*` and `/metrics`; `/health`, `/ready`, `/docs` and
  `/openapi.json` stay open (the platform probe sends no headers; the docs hold no secrets).
  The check runs in ASGI middleware before the body is read, so an unauthenticated caller gets
  401 before any parsing and cannot probe routes. Boot refuses to start with an empty key unless
  `ALLOW_NO_AUTH=true`; a forgotten env var must not expose the model. A bearer token or mTLS
  were considered and are more than two services on one platform need.
- **Request id.** `X-Request-Id` is accepted when it matches `[A-Za-z0-9._:-]{1,64}`, else a
  uuid4 hex replaces it; the response always carries it, the envelope repeats it, and a context
  variable copied into the worker threads puts it on every log line of the request, including
  the core's LLM events.
- **Logs.** JSON lines on stdout: one access line per request (method, route template, status,
  duration, request id; probes at DEBUG) and one `llm_call` line per LLM event with a fixed field
  list (prompt id, model, latency, cached, ok, fallback reason, attempts). Query strings, bodies,
  `UserState`, chat messages, review text and prompts are never logged; a test sends marker
  strings and checks they appear in no log line. `httpx`, `httpcore`, `urllib3` and the Gemini
  SDK loggers are held at WARNING so request URLs cannot reach the log.

### D78. Boot, readiness and the LLM client (2026-10-20)
- **Port first.** The lifespan starts the boot in a background thread and returns, so uvicorn
  binds the port as soon as the app is imported and `/ready` answers 503 (with `Retry-After`)
  until the bundle is loaded and warm; `/v1` calls get 503 `not_ready`. `/health` is always 200
  with the core's `health()` once loaded, plus `ready`, boot timings (`download_s`, `load_s`,
  `warm_s`), attempts, the last error and process facts. Render's health check is `/ready`.
- **Bundle.** An extracted bundle in `BUNDLE_DIR` (or its `serve_bundle/`) is used as is;
  otherwise `BUNDLE_URL` is fetched through `litsense_core.download_bundle` into a staging folder
  and moved into place, so an interrupted download can never leave a half bundle that the next
  start would try to load; the 94 MB tar is deleted after extraction.
- **LLM client.** Before `load_store`, the service reads the manifest's `llm` slice (the repo YAML
  for older bundles), sets `cache_path` to `BUNDLE_DIR/llm.sqlite` (the bundle's relative
  `.cache/` is not writable for the non-root user) and calls `configure_llm(slice, on_event)`, so
  the event hook feeds logs and Prometheus counters. The key stays in the environment, read by the
  core only. `LLM_PROVIDER=fake` exists for tests and demos.
- **Warm-up.** One cold `recommend`, one template `explain` and, when the bundle has an encoder,
  one `semantic_search`, before `ready` flips: lazy imports, page faults on the memory maps and
  the ONNX graph are paid before traffic arrives. `ONNX_THREADS`, when it differs from the
  bundle's `serving.onnx_threads`, rebuilds the encoder after load (a second ONNX session for a
  moment, so leave it at 1 on 512 MB).
- **Failure.** Up to `BOOT_RETRIES` attempts with backoff 5, 15, 45 s; after that the process stays
  up, `/health` names the error and `/ready` stays 503, so the platform's health check, not a
  crash loop, decides what happens. Shutdown: uvicorn drains in-flight requests for
  `GRACEFUL_SHUTDOWN_S` on SIGTERM.

### D79. Concurrency, limits, caches and timeouts (2026-10-20)
- **One process.** uvicorn with no workers: each worker would load its own store, and 512 MB
  holds one. Core calls run through `anyio.to_thread` on a dedicated limiter of `CORE_THREADS`
  (default 4), so the event loop keeps answering probes during a slow call. Thread safety relies
  on the contract: the store is read-only after load (memory maps, FAISS flat search, ONNX
  `run`, LightGBM `predict` all allow concurrent readers) and the LLM client locks its cache,
  budget and limiter; the service locks its own caches and buckets.
- **Ranking slots.** A semaphore of `RECOMMEND_CONCURRENCY` (default 2, 1 on Render) guards
  `recommend` and `home-rails`, the CPU-bound calls; a caller waits up to `QUEUE_TIMEOUT_S` and
  then gets 503 `overloaded` with `Retry-After: 1`. Chat is not under it because it mostly waits
  on Gemini, and holding a slot through a network wait would starve the feed.
- **Rate limit.** In-process token bucket (`RATE_LIMIT_RPS` 20, `RATE_LIMIT_BURST` 40) per
  `X-Client-Id` when the backend sends one (a per-reader pseudonym), else per client address
  (`X-Forwarded-For` behind the platform proxy). Probes are never limited. With more than one
  instance the buckets move to Redis behind the same `take(key)` call; one instance does not need it.
- **Caches.** TTL LRU (`CACHE_TTL_S` 300, `CACHE_MAX_ENTRIES` 1024) for `explain` and
  `book-insights`, keyed by `(bundle version, sha256 of the state JSON and the book id)` and
  `(bundle version, book id)`. A new bundle can never serve a stale entry; the key holds a hash,
  not the state. The core's sqlite LLM cache stays underneath (it saves Gemini calls across
  restarts on a persistent disk; the service cache saves the whole call). Answers produced by a
  timeout fallback are not cached.
- **LLM timeouts.** `explain`, `chat-turn` and `parse-onboarding` run under `LLM_TIMEOUT_S`
  (12 s: the core's own Gemini deadline is 10 s, D72). Past it the request does not fail: the
  same call runs again with `disabled_ai` (the keyword parse for onboarding) and the response
  carries `X-LitSense-Fallback: timeout`, which is the contract's "hard timeout, then template".
  The abandoned thread finishes on its own; Python cannot cancel it.
- **Other.** Gzip above 1 KB (feed responses with evidence compress well); `MAX_BODY_BYTES` 1 MB
  checked on `Content-Length` and on the streamed bytes; CORS off unless `CORS_ORIGINS` is set,
  since the Node backend, not a browser, is the caller.

### D80. Dependencies, image and deployment (2026-10-20)
- **New dependencies, service only** (`requirements-service.txt`, never in
  `requirements-core.txt`, so the core's allow-list and its tests are unchanged): `fastapi`
  0.152.0 and `starlette` 1.9.0 (the web framework the brief names; starlette pinned because the
  middleware is written against its ASGI types), `uvicorn` 0.54.0 (server), `pydantic-settings`
  2.16.1 (env config, as briefed), `prometheus-client` 0.26.0 (correct exposition format and the
  process collector for RSS and CPU; hand-writing the text format was the alternative). All
  resolved together with the core's `pydantic==2.13.5` on 2026-10-20. They are pure Python.
  `make install-ci` now installs them so CI runs `tests/service` (the CI subset grows by these
  five packages and their small transitive deps); a new `service-image` CI job builds the Docker
  image and checks that lightgbm loads (libgomp) and that the image refuses to start without a key.
- **Image** (`service/Dockerfile`, context the repo root, `.dockerignore` keeps only the core, the
  service and the pins): `python:3.12-slim`, `libgomp1` (D74), non-root user 10001 owning
  `/var/litsense`, `OMP_NUM_THREADS=1` (LightGBM would otherwise start a thread per host core on a
  0.1 CPU slice), bytecode precompiled, `HEALTHCHECK` on `/ready` with a 180 s start period.
  No `ml/` code and no `ml/config.yaml` in the image; the bundle's own `llm` slice configures the
  client.
- **Local stack** (`service/docker-compose.yml`): the model (512 MB limit, like Render), a static
  `bundle-host` serving `./local/serve_bundle.tar.gz` so the real download path runs locally before
  the GitHub Release exists, and a dependency-free Node stub that calls the service the way the
  backend should (key, request id, client id, catalogue lookups).
- **Render** (`render.yaml`, a Blueprint at the repo root): free plan, Docker, region Singapore
  (closest to the users), health check `/ready`, `SERVICE_API_KEY` generated by Render,
  `GEMINI_API_KEY` entered at creation (`sync: false`), `CORE_THREADS` 2 and
  `RECOMMEND_CONCURRENCY` 1 for 0.1 CPU. Free-plan facts checked in Render's docs on 2026-10-20:
  512 MB, 0.1 CPU, spin-down after 15 idle minutes, about a minute to spin up, 750 instance hours
  a month, ephemeral disk with no persistent disk on free, so the bundle downloads at every cold
  start. Railway: `RAILWAY_DOCKERFILE_PATH=service/Dockerfile`, health check `/ready`, optional
  volume at `/var/litsense`; its Free plan caps a service at 0.5 GB (docs checked the same day).

### D81. Service tests and measurements (2026-10-20)
- **Two test layers.** `tests/service` boots the app over the hand-built 12-book store
  (`tests/litsense_core/fake_store.py`) with the fake LLM provider: every route, the envelope,
  auth, readiness before and after load, a failed boot, request ids, rate limit, body limit (with
  and without `Content-Length`), catalogue paging, LLM timeouts with the template fallback, the
  explain cache, ranking slots, Prometheus text, JSON logs free of user content, gzip, CORS, and
  the committed OpenAPI file. One integration test builds the real tiny pipeline bundle once per
  session, serves its tar over a local HTTP server and boots the default loader: download, staging
  move, `ranker` true, every route, a 400 for an unknown id, a restart that skips the download,
  and recommend's p95 under the 80 ms budget. The bundle needs torch, which CI installs; without
  torch the test skips.
- **Found on the way.** Stages run in registration order, and importing `ml.nlp.onnx_export`
  before `ml.stages.load_stages()` registers `nlp.onnx` ahead of `eval.val`, so `run all --until
  eval.val` tried the real ONNX export. tests/litsense_core avoided it by calling `load_stages()`
  in its conftest; the session fixture now does the same before importing the bundle builder,
  and the builder's docstring says so.
- **CI was red on main for the bundle tests.** In a container built exactly like CI
  (`make install-ci` on `python:3.12-slim`), both `tests/litsense_core/test_bundle.py` tests error
  at `ranking.train`: lightgbm's `LGBMRanker` lives in `lightgbm.sklearn`, which refuses to run
  without scikit-learn, and the CI subset never installed it (D66 added the stage after the
  bde7d78 fix that kept sklearn out). Checked on 084f1c0 (main) with the same result, so it
  predates this branch; the new service integration test hit the same wall. `scikit-learn`
  (already pinned in `requirements-ml.txt`) joins `CI_ML_PACKAGES`. It is test-only: the serving
  image installs `requirements-core.txt` and `requirements-service.txt` alone, where lightgbm's
  optional sklearn import stays absent (the D66 RSS note).
- **Measured (Docker on the laptop, tiny bundle v0-20261020 with 52 books; the 94 MB real bundle
  is on Drive only, so its numbers are estimates).** In-process test client, warm, sequential:
  recommend p95 7.6 ms in the core, 13.5 ms through HTTP. The image at `--cpus 1 -m 512m`:
  imports 2.7 s, load 0.43 s, ready 3.4 s after container start; load test 300 requests,
  recommend p50 13.5 / p95 30.3 ms and explain (template) p50 3.9 / p95 6.0 ms with one client at a
  time; recommend p50 72.6 / p95 108.4 ms and 53 requests per second with four concurrent clients
  queueing for two ranking slots. The image at `--cpus 0.1 -m 512m` (the Render free plan):
  imports 33.7 s, download 1.6 s (local), load 12.2 s, warm 0.4 s, ready 48 s after container
  start; load test 100 requests with two clients and one ranking slot, recommend p50 202 /
  p95 296 ms, explain p50 98 / p95 190 ms. RSS after load 152 to 156 MB and flat through every load
  test (VmRSS; `docker stats` shows about 100 MB because it leaves out shared file pages).
- **What this means for the free plan.** The 80 ms budget is for one warm CPU thread; on 0.1 CPU
  every CPU-bound call is about ten times slower, so with the real bundle (recommend 35 to 64 ms on
  a full core, D66) expect roughly 0.4 to 0.7 s per feed request and a cold start of the platform
  spin-up plus about 35 s of imports, the download, and a slower load than the tiny bundle's 12 s
  (the 90 MB ONNX model and the 4,190-book matrices). RSS should land near the D74 estimate of 230
  to 260 MB, inside 512 MB. These are estimates until the release URL exists and the service runs
  on Render; `/health` reports `download_s`, `load_s`, `warm_s` and `rss_mb` to confirm them. A
  paid instance (0.5 CPU or more) or Railway's Trial brings the feed back inside the budget.
