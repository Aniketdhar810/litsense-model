# Aspect annotation guide (CLAUDE.md s11.5)

You are labelling review sentences from `docs/annotation/aspects_300.csv`. Each row is one
sentence from an Amazon book review. For each of the 12 aspect columns write:

| label | meaning |
| --- | --- |
| `+` | the sentence mentions this aspect and speaks well of it |
| `-` | the sentence mentions this aspect and criticises it |
| `0` | the sentence mentions this aspect without a clear opinion |
| (blank) | the sentence is not about this aspect |

Rules of thumb:

- Label what the sentence says, not what you guess about the book. "I expected great prose" says
  nothing about the prose: leave `writing_style` blank.
- A sentence can carry several aspects ("Wonderful characters, but the middle third really
  dragged." is `characters +` and `pacing -`).
- A sentence about the reader's situation ("I read it on holiday") carries no aspect.
- Complaints about shipping, price, print size or the Kindle file carry no aspect (they were
  filtered earlier, but a few slip through).
- Spoiler sentences are labelled like any other sentence; the spoiler filter is a separate step.
- Do not open the book's page or other reviews. Label from the sentence alone.
- Leave every column blank when unsure; a blank is "absent", which is the safe default.

Each of the 12 aspects below has a definition, two examples and the confusions to avoid. The
column order in the CSV is the order here.

## 1. plot

What happens: the story, its structure, twists, premise, logic, predictability.

- `+` "The plot kept me guessing until the very last page."
- `-` "The storyline was predictable and full of holes."

Not `plot`: a bare summary of the premise with no judgement is `0`; "the ending felt rushed" is
`ending` (and perhaps `pacing`), not `plot`.

## 2. writing_style (UI label: prose)

How it is written: prose, sentences, voice, language, lyricism, clunkiness.

- `+` "Beautifully written, every sentence is a pleasure to read."
- `-` "The writing was clunky and amateurish."

Not `writing_style`: typos, grammar mistakes and formatting are `editing_quality`; "well written"
used loosely about a satisfying story is still `writing_style` (label what the words say).

## 3. characters

The people: protagonist, villain, cast, development, believability, whether the reader connected.

- `+` "The protagonist is complex and believable."
- `-` "The characters felt flat and one-dimensional."

Not `characters`: "too many characters to keep track of" is `complexity` (the problem is the
reader's load, not the people); banter between characters is `dialogue`.

## 4. world_building (UI label: world-building)

The setting as a built thing: world, universe, lore, magic system, atmosphere, immersion, sense
of place (also for realistic settings: a small town captured vividly).

- `+` "The world-building is detailed and immersive."
- `-` "I never understood how this world worked."

Not `world_building`: a dark or bleak atmosphere is `tone_dark` unless the sentence is about how
the setting is constructed; "vivid descriptions" of people are `characters`.

## 5. pacing

Speed and momentum: slow, dragged, page-turner, could not put it down, rushed, tedious.

- `+` "A real page-turner from start to finish."
- `-` "The middle dragged on forever."

Not `pacing`: "the ending was rushed" is both `ending` and `pacing`; "slow burn but worth it" is
`pacing +` (the reader approves of the slow pace). Intensity (how fast) is captured separately
by the descriptor model, so label the opinion, not the speed.

## 6. editing_quality (UI label: editing)

Production quality: typos, grammar, proofreading, formatting, translation.

- `+` "Well edited and cleanly produced."
- `-` "The book is full of typos and grammatical errors."

Not `editing_quality`: "needed an editor to cut 200 pages" is about length and `pacing`, not
production errors; a bad translation of the *prose style* can be both `editing_quality` and
`writing_style`.

## 7. dialogue

Spoken exchanges: dialogue, conversations, banter, how people talk.

- `+` "The dialogue is sharp and witty."
- `-` "The conversations felt stilted and unnatural."

Not `dialogue`: funny dialogue is `dialogue +` and `humor +`; a sentence about the narrator's
voice (not speech) is `writing_style`.

## 8. humor

Funniness: laughed, hilarious, witty, jokes, sarcastic narrator, tried too hard.

- `+` "This book made me laugh out loud."
- `-` "The jokes fell flat for me."

Not `humor`: "lighthearted" alone is tone (`tone_dark -` as the opposite of dark) unless it says
the book is funny.

## 9. emotional_impact (UI label: emotional punch)

How it made the reader feel: moving, cried, heartbreaking, heartwarming, stayed with me, left me
cold.

- `+` "A deeply moving and emotional read."
- `-` "It left me feeling nothing at all."

Not `emotional_impact`: "the ending made me cry" is `ending +` and `emotional_impact +`; a sad
*subject* (grief, war) without a statement about the reader's reaction is `tone_dark`, not
emotional impact.

## 10. ending

The last part: ending, conclusion, finale, resolution, cliffhanger, epilogue, last chapters.

- `+` "The conclusion tied everything together beautifully."
- `-` "The ending felt rushed and unsatisfying."

Not `ending`: "I could not wait to see how it ended" is `pacing +` or `plot +`; a cliffhanger
mentioned neutrally ("ends on a cliffhanger, book two is out") is `ending 0`.

## 11. complexity (UI label: depth)

Mental load and layering: confusing, hard to follow, convoluted, dense, layered, intricate, easy
read.

- `+` "A complex, layered novel that rewards attention."
- `-` "It was confusing and hard to follow."

Not `complexity`: an "easy read" praised for being light is `complexity +` (the reader approves
of the level); a complicated *plot* that the reader enjoyed is `plot +` and `complexity +`. "Dense
prose" is `writing_style` and `complexity`.

## 12. tone_dark (UI label: tone)

Darkness of tone: dark, bleak, grim, disturbing, violent, gritty, depressing, harrowing, and their
opposite (cozy, gentle, lighthearted, uplifting).

- `+` "Gritty and unflinching in the best way." (the reader approves of the darkness)
- `-` "Too depressing for me."

Labelling tone: `+` means the reader liked the tone, `-` means the tone put them off, `0` means
they only described it ("this is a dark book"). A cozy book praised for being cozy is `tone_dark
+` as well: the column is about the *tone* aspect, and the plus/minus is the opinion. How dark the
book is comes from the descriptor model, not from this label.

## Edge cases in one place

| sentence | labels |
| --- | --- |
| "The ending made me cry." | `ending +`, `emotional_impact +` |
| "The ending felt rushed." | `ending -`, `pacing -` |
| "Too many characters to keep track of." | `complexity -` |
| "Wonderful characters, but the middle third really dragged." | `characters +`, `pacing -` |
| "The banter is hilarious." | `dialogue +`, `humor +` |
| "This is a very dark book." | `tone_dark 0` |
| "Arrived with a torn cover." | nothing |
| "I read it over a long weekend." | nothing |
| "The plot was convoluted." | `plot -`, `complexity -` |

## Process

1. The user labels all 300 rows in place: `docs/annotation/aspects_300.csv` (same columns, same
   row order; open it as UTF-8, for example Google Sheets import, since Excel double-click garbles
   apostrophes). The second labeller labels the first 100 rows in a copy saved as
   `docs/annotation/aspects_300_second.csv` (for agreement, Cohen's kappa per aspect).
2. Commit both files. On Colab, copy them into the raw directory and run the stage by name:
   `cp docs/annotation/aspects_300*.csv /content/raw/ && python -m ml.stages run nlp.validate_aspects`.
   The labelled file is a `raw:` input, so changed labels rerun the stage; the second file is read
   from the same directory (use `--force` after changing only that one).
3. The stage tunes the per-aspect threshold on a seeded half of the sentences and reports
   precision, recall and F1 per aspect on the other half (`results/<v>/nlp/aspect_validation.json`
   and `.csv`). It prints the proposed `aspects.tau` block; the user confirms before it goes into
   `ml/config.yaml` (which then reruns `nlp.aspects` and everything after it). Target macro-F1 at or
   above 0.6; aspects below 0.4 are flagged for a lexicon change, a merge or a drop.
