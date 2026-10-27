# Explanation audit guide (for the three raters)

The app shows a reader a book and one sentence saying why. Some sentences are written by the
AI (Gemini), some by a fixed template. You judge each sentence. Three people rate the same 100
rows independently; do not discuss rows while rating. About 45 to 60 minutes.

Open `docs/annotation/explain_audit.csv` in Excel or Google Sheets. Fill only the last four
columns. Do not use any AI tool to rate.

## What you see in each row

| Column | Meaning |
|---|---|
| `book_title` | the book being recommended |
| `explanation` | the sentence shown to the reader. Judge this. |
| `evidence_text` | what the app knew when it wrote the sentence: the reader's liked books, favourite author, what they value, what reviewers praise, one review snippet. This is the only "truth" the sentence may use. |
| `method` | ignore while rating (llm or template); it is there for the analysis |
| `evidence_json` | the same evidence in raw form; ignore |

## The four questions (fill one answer per column)

| Column | Write | Question |
|---|---|---|
| `grounded` | Y or N | Is everything the sentence claims supported by `evidence_text`? Y only if every fact (reader liked X, reviewers praise Y, same author, percentage) is there. A claim that is not in the evidence = N, even if it sounds plausible. |
| `hallucinated_title` | Y or N | Does the sentence name a book that is NOT the recommended book and NOT one of the reader's liked books listed in the evidence? Y = yes, it invented or mis-named a book. Usually N. |
| `spoiler` | Y or N | Does the sentence reveal a plot event, twist, who dies, who the killer is, or how it ends? Back-cover level descriptions ("a murder mystery in a small town") are fine = N. |
| `useful` | 1 to 5 | Would this sentence help the reader decide? 1 = useless or wrong, 3 = ok, 5 = clearly helpful and specific. |

Rules of thumb:
- Read `evidence_text` first, then the sentence.
- "Because you loved Andy Weir's Project Hail Mary" is grounded only if the evidence lists that
  book or that favourite author.
- A number in the sentence must appear in the evidence (74% positive, 5 stars).
- Vague but true sentences ("A popular pick for mystery readers") are grounded; rate `useful` low
  if they say nothing specific.
- If you cannot decide `grounded`, answer N and write why in the `notes` column (add one at the end
  if you like).

## Example

```
explanation : Since you gave five stars to Project Hail Mary, you will likely enjoy The Martian
              by Andy Weir, which reviewers praise for its pacing.
evidence_text: BOOK: The Martian by Andy Weir (Science Fiction, 2011)
               READER LIKED: Project Hail Mary (5 stars)
               READER'S FAVOURITE AUTHOR: Andy Weir
               REVIEWERS PRAISE: pacing (81% positive), plot (74% positive), humor (70% positive)
grounded = Y, hallucinated_title = N, spoiler = N, useful = 5
```

```
explanation : You will love how the twist reveals the narrator was dead all along.
grounded = N (not in evidence), hallucinated_title = N, spoiler = Y, useful = 1
```

## When done

Save as CSV with your name in the file name (`explain_audit_<name>.csv`) and send it to Zeeshan.
He runs one command over the three files; it reports the share grounded, hallucinated and
spoiler per method, the average usefulness, and how much the three of you agreed (Fleiss' kappa).
Targets from the project brief: at least 90% grounded, zero hallucinated titles, zero spoilers.
