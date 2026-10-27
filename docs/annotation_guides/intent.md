# Intent labelling guide (for the people filling `intent_candidates.csv`)

You are teaching the app what a reader *means* when they type a message in the book chat.
No technical knowledge is needed. You fill one row per message in a spreadsheet
(open `docs/annotation/intent_candidates.csv` in Excel or Google Sheets). About 1 to 2 hours
for 70 rows. Keep 50 or more.

Do not use ChatGPT, Gemini or any AI to fill the sheet. The whole point is a human answer.

## The columns

| Column | What to put | Example |
|---|---|---|
| `id` | leave as is | c01 |
| `message` | leave as is (you may fix a typo) | something like Project Hail Mary but shorter |
| `keep` | `yes` if this row goes into the final 50, else leave empty | yes |
| `intent` | ONE of the 9 words from the table below. A suggestion is prefilled; correct it if wrong | like + change |
| `book_named` | the book title written in the message, copied exactly; empty if none | Project Hail Mary |
| `author_named` | the author name written in the message, copied exactly; empty if none | Andy Weir |
| `that_one_or_author` | `yes` if the message points back to an earlier result ("that one", "that author", "the same author") | yes |
| `which_one` | for "why the second one?" type messages: 1, 2, 3... (or first/second/third) | 2 |
| `shorter_or_longer` | `shorter` or `longer` if the reader asks for length; else empty | shorter |
| `mood` | words from the mood list below, comma separated; empty if none | cozier, funnier |
| `year_from` | a year if the reader wants books from that year on ("nothing older than 2010", "recent" = 2021) | 2010 |
| `year_to` | a year if the reader wants books up to that year ("before 1980", "classic" = 1980) | 1980 |
| `only_genre` | a genre the reader wants only (Fantasy, Mystery, Thriller, Romance, Science Fiction, Horror, Historical Fiction, Literary Fiction, Young Adult) | Fantasy |
| `no_genre` | a genre the reader does not want | Horror |
| `something_else` | `yes` if the reader rejects what was shown ("something else", "not these") | yes |
| `shown_before` | if the message only makes sense after some results were shown, write 1 to 3 book titles that were shown, separated by `;` (any real titles from the app) | The Martian; Artemis |
| `notes` | anything unclear, in your own words | |

Empty means "no". Only fill what the message really says. Do not guess.

## The 9 intents (type exactly one of the words in the first column)

| Type this | Means | Example messages |
|---|---|---|
| `general` | wants recommendations, no particular book named | "recommend me a good book", "a cozy mystery for the weekend" |
| `like a book` | wants books similar to a named book, no other change | "something like Project Hail Mary", "books similar to Circe" |
| `like + change` | similar to a named book, plus a change (shorter, cozier, newer...) | "like Project Hail Mary but shorter", "like Circe but faster paced" |
| `more by author` | wants more books by an author (named, or "that author") | "more by Andy Weir", "more by that author" |
| `refine` | changes the previous results without naming a book | "a bit cozier", "something else", "nothing older than 2010", "funnier" |
| `why` | asks why a book was suggested | "why the second one?", "why did you suggest that?" |
| `filter` | only a hard limit: genre, years, pages | "only fantasy", "no horror please", "under 300 pages" |
| `mixed` | says what they liked and disliked in a named book | "I loved the characters in Circe but the pacing was slow" |
| `not about books` | anything else | "what's the weather tomorrow", "write my essay" |

If two fit, pick the one the reader mainly wants. "like X but shorter" is `like + change`, not `filter`.

## Mood words (for the `mood` column)

`cozier`, `lighter`, `less dark`, `darker`, `faster`, `gripping`, `slower`, `slow burn`, `easier`,
`simpler`, `denser`, `more literary`, `funnier`, `more humor`, `less humor`, `better characters`,
`better plot`, `better prose`, `more emotional`.

Use only these words. "less bleak" = `lighter`. "page-turner" = `faster`. "not so complicated" = `easier`.

## Six filled examples

| message | intent | book_named | author_named | that_one_or_author | which_one | shorter_or_longer | mood | year_from | something_else | shown_before |
|---|---|---|---|---|---|---|---|---|---|---|
| something like Project Hail Mary but shorter | like + change | Project Hail Mary | | | | shorter | | | | |
| more by that author | more by author | | | yes | | | | | | The Martian; Artemis |
| why the second one? | why | | | | 2 | | | | | The Martian; Artemis |
| a bit cozier, and nothing older than 2010 | refine | | | | | | cozier | 2010 | | Gone Girl |
| something else | refine | | | | | | | | yes | Gone Girl |
| what's the weather tomorrow | not about books | | | | | | | | | |

## When you are done

1. Save the file as CSV (same name is fine). Google Sheets: File > Download > CSV.
2. Send it to Zeeshan. He runs one command that turns it into the scoring file and reports the
   accuracy. If a row is wrong (unknown intent word or mood word), the command names the row id.

Thank you. Your answers become Table 10 of the report (intent-parser accuracy).
