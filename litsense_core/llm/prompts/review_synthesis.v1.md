---
id: review_synthesis
version: 1
role: synthesis
---
Summarize what reviewers think of this book in 2-3 sentences (max 70 words).
Use ONLY the statistics and snippets given. Mention no other book. Do not reveal plot twists, character fates or the ending; describe the book only at the level of its back-cover premise. Use numbers only if they appear below. Return JSON: {"summary": string}.
TITLE: $title
AUTHOR: $author
REVIEWS ANALYSED: $n_reviews
PRAISED: $praised
CRITICISED: $criticised
POSITIVE SNIPPETS: $pos_snippets
NEGATIVE SNIPPETS: $neg_snippets
JSON:
