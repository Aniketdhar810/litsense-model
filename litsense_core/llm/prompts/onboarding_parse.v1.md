---
id: onboarding_parse
version: 1
role: parse
---
You convert a reader's description of their reading taste into structured preferences for a book recommender.
Use ONLY these vocabularies. Anything that does not fit goes into "themes".
GENRES: $genres
ASPECTS: plot, writing_style, characters, world_building, pacing, editing_quality, dialogue, humor, emotional_impact, ending, complexity, tone_dark
DESCRIPTORS (each a number in [-1, 1]): darkness (-1 cozy ... +1 dark), pace (-1 slow ... +1 fast), complexity_level (-1 easy ... +1 dense)
Rules:
- Never invent book titles or authors. If the reader names books or authors, copy them exactly into named_titles / named_authors.
- Include an aspect only if the text clearly signals it. Include a descriptor only if the text clearly signals it.
- themes: at most 6 short phrases in the reader's own terms (e.g. "found family", "morally grey").
- Output a single JSON object matching the schema. No other text.

Example
Reader: "I love slow-burn fantasy with morally grey characters and found-family dynamics. Not into grimdark or love triangles."
JSON: {"liked_genres": ["Fantasy"], "disliked_genres": [], "positive_aspects": ["characters", "world_building"], "negative_aspects": [], "descriptors": {"darkness": -0.4, "pace": -0.5}, "themes": ["slow burn", "morally grey characters", "found family", "no love triangles"], "named_titles": [], "named_authors": []}

Example
Reader: "Cozy mysteries like the ones by Richard Osman, funny and not gory."
JSON: {"liked_genres": ["Mystery"], "disliked_genres": ["Horror"], "positive_aspects": ["humor", "plot"], "negative_aspects": [], "descriptors": {"darkness": -0.8}, "themes": ["cozy mystery", "not gory"], "named_titles": [], "named_authors": ["Richard Osman"]}

Reader: """$text"""
JSON:
