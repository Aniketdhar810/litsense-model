"""Aspect taxonomy, lexicons, prototypes and descriptor poles (CLAUDE.md s11.2).

``ASPECTS`` order is the aspect vector index everywhere (0-11) and must equal ``aspects.names`` in
ml/config.yaml. Lexicon terms are regular expressions matched with word boundaries,
case-insensitive; extend them during aspect validation (s11.5).
"""

import re
from collections.abc import Sequence

ASPECTS: tuple[str, ...] = (
    "plot", "writing_style", "characters", "world_building", "pacing", "editing_quality",
    "dialogue", "humor", "emotional_impact", "ending", "complexity", "tone_dark",
)  # fmt: skip
LABELS: dict[str, str] = {
    "plot": "plot", "writing_style": "prose", "characters": "characters",
    "world_building": "world-building", "pacing": "pacing", "editing_quality": "editing",
    "dialogue": "dialogue", "humor": "humor", "emotional_impact": "emotional punch",
    "ending": "ending", "complexity": "depth", "tone_dark": "tone",
}  # fmt: skip
LEXICON: dict[str, list[str]] = {
    "plot": [r"plot", r"story", r"storyline", r"narrative", r"twists?", r"subplots?", r"premise",
             r"mystery"],
    "writing_style": [r"prose", r"writing", r"written", r"style", r"voice", r"sentences",
                      r"language", r"lyrical", r"beautifully written"],
    "characters": [r"characters?", r"protagonist", r"heroine", r"hero", r"villain",
                   r"main character", r"cast", r"relatable", r"likable", r"character development"],
    "world_building": [r"world-?building", r"world", r"setting", r"universe", r"lore",
                       r"magic system", r"atmosphere", r"immersive"],
    "pacing": [r"pacing", r"pace", r"slow", r"slow-burn", r"dragged", r"drags", r"page-turner",
               r"fast-paced", r"couldn't put it down", r"tedious", r"rushed"],
    "editing_quality": [r"typos?", r"editing", r"edited", r"proofreading", r"grammar",
                        r"grammatical errors", r"formatting", r"translation", r"translated"],
    "dialogue": [r"dialogue", r"dialog", r"conversations?", r"banter", r"exchanges"],
    "humor": [r"funny", r"humou?r", r"hilarious", r"witty", r"laugh(ed)?", r"comedic", r"jokes",
              r"sarcastic"],
    "emotional_impact": [r"moving", r"cried", r"tears", r"emotional", r"heartbreaking",
                         r"heartwarming", r"touching", r"gut-wrenching"],
    "ending": [r"ending", r"conclusion", r"finale", r"resolution", r"cliffhanger", r"epilogue",
               r"last chapters?"],
    "complexity": [r"confusing", r"complex", r"complicated", r"convoluted", r"dense",
                   r"hard to follow", r"layered", r"intricate", r"easy read"],
    "tone_dark": [r"dark", r"bleak", r"grim", r"disturbing", r"brutal", r"violent", r"gritty",
                  r"depressing", r"harrowing", r"grimdark", r"cozy"],
}  # fmt: skip

PROTOTYPES: dict[str, list[str]] = {
    "plot": [
        "The plot kept me guessing until the very last page.",
        "The storyline was predictable and full of holes.",
        "What a clever, intricate story.",
        "The premise was great but the story went nowhere.",
        "The twists were genuinely surprising.",
        "Too many subplots that never connected.",
        "The mystery at the heart of the book was well constructed.",
        "Nothing much happens in this story.",
    ],
    "writing_style": [
        "The prose is gorgeous and lyrical.",
        "The writing was clunky and amateurish.",
        "Beautifully written, every sentence is a pleasure to read.",
        "The author's voice is distinctive and sharp.",
        "The writing style felt flat and repetitive.",
        "Her sentences are precise and elegant.",
        "It reads like a first draft.",
        "The language is rich without being overwrought.",
    ],
    "characters": [
        "I fell in love with the characters.",
        "The main character was annoying and hard to root for.",
        "The characters felt flat and one-dimensional.",
        "Great character development throughout the series.",
        "The protagonist is complex and believable.",
        "I couldn't connect with any of the characters.",
        "The villain was genuinely menacing.",
        "The friendship between the two leads was wonderful.",
    ],
    "world_building": [
        "The world-building is detailed and immersive.",
        "The magic system was original and well thought out.",
        "The setting came alive on the page.",
        "I never understood how this world worked.",
        "The atmosphere of the small town was perfectly captured.",
        "The lore was rich and layered.",
        "The world felt thin and underdeveloped.",
        "Vivid descriptions made me feel I was there.",
    ],
    "pacing": [
        "The pacing was perfect, I couldn't put it down.",
        "The middle dragged on forever.",
        "A real page-turner from start to finish.",
        "It took two hundred pages before anything happened.",
        "The ending felt rushed compared to the slow start.",
        "The book moves at a breakneck pace.",
        "It was a slow burn but worth it.",
        "I found it tedious and kept skimming.",
    ],
    "editing_quality": [
        "The book is full of typos and grammatical errors.",
        "It desperately needed a good editor.",
        "The translation reads smoothly.",
        "The formatting was a mess with missing words.",
        "Well edited and cleanly produced.",
        "So many spelling mistakes it pulled me out of the story.",
        "The translation felt awkward and stilted.",
        "Clearly no one proofread this.",
    ],
    "dialogue": [
        "The dialogue is sharp and witty.",
        "The conversations felt stilted and unnatural.",
        "I loved the banter between the two leads.",
        "Nobody talks like that in real life.",
        "The dialogue crackles with energy.",
        "Too much of the book is long expository conversation.",
        "Their exchanges were funny and believable.",
        "The dialogue was cheesy.",
    ],
    "humor": [
        "This book made me laugh out loud.",
        "The humor is dry and clever.",
        "The jokes fell flat for me.",
        "Hilarious from start to finish.",
        "A witty, sarcastic narrator.",
        "It tried too hard to be funny.",
        "There's a lovely thread of humor throughout.",
        "I chuckled on almost every page.",
    ],
    "emotional_impact": [
        "I cried my eyes out.",
        "A deeply moving and emotional read.",
        "It left me feeling nothing at all.",
        "Heartwarming and hopeful.",
        "This book broke my heart.",
        "It stayed with me for days after I finished.",
        "Emotionally manipulative and shallow.",
        "A touching story about family and loss.",
    ],
    "ending": [
        "The ending was perfect.",
        "The ending felt rushed and unsatisfying.",
        "I hated the cliffhanger.",
        "The conclusion tied everything together beautifully.",
        "The last few chapters were a letdown.",
        "What a satisfying finale to the series.",
        "The ending came out of nowhere.",
        "The epilogue was the cherry on top.",
    ],
    "complexity": [
        "It was confusing and hard to follow.",
        "A complex, layered novel that rewards attention.",
        "Too many characters to keep track of.",
        "An easy, light read.",
        "The plot was convoluted.",
        "Dense but worth the effort.",
        "I had to reread passages to understand what was happening.",
        "Intricate without ever being confusing.",
    ],
    "tone_dark": [
        "This is a dark and disturbing book.",
        "Very violent and graphic in places.",
        "It's grim and bleak throughout.",
        "A cozy, gentle story.",
        "Too depressing for me.",
        "Gritty and unflinching.",
        "Lighthearted and uplifting.",
        "Harrowing but powerful.",
    ],
}

# Bipolar descriptors: intensity, not sentiment; scored only on sentences hitting the linked aspect.
DESCRIPTOR_POLES: dict[str, dict[str, object]] = {
    "darkness": {
        "aspect": "tone_dark",
        "plus": ["This is a dark and disturbing book.", "Grim, bleak and violent.",
                 "Gritty and unflinching.", "Harrowing and brutal at times."],
        "minus": ["A cozy, gentle story.", "Lighthearted and uplifting.",
                  "Sweet and comforting.", "A feel-good read."],
    },
    "pace": {
        "aspect": "pacing",
        "plus": ["A fast-paced page-turner.", "Action from the first page.",
                 "It moves at a breakneck pace.", "I read it in one sitting."],
        "minus": ["A slow, meditative read.", "It's a slow burn.",
                  "Very little happens for long stretches.", "It takes its time."],
    },
    "complexity_level": {
        "aspect": "complexity",
        "plus": ["Dense and complex.", "Layered and intricate, it demands attention.",
                 "Hard to follow at times.", "A challenging literary read."],
        "minus": ["An easy, light read.", "Simple and straightforward.", "A quick, breezy read.",
                  "Perfect beach read."],
    },
}  # fmt: skip


def compile_lexicon(aspects: Sequence[str] = ASPECTS) -> dict[str, re.Pattern[str]]:
    """One case-insensitive, word-bounded pattern per aspect."""
    return {a: re.compile(r"\b(?:" + "|".join(LEXICON[a]) + r")\b", re.IGNORECASE) for a in aspects}


_ANY_ASPECT = re.compile(
    r"\b(?:" + "|".join(t for terms in LEXICON.values() for t in terms) + r")\b", re.IGNORECASE
)


def has_aspect_term(text: str) -> bool:
    """True if ``text`` mentions any aspect lexicon term (data.clean's shipping-complaint rule)."""
    return bool(_ANY_ASPECT.search(text))
