"""Naming objects with Qwen2.5-VL, from a highlight crop.

Every object (foreground or background) is named the same way: its crop at
the frame where its mask is largest, with everything outside the mask dimmed
and the mask outlined (:func:`~.clip_geometry.highlight_crop`), is shown to Qwen2.5-VL-7B-
Instruct (greedy decoding), which answers a descriptor phrase and a head
noun (:func:`name_object`). The answer is cleaned (a denylist of
generic words, a phrase/noun consistency rule) into an :class:`ObjectName`.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from PIL import Image

from trajloc.preprocess.stage1.qwen import QwenVL

# Names too generic to identify an object (surfaces, atmospherics, ...): an
# answer in this set is replaced, the phrase by the noun and the noun by the
# phrase's last word (else FALLBACK_NOUN).
GENERIC_NAME_DENYLIST = frozenset(
    {
        # surfaces / atmospherics
        "sky",
        "ground",
        "floor",
        "grass",
        "road",
        "pavement",
        "water",
        "wall",
        "ceiling",
        "dirt",
        "sand",
        "atmosphere",
        "lighting",
        "shadow",
        "blur",
        "background",
        "foreground",
        "scene",
        "image",
        "picture",
        "view",
        "horizon",
        "field",
        "sea",
        "ocean",
        "snow",
        "ice",
        "mud",
        "smoke",
        "fog",
        "haze",
        "carpet",
        "tile",
        "wood",
        "concrete",
        "lawn",
        "puddle",
        "cloud",
        "clouds",
        "surface",
        # bare colors
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        "tan",
        "beige",
        "gold",
        "silver",
        # bare adjectives / participles
        "folding",
        "wooden",
        "plastic",
        "metal",
        "large",
        "small",
        "tall",
        "short",
        "sitting",
        "standing",
        "hanging",
        "framed",
        "parked",
        # abstract / category words
        "decoration",
        "decorations",
        "equipment",
        "item",
        "items",
        "object",
        "objects",
        "thing",
        "things",
        "stuff",
        "scenery",
        "environment",
        "area",
        "space",
        "place",
        "side",
        "part",
        "section",
    }
)


FALLBACK_NOUN = "object"
PHRASE_MAX_TOKENS = 24
NOUN_MAX_TOKENS = 8


# Naming uses two short, single-purpose Qwen calls (a combined prompt made
# Qwen echo instructions or drop the noun):
#
#   1. PHRASE_PROMPT  -> a short descriptor phrase ('wooden bench',
#      'person in a red jacket', 'green forest', 'swan').
#   2. NOUN_PROMPT    -> the single head noun, one word ('bench', 'person',
#      'forest', 'swan') — the main thing the region IS, never a body part,
#      clothing, or surrounding terrain.
#
# The crop keeps context (the object at full brightness with a magenta
# outline, the rest dimmed), so small or ambiguous regions stay recognizable.
# The noun must be one of the phrase's words, because the prompt builder
# (:mod:`.prompts`) finds it there; see :func:`name_object`. The prompts are
# the model's input as the released data was built: keep them verbatim
# (spelling included).
PHRASE_PROMPT = (
    "One region in this image is outlined by a bright magenta contour and "
    "shown at full brightness; everything else is dimmed. In 1 to 4 words, "
    "name ONLY the magenta-outlined region (use the dimmed surroundings just "
    "as context). Use a short noun phrase with at most a colour/material/"
    "clothing modifier — no verbs, no actions, no 'on a...'/'with...' "
    "clauses. Examples: 'wooden bench', 'red car', 'person in a blue jacket', "
    "'green forest', 'swan'. Answer with the phrase only, lowercase, no "
    "period, no mention of the outline."
)


NOUN_PROMPT = (
    "One region in this image is outlined by a bright magenta contour. In a "
    "SINGLE word, what is that outlined thing? Give the main object it IS — "
    "for a person or animal the creature itself (person, man, woman, dog, "
    "swan, ...), never a body part, clothing, or the ground/terrain around "
    "it. One lowercase noun, nothing else."
)


def _normalize_answer(s: str) -> str:
    """A Qwen answer, lowercased, without list markers, trailing
    punctuation, a leading article or anything after a comma; runs of
    whitespace collapsed."""
    s = re.sub(r"^[-*\d.\)\(]+\s*", "", s)
    s = s.lower().rstrip(".!?:;,").strip()
    for article in ("a ", "an ", "the "):
        if s.startswith(article):
            s = s[len(article) :]
    if "," in s:
        s = s.split(",")[0].strip()
    return re.sub(r"\s+", " ", s)


def _clean_phrase(raw: str) -> str:
    """The descriptor phrase in Qwen's answer: its first line, normalized."""
    return _normalize_answer(raw.strip().splitlines()[0] if raw else "")


def _clean_token(s: str) -> str:
    """The noun in Qwen's answer, normalized (and stripped)."""
    return _normalize_answer((s or "").strip()).strip()


class ObjectName(NamedTuple):
    """An object's name: its head noun, and the descriptor phrase ending in it."""

    noun: str
    phrase: str


def name_object(qwen: QwenVL, pil_crop: Image.Image | None) -> ObjectName:
    """Name an object from its highlight crop.

    Two short Qwen calls, one per answer (a single combined prompt made Qwen
    echo instructions or drop the noun): a descriptor phrase, then the head
    noun. The noun must be one of the phrase's words, because the prompt
    builder finds it there; if it isn't, the phrase is the bare noun. Without
    a crop, or without a usable answer, the name falls back to ``"object"``.

    The scene caption is not shown: it biased ambiguous crops toward the
    caption's nouns.
    """
    if pil_crop is None:
        return ObjectName(FALLBACK_NOUN, FALLBACK_NOUN)

    # Call 1: short descriptor phrase.
    phrase = _clean_phrase(
        qwen.ask_image(pil_crop, PHRASE_PROMPT, max_new_tokens=PHRASE_MAX_TOKENS)
    )
    # Call 2: single head noun.
    noun_raw = qwen.ask_image(pil_crop, NOUN_PROMPT, max_new_tokens=NOUN_MAX_TOKENS)
    noun_clean = _clean_token(noun_raw)
    noun = noun_clean.split()[0] if noun_clean else ""

    # Fallbacks if either call returned nothing usable.
    if not phrase or phrase in GENERIC_NAME_DENYLIST:
        phrase = noun or FALLBACK_NOUN
    if not noun or noun in GENERIC_NAME_DENYLIST:
        noun = phrase.split()[-1] if phrase else FALLBACK_NOUN

    # Enforce noun is a contiguous-token substring of phrase (the prompt
    # builder locates the noun inside the phrase).
    # When the two calls DISAGREE (noun not in phrase — e.g. phrase 'brown
    # leather', noun 'arm'), the noun is the more reliable answer (dedicated
    # single-word question), so fall back to the bare noun as the phrase
    # rather than mashing the two into 'arm brown leather'.
    if noun not in phrase.split():
        phrase = noun

    return ObjectName(noun, phrase)
