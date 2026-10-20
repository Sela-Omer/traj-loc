"""Benchmark prompts: one clause per named object, ending in its ``<traj_i>``.

Each object (named by :mod:`.naming`: a phrase ending in its head noun) gets
``<article> <phrase> <traj_i>``, and the clauses are joined with commas and a
final "and". The metadata per object records the word indices of its head
noun and ``<traj_i>`` token, which Stage 2 maps to token positions.

Input::

    [{"noun": "shopping cart", "phrase": "red shopping cart", "source_object_id": 5}, ...]

Output: a :class:`~trajloc.preprocess.stage1.records.PromptSynthesis`
(every object's ``modifier`` is empty).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, NamedTuple

from trajloc.preprocess.stage1.prompt_words import locate_object_words, starts_with_vowel
from trajloc.preprocess.stage1.records import PromptSynthesis

# Plural / mass / collective nouns that don't take 'a' / 'an' / 'another'.
# 'another' on a plural is also wrong ('another sheep' is OK because sheep
# is a zero-plural; 'another people' is not).
NO_ARTICLE_HEADS = frozenset(
    {
        "people",
        "spectators",
        "crowd",
        "uniforms",
        "belts",
        "trousers",
        "pants",
        "jeans",
        "shorts",
        "leggings",
        "gloves",
        "shoes",
        "boots",
        "sneakers",
        "sandals",
        "slippers",
        "books",
        "shelves",
        "chairs",
        "decorations",
        "lambs",
        "pictures",
        "carpeting",
    }
)


def _article_for(phrase: str, head_noun: str, *, repeat: bool) -> str:
    """Return article + space, or empty string for plurals/mass nouns.

    'a' / 'an' / 'another' for normal singulars; '' for plurals."""
    if head_noun in NO_ARTICLE_HEADS:
        return ""
    if repeat:
        return "another "
    return "an " if starts_with_vowel(phrase) else "a "


def build_benchmark_prompt(object_names: list[dict[str, Any]]) -> PromptSynthesis:
    """The object-enumeration prompt for a clip's named objects.

    Args:
        object_names: per object, in order (object ``i`` is ``<traj_i>``):
            ``noun`` (the head noun, e.g. "shopping cart"), ``phrase`` (the
            descriptor, e.g. "person in a blue shirt") and
            ``source_object_id`` (its index in ``objects/<stem>.json``, for
            the identity check).

    Every object's ``modifier`` is empty.
    """
    if not object_names:
        return PromptSynthesis(prompt="", objects=[], ok=False, errors=["no_instances"])

    clauses = _clauses(object_names)
    prompt = _join(clauses)

    words = prompt.split()
    obj_meta: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, c in enumerate(clauses):
        trajectory_token_word_index, noun_word_index = locate_object_words(words, i, c.noun)
        if trajectory_token_word_index is None:
            errors.append(f"<traj_{i}> not found in prompt")
        if noun_word_index is None:
            errors.append(f"noun {c.noun!r} not recoverable for object {i}")
        obj_meta.append(
            {
                "object_index": i,
                "source_object_id": int(object_names[i]["source_object_id"]),
                "noun": c.noun,
                "modifier": "",
                "noun_word_index": noun_word_index,
                "trajectory_token_word_index": trajectory_token_word_index,
            }
        )

    return PromptSynthesis(prompt=prompt, objects=obj_meta, ok=len(errors) == 0, errors=errors)


class _Clause(NamedTuple):
    text: str  # '<article> <phrase> <traj_i>'
    noun: str  # its head noun, lowercased


def _clauses(object_names: list[dict[str, Any]]) -> list[_Clause]:
    """Per object, ``<article> <phrase> <traj_i>``. The article is 'another'
    when an earlier object has the same head noun, whatever its descriptor
    ("a chair ... another blue chair")."""
    seen_nouns: defaultdict[str, int] = defaultdict(int)
    clauses = []
    for i, names in enumerate(object_names):
        phrase = (names.get("phrase") or names.get("noun") or "object").strip()
        noun = (names.get("noun") or phrase.split()[-1]).strip().lower()
        seen_nouns[noun] += 1
        article = _article_for(phrase, noun, repeat=seen_nouns[noun] > 1)
        clauses.append(_Clause(f"{article}{phrase} <traj_{i}>", noun))
    return clauses


def _join(clauses: list[_Clause]) -> str:
    """``clause_0[, clause_1, ...], and clause_N.``, capitalized."""
    body = clauses[0].text
    if len(clauses) > 1:
        body = ", ".join(c.text for c in clauses[:-1]) + f", and {clauses[-1].text}"
    prompt = body + "."
    prompt = prompt[0].upper() + prompt[1:]
    return re.sub(r"\s+", " ", prompt).strip()
