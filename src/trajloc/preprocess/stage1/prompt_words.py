"""Where an object's words are in a Stage-1 prompt.

Every prompt record stores, per object, two word indices into
``prompt.split()``: its ``<traj_i>`` token and its head noun (the last word of
its label), which Stage 2 maps to token positions. Words are compared with
trailing commas and periods removed, case-insensitively for the noun.
"""

from __future__ import annotations


def starts_with_vowel(word: str) -> bool:
    """Whether ``word`` starts with a vowel letter (the 'a' / 'an' rule both
    prompt builders use; good enough for nouns)."""
    return word[:1].lower() in "aeiou" if word else False


def locate_object_words(
    words: list[str], object_index: int, noun: str, *, search_forward: bool = False
) -> tuple[int | None, int | None]:
    """``(trajectory_token_word_index, noun_word_index)`` of object ``object_index`` in ``words``.

    ``<traj_i>`` is its first occurrence. The noun (possibly several words)
    is searched backwards from ``<traj_i>``, then, with ``search_forward``,
    forwards; its index is that of its last word. ``None`` where not found.
    """
    trajectory_token = f"<traj_{object_index}>"
    trajectory_token_word_index = None
    for wi, w in enumerate(words):
        if w.rstrip(",.") == trajectory_token:
            trajectory_token_word_index = wi
            break
    noun_words = [w.rstrip(",.").lower() for w in noun.split()]
    n_w = len(noun_words)
    if trajectory_token_word_index is None or n_w == 0:
        return trajectory_token_word_index, None

    def matches_at(j: int) -> bool:
        return j + n_w <= len(words) and all(
            words[j + off].rstrip(",.").lower() == noun_words[off] for off in range(n_w)
        )

    match = next((j for j in range(trajectory_token_word_index - 1, -1, -1) if matches_at(j)), None)
    if match is None and search_forward:
        match = next(
            (
                j
                for j in range(trajectory_token_word_index + 1, len(words) - n_w + 1)
                if matches_at(j)
            ),
            None,
        )
    return trajectory_token_word_index, (match + n_w - 1) if match is not None else None
