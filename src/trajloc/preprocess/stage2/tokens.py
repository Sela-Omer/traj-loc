"""Where each object's words land in a tokenized prompt.

Stage 1 records, per object, the word index (in ``prompt.split()``) of its
head noun and of its ``<traj_i>`` token. A text encoder reads tokens, so each
word index is mapped to the position of the word's first token, through the
tokenizer's character offsets.
"""

from __future__ import annotations

from dataclasses import dataclass


def word_char_spans(prompt: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` character span of each whitespace-split word of
    ``prompt`` (as many as ``prompt.split()``), in one pass over the prompt."""
    words = prompt.split()
    spans: list[tuple[int, int]] = []
    cursor = 0
    for w in words:
        s = prompt.find(w, cursor)
        if s == -1:
            raise RuntimeError(
                f"word_char_spans: could not locate word {w!r} "
                f"in prompt starting at cursor={cursor}: {prompt!r}"
            )
        spans.append((s, s + len(w)))
        cursor = s + len(w)
    return spans


@dataclass
class ObjectTokenPositions:
    """Result of mapping Stage-1 word indices to token sequence positions."""

    trajectory_token_positions: dict[int, int]  # object_index -> token seq position of <traj_i>
    noun_token_positions: dict[int, int]  # object_index -> first sub-token pos of the noun


def map_word_indices_to_tokens(
    prompt: str,
    offset_mapping: list[tuple[int, int]],
    objects_meta: list[dict],
) -> ObjectTokenPositions:
    """Map Stage-1 ``(noun_word_index, trajectory_token_word_index)`` to token positions.

    Uses the tokenizer's ``offset_mapping`` (already produced by tokenization,
    no second tokenizer needed) to find the contiguous token range covering
    each word. The noun word's first sub-token becomes
    ``noun_token_positions[i]``, the token of ``<traj_i>``
    ``trajectory_token_positions[i]``. Raises if an expected word has no token.
    """
    word_spans = word_char_spans(prompt)
    num_words = len(word_spans)

    # The first token of each word, from the offsets: a token belongs to word
    # k if its (non-empty) character span lies inside word k's span (a
    # sub-token of the word, or a token spanning exactly the word).
    first_token_of_word: dict[int, int] = {}
    for token_position, (start, end) in enumerate(offset_mapping):
        if start == end:
            # Special tokens (</s>, padding) and SentencePiece's bare '▁'
            # markers, which share an offset with the next token: not part
            # of any word.
            continue
        for k in range(num_words):
            word_start, word_end = word_spans[k]
            if start >= word_start and end <= word_end:
                if k not in first_token_of_word:
                    first_token_of_word[k] = token_position
                break

    trajectory_token_positions: dict[int, int] = {}
    noun_token_positions: dict[int, int] = {}
    for meta in objects_meta:
        object_index = meta["object_index"]
        noun_word = meta["noun_word_index"]
        trajectory_word = meta["trajectory_token_word_index"]

        if trajectory_word not in first_token_of_word:
            last_end = max((end for start, end in offset_mapping if start != end), default=0)
            cut = word_spans[trajectory_word][0] >= last_end  # past the tokens the encoder read
            hint = " (the prompt is longer than the text encoder reads: shorten it)" if cut else ""
            raise RuntimeError(
                f"map_word_indices_to_tokens: <traj_{object_index}> at word index "
                f"{trajectory_word} has no matching token{hint} (offset_mapping len="
                f"{len(offset_mapping)}). prompt: {prompt!r}"
            )
        if noun_word not in first_token_of_word:
            raise RuntimeError(
                f"map_word_indices_to_tokens: noun {meta['noun']!r} of object {object_index} "
                f"at word index {noun_word} has no matching token. prompt: {prompt!r}"
            )
        trajectory_token_positions[object_index] = first_token_of_word[trajectory_word]
        noun_token_positions[object_index] = first_token_of_word[noun_word]

    return ObjectTokenPositions(
        trajectory_token_positions=trajectory_token_positions,
        noun_token_positions=noun_token_positions,
    )
