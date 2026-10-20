"""Prompt text normalization.

Wan's UMT5 tokenizer wrapper cleans every prompt before tokenizing it
(``wan/modules/tokenizers.py:basic_clean``): ``ftfy.fix_text``, two rounds
of ``html.unescape``, then ``strip``. Stage 2 (Wan's text cache) refuses to cache a
prompt whose cleaned tokens differ from its raw tokens, because the
``<traj_i>`` positions would then be computed on the wrong string. So
captions are stored already cleaned: a cleaned string is a fixed point of
the wrapper, and both backbones' encoders see the same bytes.
"""

from __future__ import annotations

import html

import ftfy


def basic_clean(text: str) -> str:
    """The Wan tokenizer wrapper's cleaning (``ftfy``, then HTML unescaping
    twice). Not idempotent in general (``&amp;amp;amp;`` unescapes further
    each time); Stage 2 checks that a prompt's tokens are unchanged by it."""
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()
