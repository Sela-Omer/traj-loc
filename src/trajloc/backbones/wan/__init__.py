"""Wan 2.1 I2V-14B (via DiffSynth-Studio).

Modules: ``constants``; ``supervisor`` / ``attention`` (exact cross-attention
column replacement); ``model`` (the training module) and ``config`` (its
``WanModelSettings``); ``loss``; ``checkpoint``; ``training`` (the training
loop); ``inference`` (the generator); ``text_encoder`` (UMT5 with input
embeddings, for Stage 2); ``upstream`` (the facade over the pinned Wan2.1 and
DiffSynth-Studio checkouts). Importing this package imports none of them.
"""
