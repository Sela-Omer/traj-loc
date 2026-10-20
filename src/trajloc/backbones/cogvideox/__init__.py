"""CogVideoX-5B I2V (via diffusers and CogVideo's finetuning code).

Modules: ``constants``; ``supervisor`` / ``attention`` (the
joint-attention processor: the additive correction); ``modules`` (building TrajLoc's modules);
``trainer`` (the upstream trainer, extended) and ``training`` (its entry
point); ``inference`` (the generator); ``upstream`` (import-time shims).
Importing this package imports none of them.
"""
