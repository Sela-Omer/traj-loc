"""Stage 2: Stage-1 clips -> the cached tensors training and inference read.

Every step is a :class:`~trajloc.preprocess.stage2.step.Stage2Step`: resumable
and shardable, one ``.pt`` per clip under ``<root>/cache/<backbone>/``. The
steps are backbone-independent; each backbone supplies its encoders
(:class:`~trajloc.backbones.base.VideoEncoder`,
:class:`~trajloc.backbones.base.PromptEncoder`) through its
:class:`~trajloc.backbones.base.BackboneSpec`.

- :class:`~.latents.LatentCacheStep`: VAE latents (``latents/``).
- :class:`~.text_cache.TextCacheStep`: the text caches (``text/``): the
  prompt embedding, with each ``<traj_i>`` token's input embedding replaced
  by the frozen trajectory encoder's embedding of object ``i``'s trajectory;
  the token positions of each object's ``<traj_i>`` and noun; and the
  trajectory tensors (:class:`~trajloc.datasets.caches.TextCache`).

Shared by the steps: :mod:`.clip_inputs` (reading a clip's Stage-1 files),
:mod:`.tokens` (where each object's words land in the tokenized prompt) and
:mod:`.models` (the released trajectory encoder).
"""
