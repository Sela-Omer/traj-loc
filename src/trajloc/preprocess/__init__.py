"""Data preprocessing: raw videos -> the cached tensors training reads.

Two stages, each a sequence of standalone steps:

- **Stage 1** (:mod:`trajloc.preprocess.stage1`): per clip, a 49-frame
  480x720 (HxW) mp4, per-object trajectories and a prompt whose words point at the
  objects. Layout under a dataset root::

      videos/<stem>.mp4  first_frames/<stem>.png
      trajectories/<stem>.json  prompts/<stem>.json

- **Stage 2** (:mod:`trajloc.preprocess.stage2`): per clip and backbone, the
  VAE latents and the text-encoder cache (prompt embedding with trajectory
  embeddings injected at the ``<traj_i>`` tokens), under
  ``<root>/cache/<backbone>/`` (see :mod:`trajloc.datasets.layout`).
"""
