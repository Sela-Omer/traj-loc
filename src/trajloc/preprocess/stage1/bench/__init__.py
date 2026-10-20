"""Stage 1 of the evaluation benchmarks (MoveBench, MagicBench).

- 1A, :mod:`.object_tracking` (SAM 2 + CoTracker3): objects, masks,
  trajectories, highlight crops.
- 1B, :mod:`.object_naming` (Qwen2.5-VL): object names, prompts.

What differs between benchmarks is in :mod:`.benchmarks`.
"""
