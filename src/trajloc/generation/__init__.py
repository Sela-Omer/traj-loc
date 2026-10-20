"""Generation from a user's image, prompt and trajectories (``trajloc generate``).

:mod:`~trajloc.generation.scene` is the input format;
:mod:`~trajloc.generation.pipeline` runs it through the evaluation path.
"""

from trajloc.generation.pipeline import GenerateConfig, generate
from trajloc.generation.scene import Scene, SceneError, SceneObject, load_scene, parse_scene

__all__ = [
    "GenerateConfig",
    "Scene",
    "SceneError",
    "SceneObject",
    "generate",
    "load_scene",
    "parse_scene",
]
