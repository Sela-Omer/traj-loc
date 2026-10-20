"""The ``trajloc`` command: train, evaluate and use the TrajLoc models.

    trajloc train    <backbone> --config configs/<backbone>.yaml --run_name <name>
    trajloc infer    <backbone> --run_name <name>
    trajloc generate <backbone> --image <png> --scene <json> --run_name <name>

``<backbone>`` is ``cogvideox`` or ``wan``; ``trajloc <command> <backbone>
--help`` lists the options with that backbone's released defaults. ``infer``
and ``generate`` use the released model unless given a ``--checkpoint``
(downloaded into ``checkpoints/`` on first use). Outputs go to
``runs/<command>/<backbone>/<run_name>/``. Paths given on the command
line are taken from the working directory; paths inside the configs are
relative to the repository root, which the commands run from.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from trajloc import hub
from trajloc.backbones import BACKBONES, BackboneSpec
from trajloc.cli.arguments import (
    DefaultsHelpFormatter,
    add_shard_arguments,
    add_stems_argument,
    check_shard_arguments,
    stems_from_args,
)
from trajloc.logs import setup_logging
from trajloc.paths import project_root

COMMANDS: dict[str, str] = {
    "train": "train a TrajLoc model from a config",
    "infer": "generate the evaluation sets with a checkpoint",
    "generate": "generate a video from an image, a prompt and trajectories",
}


def main(argv: list[str] | None = None) -> int:
    """The ``trajloc`` entry point."""
    p = argparse.ArgumentParser(
        prog="trajloc",
        description=__doc__.split("\n\n")[0].replace("``", ""),
        epilog="Set TRAJLOC_DEBUG=1 to see an error's full traceback.",
    )
    commands = p.add_subparsers(dest="command", required=True, metavar="<command>")
    for command, help_text in COMMANDS.items():
        backbones = commands.add_parser(command, help=help_text, description=help_text)
        per_backbone = backbones.add_subparsers(
            dest="backbone", required=True, metavar="<backbone>"
        )
        for spec in BACKBONES.values():
            sub = per_backbone.add_parser(
                spec.name,
                help=spec.display_name,
                description=f"{help_text} ({spec.display_name}).",
                formatter_class=DefaultsHelpFormatter,
            )
            _ARGUMENTS[command](sub, spec)
            sub.set_defaults(run=_RUNNERS[command], parser=sub, spec=spec)
    args = p.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runner: Runner = args.run
    try:
        return runner(args.parser, args.spec, args)
    except USER_ERRORS as e:  # a bad input, not a bug: no traceback (unless TRAJLOC_DEBUG=1)
        if os.environ.get("TRAJLOC_DEBUG"):
            raise
        print(f"trajloc {args.command}: error: {e}", file=sys.stderr)
        return 1


# The errors a command reports as a message: missing or existing files, and
# invalid configs, scenes or selections (SceneError is a ValueError).
USER_ERRORS = (FileNotFoundError, FileExistsError, ValueError)


# ------------------------------------------------------------------- train


def _train_arguments(p: argparse.ArgumentParser, spec: BackboneSpec) -> None:
    p.add_argument(
        "--config", required=True, type=Path, help=f"the training config (configs/{spec.name}.yaml)"
    )
    p.add_argument(
        "--run_name", required=True, help=f"output directory: runs/train/{spec.name}/<run_name>/"
    )
    p.add_argument("--train_steps", type=int, default=None, help="override the config's")
    p.add_argument("--batch_size", type=int, default=None, help="override the config's")
    p.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="override the config's (save every N steps)",
    )


def _train(p: argparse.ArgumentParser, spec: BackboneSpec, args: argparse.Namespace) -> int:
    from trajloc.training import run_training

    config = _existing_file(p, args.config, "--config")
    root = _enter_project_root()
    run_training(
        spec.name,
        config,
        args.run_name,
        project_root=root,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        checkpointing_steps=args.checkpointing_steps,
    )
    return 0


# ------------------------------------------------------------------- infer


def _infer_arguments(p: argparse.ArgumentParser, spec: BackboneSpec) -> None:
    from trajloc.datasets import EVAL_SETS

    _add_model_arguments(p, spec, command="infer")
    p.add_argument("--max_samples", type=int, default=None, help="first N clips of each set")
    add_stems_argument(
        p,
        "only these clips (numbered within the selection for --seed, so their seeds differ "
        "from a full run's)",
    )
    p.add_argument(
        "--eval_sets",
        nargs="+",
        default=None,
        choices=sorted(EVAL_SETS),
        help="evaluation sets (default: all)",
    )
    add_shard_arguments(p)


def _infer(p: argparse.ArgumentParser, spec: BackboneSpec, args: argparse.Namespace) -> int:
    from trajloc.inference import InferenceConfig, run_inference

    check_shard_arguments(p, args)
    if args.max_samples is not None and args.max_samples < 1:
        p.error("--max_samples must be at least 1")
    checkpoint, config = _checkpoint_and_config(p, spec, args)
    root = _enter_project_root()
    run_inference(
        spec.name,
        InferenceConfig(
            training_config_path=config,
            checkpoint_dir=checkpoint,
            run_name=args.run_name,
            project_root=root,
            device=args.device,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            shift=getattr(args, "shift", None),
            seed=args.seed,
            max_samples=args.max_samples,
            eval_sets=args.eval_sets,
            stems=stems_from_args(args.stems),
            shard=args.shard,
            num_shards=args.num_shards,
        ),
    )
    return 0


# ---------------------------------------------------------------- generate


def _generate_arguments(p: argparse.ArgumentParser, spec: BackboneSpec) -> None:
    _add_model_arguments(p, spec, command="generate")
    p.add_argument(
        "--image", required=True, type=Path, help="the first frame (center-cropped to 3:2)"
    )
    p.add_argument(
        "--scene",
        required=True,
        type=Path,
        help="the prompt and the objects' paths (a scene.json; see tools/scene_editor.html)",
    )
    p.add_argument(
        "--trajectory_encoder",
        type=Path,
        default=None,
        help="the trajectory encoder's weights (default: the checkpoint's "
        "trajectory_encoder.safetensors, else checkpoints/trajectory_encoder.safetensors)",
    )


def _generate(p: argparse.ArgumentParser, spec: BackboneSpec, args: argparse.Namespace) -> int:
    from trajloc.generation import GenerateConfig, SceneError, generate, load_scene

    checkpoint, config = _checkpoint_and_config(p, spec, args)
    image = _existing_file(p, args.image, "--image")
    try:
        scene = load_scene(_existing_file(p, args.scene, "--scene"))
    except SceneError as e:
        p.error(str(e))
    encoder = args.trajectory_encoder or _default_trajectory_encoder(checkpoint)
    encoder = _existing_file(p, encoder, "--trajectory_encoder")
    root = _enter_project_root()
    run_dir = root / "runs" / "generate" / spec.name / args.run_name
    if run_dir.exists():
        p.error(f"{run_dir} exists: use a new --run_name")
    generate(
        spec.name,
        scene,
        GenerateConfig(
            image_path=image,
            scene_path=args.scene.resolve(),
            training_config_path=config,
            checkpoint_dir=checkpoint,
            trajectory_encoder_path=encoder,
            run_dir=run_dir,
            device=args.device,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            shift=getattr(args, "shift", None),
            seed=args.seed,
        ),
    )
    return 0


# ----------------------------------------------------------------- helpers


def _add_model_arguments(p: argparse.ArgumentParser, spec: BackboneSpec, *, command: str) -> None:
    """The checkpoint, the run and the sampling settings (released defaults)."""
    p.add_argument(
        "--checkpoint",
        default=spec.release_repo,
        help="a TrajLoc checkpoint: a directory (a training run's checkpoint-<step>/) or a "
        "Hugging Face Hub repository, downloaded into checkpoints/<name>/ on first use",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="the checkpoint's training config (default: <checkpoint>/config.yaml)",
    )
    p.add_argument(
        "--run_name",
        required=True,
        help=f"output directory: runs/{command}/{spec.name}/<run_name>/",
    )
    p.add_argument("--device", default="cuda:0", help="torch device")
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=spec.default_guidance_scale,
        help="classifier-free guidance scale",
    )
    p.add_argument("--num_steps", type=int, default=50, help="sampling steps")
    if spec.default_shift is not None:
        p.add_argument(
            "--shift", type=float, default=spec.default_shift, help="flow-matching scheduler shift"
        )
    seed_help = "clip i is generated with seed + i" if command == "infer" else "the sampling seed"
    p.add_argument("--seed", type=int, default=42, help=seed_help)


def _checkpoint_and_config(
    p: argparse.ArgumentParser, spec: BackboneSpec, args: argparse.Namespace
) -> tuple[Path, Path]:
    """The checkpoint directory (downloaded first if ``--checkpoint`` names a
    Hub repository) and its training config, resolved; a usage error if
    either is missing."""
    checkpoint = _checkpoint_dir(spec, args.checkpoint)
    missing = spec.missing_checkpoint_files(checkpoint)
    if missing:
        no = ", ".join(missing)
        p.error(f"{args.checkpoint} is not a {spec.display_name} checkpoint: no {no}")
    config = args.config if args.config is not None else checkpoint / "config.yaml"
    return checkpoint.resolve(), _existing_file(p, config, "--config")


def _checkpoint_dir(spec: BackboneSpec, checkpoint: str) -> Path:
    """``checkpoint`` as a directory: a local one as is; a Hub repository's
    download, ``checkpoints/<name>/``, fetched unless already complete."""
    if not hub.is_repo_id(checkpoint):
        return Path(checkpoint)
    local = hub.local_dir_for(checkpoint)
    files = (*spec.checkpoint_files, "config.yaml", TRAJECTORY_ENCODER)
    if any(not (local / f).is_file() for f in files):
        revision = spec.release_revision if checkpoint == spec.release_repo else None
        try:
            hub.fetch(checkpoint, local, revision=revision, allow_patterns=files)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{checkpoint} is not a directory, and {e}") from e
    return local


def _existing_file(p: argparse.ArgumentParser, path: Path, option: str) -> Path:
    if not path.is_file():
        p.error(f"{option}: no file at {path}")
    return path.resolve()


TRAJECTORY_ENCODER = "trajectory_encoder.safetensors"


def _default_trajectory_encoder(checkpoint: Path) -> Path:
    """A released model's own trajectory encoder; for a checkpoint of a run
    trained here, the one Stage 2 used (``checkpoints/``)."""
    bundled = checkpoint / TRAJECTORY_ENCODER
    return bundled if bundled.is_file() else project_root() / "checkpoints" / TRAJECTORY_ENCODER


def _enter_project_root() -> Path:
    """``chdir`` to the repository root, where the configs' paths start (call
    after resolving the command-line paths), and set up logging."""
    root = project_root()
    os.chdir(root)
    setup_logging()
    return root


Arguments = Callable[[argparse.ArgumentParser, BackboneSpec], None]
Runner = Callable[[argparse.ArgumentParser, BackboneSpec, argparse.Namespace], int]
_ARGUMENTS: dict[str, Arguments] = {
    "train": _train_arguments,
    "infer": _infer_arguments,
    "generate": _generate_arguments,
}
_RUNNERS: dict[str, Runner] = {"train": _train, "infer": _infer, "generate": _generate}


if __name__ == "__main__":
    sys.exit(main())
