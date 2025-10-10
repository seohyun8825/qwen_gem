"""Wrapper around llava.eval.infer that registers VTimeLLM config first."""

from __future__ import annotations

import os
import sys
import argparse


def parse_args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    wrapper_args = parse_args_from_cli()
    os.environ.setdefault(
        "TRANSFORMERS_ATTENTION_IMPLEMENTATION", wrapper_args.attn_implementation
    )
    # Ensure VTimeLLM modules are imported so AutoConfig registers "VTimeLLM".
    try:
        import vtimellm.model.vtimellm_llama  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "vtimellm package not found on PYTHONPATH."
        ) from exc

    from llava.eval.infer import main as infer_main

    infer_main()


if __name__ == "__main__":
    sys.exit(main())
