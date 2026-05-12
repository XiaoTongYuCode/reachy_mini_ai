from __future__ import annotations

import sys
import argparse
import logging
import subprocess
from collections.abc import Sequence

from reachy_mini_hf_realtime_gateway.config import ConfigError, config_from_env, load_dotenv_for_gateway
from reachy_mini_hf_realtime_gateway.healthcheck import run_healthcheck
from reachy_mini_hf_realtime_gateway.command_builder import build_command, format_command


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reachy-mini-hf-realtime-gateway",
        description="Start a local speech-to-speech OpenAI Realtime-compatible gateway for Reachy Mini.",
    )
    parser.add_argument("--env-file", help="Path to a gateway .env file. Defaults to nearest .env from cwd.")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated command without starting it.")
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Probe the configured /v1/realtime endpoint instead of starting the gateway.",
    )
    parser.add_argument("--healthcheck-timeout", type=float, default=5.0, help="Healthcheck timeout in seconds.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level for the wrapper.")
    return parser


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def run_gateway(argv: Sequence[str] | None = None) -> int:
    """Run the gateway wrapper and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        load_dotenv_for_gateway(args.env_file)
        config = config_from_env()
        command = build_command(config)
    except ConfigError as exc:
        parser.error(str(exc))
        return 2

    if args.healthcheck:
        result = run_healthcheck(config, timeout_seconds=args.healthcheck_timeout)
        message = f"{result.url}: {result.message}"
        if result.ok:
            print(f"ok {message}")
            return 0
        print(f"error {message}", file=sys.stderr)
        return 1

    rendered_command = format_command(command)
    if args.dry_run:
        print(rendered_command)
        return 0

    logger.info("Starting gateway: %s", rendered_command)
    logger.info("Reachy app URL: %s", config.realtime_url)
    try:
        process = subprocess.Popen(command)
    except FileNotFoundError:
        logger.error(
            "Could not find %r. Install this service package first, then ensure speech-to-speech is on PATH.",
            config.speech_to_speech_bin,
        )
        return 127
    except OSError as exc:
        logger.error("Failed to start gateway: %s", exc)
        return 1

    try:
        return int(process.wait())
    except KeyboardInterrupt:
        logger.info("Stopping gateway...")
        process.terminate()
        try:
            return int(process.wait(timeout=10))
        except subprocess.TimeoutExpired:
            process.kill()
            return int(process.wait())


def main(argv: Sequence[str] | None = None) -> int:
    return run_gateway(argv)


if __name__ == "__main__":
    raise SystemExit(main())
