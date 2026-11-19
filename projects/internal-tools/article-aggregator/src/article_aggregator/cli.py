"""Command line entry point: `article-aggregator digest`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from article_aggregator.config import ConfigError, load_config
from article_aggregator.digest import AllFeedsFailedError, PartialPostError, run_digest
from article_aggregator.discord.client import DiscordClient, DiscordError

TOKEN_VAR = "DISCORD_DIGEST_BOT_TOKEN"
CHANNEL_VARS = {
    "production": "DISCORD_DIGEST_CHANNEL_ID",
    "test": "DISCORD_DIGEST_TEST_CHANNEL_ID",
}

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="article-aggregator")
    commands = parser.add_subparsers(dest="command", required=True)
    digest = commands.add_parser("digest", help="fetch feeds and post today's digest")
    digest.add_argument("--dry-run", action="store_true", help="print the payloads, do not post")
    digest.add_argument(
        "--target",
        choices=sorted(CHANNEL_VARS),
        default="test",
        help="which channel to post to (default: test)",
    )
    digest.add_argument(
        "--config-dir", type=Path, default=Path("config"), help="directory with the YAML config"
    )
    return parser


def _require(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"environment variable {name} is not set")
    return value


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    env = os.environ if env is None else env
    log = logging.getLogger("article_aggregator")

    try:
        config = load_config(args.config_dir)
        if not args.dry_run:
            token = _require(env, TOKEN_VAR)
            channel_id = _require(env, CHANNEL_VARS[args.target])
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG

    now = datetime.now(UTC)
    try:
        if args.dry_run:
            result = run_digest(config, now=now, poster=None)
        else:
            with DiscordClient(token) as client:
                result = run_digest(
                    config, now=now, poster=lambda payload: client.post_message(channel_id, payload)
                )
    except (AllFeedsFailedError, DiscordError, PartialPostError) as exc:
        log.error("%s", exc)
        return EXIT_FAILED

    if args.dry_run:
        if not result.payloads:
            print("No matches in the last 24 h; nothing would be posted.")
        else:
            json.dump(result.payloads, sys.stdout, indent=2, ensure_ascii=False)
            print()
    elif result.message_ids:
        log.info("posted %d messages to the %s channel", len(result.message_ids), args.target)
    else:
        log.info("quiet day: no matches, nothing posted")
    return EXIT_OK
