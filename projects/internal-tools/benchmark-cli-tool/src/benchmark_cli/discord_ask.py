"""Always-on Discord gateway bot that answers with the benchmark RAG agent.

Messages posted in ``DISCORD_ASK_CHANNEL_ID`` (and threads under that channel)
are answered in place. ``/ask`` is a separate entry point and works in every
channel of the guild. This process does not ingest Discord history; ingestion
stays in ``discord_sync``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import discord
from discord import app_commands

from .agent import run_agent
from .models import AnswerResponse, Citation

logger = logging.getLogger(__name__)

DISCORD_MESSAGE_LIMIT = 2000
THREAD_NAME_LIMIT = 100
_QUOTE_LIMIT = 300
_TRUNCATION_MARKER = "…(truncated)"
_SNOWFLAKE_LENGTH = range(17, 21)
ANSWERABLE_MESSAGE_TYPES = frozenset({discord.MessageType.default, discord.MessageType.reply})
NO_MENTIONS = discord.AllowedMentions.none()
EMPTY_QUESTION_REPLY = "Please include a question."
FAILURE_REPLY = "I couldn't answer that just now. Please try again in a moment."
_PLACEHOLDER_OPENAI_KEY = "your_openai_api_key_here"
_REQUIRED_ENV = (
    "DISCORD_ANSWER_BOT_TOKEN",
    "DISCORD_ASK_CHANNEL_ID",
    "DATABASE_URL",
    "OPENAI_API_KEY",
)

AgentRunner = Callable[[str], AnswerResponse]


@dataclass(frozen=True, slots=True)
class AskConfig:
    """Non-secret settings for the answer bot."""

    ask_channel_id: int
    guild_id: int | None


def parse_snowflake(value: str, *, name: str) -> int:
    """Parse a Discord snowflake from an environment value."""
    text = value.strip()
    if not text.isdigit() or len(text) not in _SNOWFLAKE_LENGTH:
        raise ValueError(f"{name} must be a Discord snowflake (17-20 digits).")
    return int(text)


def load_runtime() -> tuple[str, AskConfig]:
    """Read the bot token and channel settings from the environment."""
    missing = []
    for name in _REQUIRED_ENV:
        value = os.getenv(name, "").strip()
        if not value or (name == "OPENAI_API_KEY" and value == _PLACEHOLDER_OPENAI_KEY):
            missing.append(name)
    if missing:
        raise SystemExit(f"Missing required environment: {', '.join(missing)}.")
    try:
        channel_id = parse_snowflake(
            os.environ["DISCORD_ASK_CHANNEL_ID"], name="DISCORD_ASK_CHANNEL_ID"
        )
        guild_raw = os.getenv("DISCORD_ASK_GUILD_ID", "").strip()
        guild_id = parse_snowflake(guild_raw, name="DISCORD_ASK_GUILD_ID") if guild_raw else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return os.environ["DISCORD_ANSWER_BOT_TOKEN"].strip(), AskConfig(channel_id, guild_id)


def gateway_intents() -> discord.Intents:
    """Gateway intents for channel text plus slash commands."""
    intents = discord.Intents.default()
    intents.message_content = True
    return intents


def in_ask_channel(*, channel_id: int, parent_channel_id: int | None, ask_channel_id: int) -> bool:
    """Return whether a message lives in the ask channel or one of its threads."""
    return channel_id == ask_channel_id or parent_channel_id == ask_channel_id


def should_answer_channel_message(
    *,
    channel_id: int,
    parent_channel_id: int | None,
    author_is_bot: bool,
    author_id: int,
    bot_user_id: int | None,
    content: str,
    message_type: discord.MessageType,
    ask_channel_id: int,
) -> bool:
    """Channel messages are answered only for human text in the ask channel."""
    if author_is_bot or (bot_user_id is not None and author_id == bot_user_id):
        return False
    if message_type not in ANSWERABLE_MESSAGE_TYPES:
        return False
    if not content.strip():
        return False
    return in_ask_channel(
        channel_id=channel_id,
        parent_channel_id=parent_channel_id,
        ask_channel_id=ask_channel_id,
    )


def slash_reply_is_ephemeral(*, question: str) -> bool:
    """Blank slash questions are private. Every other answer is posted in-channel."""
    return not question.strip()


def thread_name_for(question: str) -> str:
    """Build a Discord thread title from the user's message."""
    collapsed = " ".join(question.split())
    if not collapsed:
        return "Question"
    if len(collapsed) <= THREAD_NAME_LIMIT:
        return collapsed
    return collapsed[: THREAD_NAME_LIMIT - 1].rstrip() + "…"


def truncate_discord(text: str, *, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    """Shorten text to Discord's message character limit."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    if len(text) <= limit:
        return text
    if limit <= len(_TRUNCATION_MARKER):
        return _TRUNCATION_MARKER[:limit]
    return text[: limit - len(_TRUNCATION_MARKER)].rstrip() + _TRUNCATION_MARKER


def _format_citation(index: int, citation: Citation) -> str:
    source = " ".join(citation.source_file.split()).replace("`", "'")
    quote = " ".join(citation.quote.split())
    if len(quote) > _QUOTE_LIMIT:
        quote = quote[: _QUOTE_LIMIT - 1].rstrip() + "…"
    return f"{index}. `{source}` (chunk {citation.chunk_id}): {quote}"


def format_discord_answer(response: AnswerResponse, *, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    """Render an agent response, keeping citations when they fit in one message."""
    answer = response.answer.strip() or "No answer text was returned."
    if response.citations:
        citation_block = "**Citations**\n" + "\n".join(
            _format_citation(index, citation)
            for index, citation in enumerate(response.citations, start=1)
        )
        separator = "\n\n"
        if len(separator) + len(citation_block) < limit:
            answer_limit = limit - len(separator) - len(citation_block)
            answer = truncate_discord(answer, limit=answer_limit)
            composed = f"{answer}{separator}{citation_block}"
        else:
            composed = truncate_discord(f"{answer}{separator}{citation_block}", limit=limit)
    else:
        composed = truncate_discord(answer, limit=limit)
    rendered = truncate_discord(discord.utils.escape_mentions(composed), limit=limit)
    return rendered or "No answer text was returned."[:limit]


async def deliver_channel_answer(message: discord.Message, body: str) -> None:
    """Reply in the current thread, or open one on the channel message."""
    if isinstance(message.channel, discord.Thread):
        await message.reply(body, mention_author=False, allowed_mentions=NO_MENTIONS)
        return
    existing = message.thread
    if existing is not None:
        try:
            await existing.send(body, allowed_mentions=NO_MENTIONS)
            return
        except discord.HTTPException:
            logger.warning("Could not post in the existing thread for message %s.", message.id)
    try:
        thread = await message.create_thread(name=thread_name_for(message.content))
        await thread.send(body, allowed_mentions=NO_MENTIONS)
    except (discord.HTTPException, ValueError):
        logger.warning(
            "Could not open a thread on message %s; replying in the channel.",
            message.id,
        )
        await message.reply(body, mention_author=False, allowed_mentions=NO_MENTIONS)


async def _safe_followup(interaction: discord.Interaction, body: str) -> None:
    try:
        await interaction.followup.send(body, allowed_mentions=NO_MENTIONS)
    except discord.HTTPException:
        logger.exception("Could not send /ask followup in channel %s.", interaction.channel_id)


class AskClient(discord.Client):
    """Gateway client for the ask channel and the guild-wide `/ask` command."""

    def __init__(self, config: AskConfig, *, agent: AgentRunner = run_agent) -> None:
        super().__init__(intents=gateway_intents())
        self.config = config
        self._agent = agent
        self._agent_lock = asyncio.Lock()
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        self.tree.add_command(ask_command)
        await self.sync_commands()

    async def sync_commands(self) -> None:
        """Register `/ask` for one guild, or globally when no guild id is set."""
        try:
            if self.config.guild_id is None:
                synced = await self.tree.sync()
                logger.info(
                    "Synced %s global command(s). Global commands can take up to an hour to appear.",
                    len(synced),
                )
                return
            guild = discord.Object(id=self.config.guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info(
                "Synced %s command(s) to guild %s. /ask works in every channel of that guild.",
                len(synced),
                self.config.guild_id,
            )
        except discord.HTTPException:
            logger.exception(
                "Slash-command sync failed. Messages in the ask channel are still answered. "
                "Restart after the bot is invited with the applications.commands scope."
            )

    async def on_ready(self) -> None:
        logger.info(
            "Logged in as %s. Answering messages in channel %s.",
            self.user,
            self.config.ask_channel_id,
        )

    async def answer_question(self, question: str) -> str:
        """Run the RAG agent and format the reply.

        Questions run one at a time. Usage telemetry is appended without a file
        lock, and one in-flight retrieval is enough for this bot.
        """
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty.")
        async with self._agent_lock:
            response = await asyncio.to_thread(self._agent, cleaned)
        return format_discord_answer(response)

    async def on_message(self, message: discord.Message) -> None:
        channel = message.channel
        parent_id = channel.parent_id if isinstance(channel, discord.Thread) else None
        bot_user_id = self.user.id if self.user is not None else None
        if not should_answer_channel_message(
            channel_id=message.channel.id,
            parent_channel_id=parent_id,
            author_is_bot=message.author.bot,
            author_id=message.author.id,
            bot_user_id=bot_user_id,
            content=message.content,
            message_type=message.type,
            ask_channel_id=self.config.ask_channel_id,
        ):
            if (
                in_ask_channel(
                    channel_id=message.channel.id,
                    parent_channel_id=parent_id,
                    ask_channel_id=self.config.ask_channel_id,
                )
                and not message.author.bot
                and message.type in ANSWERABLE_MESSAGE_TYPES
                and not message.content.strip()
            ):
                logger.info(
                    "Skipping message %s because it has no text. Enable the Message Content "
                    "Intent if text messages in the ask channel arrive empty.",
                    message.id,
                )
            return
        logger.info("Answering message %s in channel %s.", message.id, message.channel.id)
        try:
            async with message.channel.typing():
                body = await self.answer_question(message.content)
        except Exception:
            logger.exception("Failed to answer message %s.", message.id)
            try:
                await message.reply(FAILURE_REPLY, mention_author=False, allowed_mentions=NO_MENTIONS)
            except discord.HTTPException:
                logger.exception("Failed to send the error reply for message %s.", message.id)
            return
        try:
            await deliver_channel_answer(message, body)
        except discord.HTTPException:
            logger.exception("Failed to deliver the answer for message %s.", message.id)


@app_commands.command(name="ask", description="Ask the company knowledge agent.")
@app_commands.guild_only()
@app_commands.describe(question="Question to answer from company evidence.")
async def ask_command(interaction: discord.Interaction, question: str) -> None:
    """Answer in whichever guild channel the command was used."""
    client = interaction.client
    if not isinstance(client, AskClient):
        raise RuntimeError("The /ask command is registered on the answer bot.")
    if slash_reply_is_ephemeral(question=question):
        await interaction.response.send_message(EMPTY_QUESTION_REPLY, ephemeral=True)
        return
    await interaction.response.defer(thinking=True, ephemeral=False)
    try:
        body = await client.answer_question(question)
    except Exception:
        logger.exception("Failed to answer /ask in channel %s.", interaction.channel_id)
        await _safe_followup(interaction, FAILURE_REPLY)
        return
    await _safe_followup(interaction, body)


def main() -> None:
    token, config = load_runtime()
    AskClient(config).run(token)


if __name__ == "__main__":
    main()
