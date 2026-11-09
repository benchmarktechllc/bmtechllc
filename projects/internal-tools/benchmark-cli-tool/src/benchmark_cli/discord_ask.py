"""Always-on Discord gateway bot that answers with the benchmark RAG agent.

Messages posted in ``DISCORD_ASK_CHANNEL_ID`` (and threads under that channel)
are answered in place. ``/ask`` is a separate entry point and works in every
channel of the guild. This process does not ingest Discord history; ingestion
stays in ``discord_sync``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

import discord
from discord import app_commands

from .agent import MODEL_UNAVAILABLE_REPLY, run_agent
from .contributions import (
    THANKS_REPLY,
    UNKNOWN_INVITE,
    is_contribution_reply,
    store_contribution,
)
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


def thread_question_for(content: str, starter: str | None) -> str | None:
    """The thread's starting question, when this message is a follow-up."""
    current = " ".join(content.split())
    prior = " ".join((starter or "").split())
    if not prior or prior == current:
        return None
    return prior


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


def _linked_list_answer(answer: str) -> bool:
    """Trello list replies already put each card link in the body."""
    return "Trello cards in " in answer and "](https://" in answer


def _summary_total(answer: str) -> int | None:
    """Sum of the per-list counts on the first line, such as ``Backlog 11, Doing 4``."""
    first = answer.splitlines()[0].strip() if answer else ""
    if not first or "Trello cards in " in first:
        return None
    total = 0
    for part in first.split(","):
        match = re.search(r"(\d+)\s*$", part.strip())
        if match is None:
            return None
        total += int(match.group(1))
    return total or None


def split_discord_messages(text: str, *, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split a long reply on line boundaries so each Discord message fits."""
    cleaned = text.strip() or "No answer text was returned."
    if len(cleaned) <= limit:
        return [cleaned]
    parts: list[str] = []
    buffer = ""
    for line in cleaned.split("\n"):
        piece = line if not buffer else f"{buffer}\n{line}"
        if len(piece) <= limit:
            buffer = piece
            continue
        if buffer:
            parts.append(buffer)
        if len(line) <= limit:
            buffer = line
            continue
        while line:
            parts.append(line[:limit])
            line = line[limit:]
        buffer = ""
    if buffer:
        parts.append(buffer)
    return parts or ["No answer text was returned."[:limit]]


def format_discord_messages(
    response: AnswerResponse, *, limit: int = DISCORD_MESSAGE_LIMIT
) -> list[str]:
    """One or more Discord messages. Long board lists are split, not dropped."""
    answer = response.answer.strip() or "No answer text was returned."
    if response.citations and _linked_list_answer(answer):
        shown = len(response.citations)
        total = _summary_total(answer)
        if total is not None and total != shown:
            footer = f"\n\n**Sources**\n{shown} shown of {total}; links are in the list above."
        else:
            noun = "card" if shown == 1 else "cards"
            footer = f"\n\n**Sources**\n{shown} indexed {noun}; links are in the list above."
        text = answer + footer
    elif response.citations:
        text = answer + "\n\n**Citations**\n" + "\n".join(
            _format_citation(index, citation)
            for index, citation in enumerate(response.citations, start=1)
        )
    else:
        text = answer
    if (
        response.status == "no_relevant_information"
        and UNKNOWN_INVITE not in text
        and MODEL_UNAVAILABLE_REPLY not in text
    ):
        text = f"{text.rstrip()}\n\n{UNKNOWN_INVITE}"
    return split_discord_messages(discord.utils.escape_mentions(text), limit=limit)


def format_discord_answer(response: AnswerResponse, *, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    """Render an agent response, keeping citations when they fit in one message."""
    answer = response.answer.strip() or "No answer text was returned."
    if (
        response.status == "no_relevant_information"
        and UNKNOWN_INVITE not in answer
        and MODEL_UNAVAILABLE_REPLY not in answer
    ):
        answer = f"{answer.rstrip()}\n\n{UNKNOWN_INVITE}"
    if response.citations and _linked_list_answer(answer):
        shown = len(response.citations)
        total = _summary_total(answer)
        if total is not None and total != shown:
            citation_block = f"**Sources**\n{shown} shown of {total}; links are in the list above."
        else:
            noun = "card" if shown == 1 else "cards"
            citation_block = f"**Sources**\n{shown} indexed {noun}; links are in the list above."
    elif response.citations:
        citation_block = "**Citations**\n" + "\n".join(
            _format_citation(index, citation)
            for index, citation in enumerate(response.citations, start=1)
        )
    else:
        citation_block = ""
    if citation_block:
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


async def read_thread_starter(message: discord.Message) -> str | None:
    """Text of the message that opened this thread, when Discord still has it."""
    channel = message.channel
    if not isinstance(channel, discord.Thread):
        return None
    starter = channel.starter_message
    if starter is None and channel.parent is not None:
        try:
            starter = await channel.parent.fetch_message(channel.id)
        except discord.HTTPException:
            logger.warning("Could not read the thread starter for message %s.", message.id)
            starter = None
    if starter is None or starter.id == message.id:
        return None
    return starter.content


class AddAnswerModal(discord.ui.Modal, title="Add answer"):
    """Collect a missing answer and store it for the next question."""

    answer = discord.ui.TextInput(
        label="Answer",
        style=discord.TextStyle.paragraph,
        max_length=2000,
        required=True,
    )

    def __init__(
        self,
        *,
        question: str,
        author_id: int,
        author_name: str,
        channel_id: int,
        message_id: int,
        message_url: str,
    ) -> None:
        super().__init__()
        self.question = question
        self.author_id = author_id
        self.author_name = author_name
        self.channel_id = channel_id
        self.message_id = message_id
        self.message_url = message_url

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            store_contribution(
                question=self.question,
                text=str(self.answer.value),
                author_id=self.author_id,
                author_name=self.author_name,
                channel_id=self.channel_id,
                message_id=self.message_id,
                message_url=self.message_url,
            )
        except (RuntimeError, ValueError):
            logger.exception("Could not store a contributed answer.")
            await interaction.response.send_message(
                "I couldn't save that. Please try again in a moment.",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(THANKS_REPLY, ephemeral=True)


ADD_ANSWER_CUSTOM_ID = "bmtech:add_answer"


async def original_question_message(bot_message: discord.Message) -> discord.Message | None:
    """The human question this unknown reply was posted for.

    After a restart the button has no saved state, so the question is read from
    the message the bot replied to, or from the thread starter.
    """
    reference = bot_message.reference
    if reference is not None:
        resolved = reference.resolved
        if isinstance(resolved, discord.Message):
            return resolved
        message_id = reference.message_id
        if message_id is not None:
            try:
                return await bot_message.channel.fetch_message(message_id)
            except (discord.HTTPException, AttributeError):
                logger.warning("Could not read the question message %s.", message_id)
    channel = bot_message.channel
    if isinstance(channel, discord.Thread):
        starter = channel.starter_message
        if starter is None and channel.parent is not None:
            try:
                starter = await channel.parent.fetch_message(channel.id)
            except discord.HTTPException:
                starter = None
        if starter is not None and starter.id != bot_message.id:
            return starter
    return None


class AddAnswerView(discord.ui.View):
    """Persistent button that opens the contribution modal."""

    def __init__(self) -> None:
        super().__init__(timeout=None)

    @discord.ui.button(
        label="Add answer",
        style=discord.ButtonStyle.primary,
        custom_id=ADD_ANSWER_CUSTOM_ID,
    )
    async def add_answer(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        bot_message = interaction.message
        original = await original_question_message(bot_message) if bot_message is not None else None
        target = original or bot_message
        question = original.content if original is not None else ""
        user = interaction.user
        if target is None:
            await interaction.response.send_message(
                "I couldn't find the original question. Reply to my message with the answer instead.",
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(
            AddAnswerModal(
                question=question,
                author_id=user.id,
                author_name=getattr(user, "display_name", "") or getattr(user, "name", "a teammate"),
                channel_id=target.channel.id,
                message_id=target.id,
                message_url=message_jump_url(target),
            )
        )


def message_jump_url(message: discord.Message) -> str:
    guild_id = message.guild.id if message.guild is not None else "@me"
    return f"https://discord.com/channels/{guild_id}/{message.channel.id}/{message.id}"


async def deliver_channel_messages(
    message: discord.Message,
    parts: list[str],
    *,
    view: discord.ui.View | None = None,
) -> None:
    """Post every part in the thread. A long answer is several messages, not one cut reply."""
    if not parts:
        return
    if isinstance(message.channel, discord.Thread):
        await message.reply(
            parts[0],
            mention_author=False,
            allowed_mentions=NO_MENTIONS,
            view=view if len(parts) == 1 else None,
        )
        for index, extra in enumerate(parts[1:], start=1):
            await message.channel.send(
                extra,
                allowed_mentions=NO_MENTIONS,
                view=view if index == len(parts) - 1 else None,
            )
        return
    existing = message.thread
    if existing is not None:
        try:
            for index, part in enumerate(parts):
                await existing.send(
                    part,
                    allowed_mentions=NO_MENTIONS,
                    view=view if index == len(parts) - 1 else None,
                )
            return
        except discord.HTTPException:
            logger.warning("Could not post in the existing thread for message %s.", message.id)
    try:
        thread = await message.create_thread(name=thread_name_for(message.content))
        for index, part in enumerate(parts):
            await thread.send(
                part,
                allowed_mentions=NO_MENTIONS,
                view=view if index == len(parts) - 1 else None,
            )
    except (discord.HTTPException, ValueError):
        logger.warning(
            "Could not open a thread on message %s; replying in the channel.",
            message.id,
        )
        await message.reply(
            parts[0],
            mention_author=False,
            allowed_mentions=NO_MENTIONS,
            view=view if len(parts) == 1 else None,
        )
        for index, extra in enumerate(parts[1:], start=1):
            await message.reply(
                extra,
                mention_author=False,
                allowed_mentions=NO_MENTIONS,
                view=view if index == len(parts) - 1 else None,
            )


async def deliver_channel_answer(message: discord.Message, body: str) -> None:
    """Reply in the current thread, or open one on the channel message."""
    await deliver_channel_messages(message, [body])


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
        self.add_view(AddAnswerView())
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

    async def answer_question(self, question: str, *, thread_question: str | None = None) -> str:
        """Run the RAG agent and format the reply.

        Questions run one at a time. Usage telemetry is appended without a file
        lock, and one in-flight retrieval is enough for this bot. A thread
        follow-up also receives the thread's starting question.
        """
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty.")

        def _invoke() -> AnswerResponse:
            if thread_question and "thread_question" in inspect.signature(self._agent).parameters:
                return self._agent(cleaned, thread_question=thread_question)
            return self._agent(cleaned)

        async with self._agent_lock:
            response = await asyncio.to_thread(_invoke)
        return format_discord_answer(response)

    async def answer_messages(self, question: str, *, thread_question: str | None = None) -> list[str]:
        """Format the reply as one Discord message per 2000 characters."""
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty.")

        def _invoke() -> AnswerResponse:
            if thread_question and "thread_question" in inspect.signature(self._agent).parameters:
                return self._agent(cleaned, thread_question=thread_question)
            return self._agent(cleaned)

        async with self._agent_lock:
            response = await asyncio.to_thread(_invoke)
        return format_discord_messages(response)

    async def answer_bundle(
        self, question: str, *, thread_question: str | None = None
    ) -> tuple[list[str], AnswerResponse]:
        """The Discord parts and the response, so an unknown answer can carry a button."""
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("question must not be empty.")

        def _invoke() -> AnswerResponse:
            if thread_question and "thread_question" in inspect.signature(self._agent).parameters:
                return self._agent(cleaned, thread_question=thread_question)
            return self._agent(cleaned)

        async with self._agent_lock:
            response = await asyncio.to_thread(_invoke)
        return format_discord_messages(response), response

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
        if await self._maybe_record_contribution(message):
            return
        logger.info("Answering message %s in channel %s.", message.id, message.channel.id)
        try:
            async with message.channel.typing():
                starter = await read_thread_starter(message)
                parts, response = await self.answer_bundle(
                    message.content,
                    thread_question=thread_question_for(message.content, starter),
                )
        except Exception:
            logger.exception("Failed to answer message %s.", message.id)
            try:
                await message.reply(FAILURE_REPLY, mention_author=False, allowed_mentions=NO_MENTIONS)
            except discord.HTTPException:
                logger.exception("Failed to send the error reply for message %s.", message.id)
            return
        view = None
        if response.status == "no_relevant_information" and MODEL_UNAVAILABLE_REPLY not in response.answer:
            view = AddAnswerView()
        try:
            await deliver_channel_messages(message, parts, view=view)
        except discord.HTTPException:
            logger.exception("Failed to deliver the answer for message %s.", message.id)

    async def _maybe_record_contribution(self, message: discord.Message) -> bool:
        """Store a human reply to an unknown answer. Bot messages and other replies are ignored."""
        reference = message.reference
        if reference is None or message.author.bot:
            return False
        channel = message.channel
        try:
            referenced = reference.cached_message or await channel.fetch_message(reference.message_id)
        except (discord.HTTPException, AttributeError):
            return False
        if referenced is None or not is_contribution_reply(
            author_is_bot=message.author.bot,
            referenced_author_is_bot=referenced.author.bot,
            referenced_text=referenced.content or "",
        ):
            return False
        question = message.content
        if referenced.reference is not None:
            try:
                original = await channel.fetch_message(referenced.reference.message_id)
                if original is not None and original.content.strip():
                    question = original.content
            except (discord.HTTPException, AttributeError):
                question = referenced.content
        try:
            store_contribution(
                question=question,
                text=message.content,
                author_id=message.author.id,
                author_name=getattr(message.author, "display_name", "") or message.author.name,
                channel_id=message.channel.id,
                message_id=message.id,
                message_url=message_jump_url(message),
            )
        except (RuntimeError, ValueError):
            logger.exception("Could not store a reply contribution for message %s.", message.id)
            return False
        try:
            await message.reply(THANKS_REPLY, mention_author=False, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException:
            logger.exception("Could not confirm contribution %s.", message.id)
        return True


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
        parts = await client.answer_messages(question)
    except Exception:
        logger.exception("Failed to answer /ask in channel %s.", interaction.channel_id)
        await _safe_followup(interaction, FAILURE_REPLY)
        return
    for part in parts:
        await _safe_followup(interaction, part)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    token, config = load_runtime()
    from .grounding import ANSWER_PATH_REVISION
    from .portal_urls import canonical_portal_base_url

    logger.info(
        "benchmark-discord-ask revision=%s portal_base_configured=%s",
        ANSWER_PATH_REVISION,
        bool(canonical_portal_base_url()),
    )
    AskClient(config).run(token)


if __name__ == "__main__":
    main()
