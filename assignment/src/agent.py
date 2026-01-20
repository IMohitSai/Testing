from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .settings import Settings

logger = logging.getLogger("placementsprint.agent")


Mode = Literal["auto", "plan", "resume", "interview"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ActionItem(BaseModel):
    title: str = Field(min_length=1, max_length=140)
    why: str = Field(min_length=1, max_length=200)
    eta_minutes: int = Field(ge=1, le=240)
    priority: Literal["low", "med", "high"]


class AgentResponse(BaseModel):
    reply_markdown: str = Field(min_length=1, max_length=12000)
    action_items: list[ActionItem] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Intent(BaseModel):
    intent: Mode
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=300)


def _build_openrouter_model(settings: Settings, model_id: str) -> OpenAIChatModel:
    # OpenRouter is OpenAI-compatible and uses base_url https://openrouter.ai/api/v1 :contentReference[oaicite:4]{index=4}
    headers: dict[str, str] = {}
    if settings.site_url:
        headers["HTTP-Referer"] = settings.site_url
    if settings.app_name:
        headers["X-Title"] = settings.app_name

    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=headers or None,
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIChatModel(model_id, provider=provider)


@dataclass
class Orchestrator:
    intent_agent_primary: Agent[None, Intent]
    intent_agent_fallback: Agent[None, Intent]
    main_agent_primary: Agent[None, AgentResponse]
    main_agent_fallback: Agent[None, AgentResponse]

    async def _run_with_retries(self, run_fn, *, max_attempts: int = 3):
        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await run_fn()
            except Exception as e:
                last_err = e
                sleep_s = min(2.0 * attempt, 6.0)
                logger.warning("attempt=%s failed: %s; retrying in %.1fs", attempt, repr(e), sleep_s)
                await asyncio.sleep(sleep_s)
        assert last_err is not None
        raise last_err

    @staticmethod
    def _format_history(messages: list[ChatMessage], keep_last: int = 12) -> str:
        trimmed = messages[-keep_last:]
        lines = []
        for m in trimmed:
            role = "USER" if m.role == "user" else "ASSISTANT"
            lines.append(f"{role}: {m.content.strip()}")
        return "\n".join(lines)

    async def classify_intent(self, latest_user_text: str) -> Intent:
        prompt = (
            "Classify the user's intent into one of: auto, plan, resume, interview.\n"
            "Return the intent with confidence and a short rationale.\n\n"
            f"USER_MESSAGE:\n{latest_user_text}"
        )

        async def primary():
            r = await self.intent_agent_primary.run(prompt)
            return r.output

        async def fallback():
            r = await self.intent_agent_fallback.run(prompt)
            return r.output

        try:
            return await self._run_with_retries(primary)
        except Exception:
            logger.exception("intent primary failed; switching to fallback")
            return await self._run_with_retries(fallback)

    async def respond(self, messages: list[ChatMessage], mode: Mode) -> AgentResponse:
        if not messages or messages[-1].role != "user":
            raise ValueError("Last message must be from the user.")

        history = self._format_history(messages)
        latest = messages[-1].content.strip()

        resolved_mode = mode
        if mode == "auto":
            intent = await self.classify_intent(latest)
            # If low confidence, keep auto; otherwise follow classifier
            if intent.confidence >= 0.55:
                resolved_mode = intent.intent

        system_context = (
            "You are PlacementSprint, a practical placement-prep agent.\n"
            "You must be concise, structured, and action-oriented.\n"
            "If the prompt contains a section starting with 'RESUME_CONTEXT:' treat it as the user's resume text.\n"
            "Do not repeat the resume verbatim; extract only relevant facts.\n"
            "Output MUST be valid per the schema (reply_markdown, action_items, follow_up_questions, warnings).\n"
            "If inputs are missing (role, deadline, skills), ask focused follow-up questions.\n"
        )

        mode_instruction = {
            "plan": "Generate a timeboxed plan (today + next 7 days). Include action_items.",
            "resume": "Improve resume bullets based on user info/JD; provide 4-8 bullets and 3 fixes.",
            "interview": "Generate an interview prep set: 10 questions + what a strong answer includes.",
            "auto": "Decide whether plan/resume/interview is best, then proceed.",
        }[resolved_mode]

        prompt = (
            f"{system_context}\n"
            f"MODE: {resolved_mode}\n"
            f"MODE_INSTRUCTION: {mode_instruction}\n\n"
            "CONVERSATION_HISTORY:\n"
            f"{history}\n\n"
            "USER_LATEST:\n"
            f"{latest}\n"
        )

        async def primary():
            r = await self.main_agent_primary.run(prompt)
            return r.output

        async def fallback():
            r = await self.main_agent_fallback.run(prompt)
            return r.output

        try:
            return await self._run_with_retries(primary)
        except Exception:
            logger.exception("main primary failed; switching to fallback")
            resp = await self._run_with_retries(fallback)
            resp.warnings.append("Primary model failed; response generated with fallback model.")
            return resp


def build_orchestrator(settings: Settings) -> Orchestrator:
    primary_model = _build_openrouter_model(settings, settings.openrouter_model)
    fallback_model = _build_openrouter_model(settings, settings.openrouter_fallback_model)

    # PydanticAI uses output_type for structured outputs (not result_type). :contentReference[oaicite:5]{index=5}
    intent_agent_primary = Agent(
        model=primary_model,
        instructions="You only classify intent. Be strict and short.",
        output_type=Intent,
    )
    intent_agent_fallback = Agent(
        model=fallback_model,
        instructions="You only classify intent. Be strict and short.",
        output_type=Intent,
    )

    main_agent_primary = Agent(
        model=primary_model,
        instructions="You generate the final response as PlacementSprint.",
        output_type=AgentResponse,
    )
    main_agent_fallback = Agent(
        model=fallback_model,
        instructions="You generate the final response as PlacementSprint.",
        output_type=AgentResponse,
    )

    return Orchestrator(
        intent_agent_primary=intent_agent_primary,
        intent_agent_fallback=intent_agent_fallback,
        main_agent_primary=main_agent_primary,
        main_agent_fallback=main_agent_fallback,
    )
