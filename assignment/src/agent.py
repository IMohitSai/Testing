from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

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


def _build_openrouter_model(settings: Settings, model_id: str) -> OpenAIChatModel:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

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


def _extract_json(text: str) -> dict:
    """Extract a JSON object from model text (handles fenced blocks and extra chatter)."""
    t = (text or "").strip()

    # Prefer ```json ... ```
    if "```" in t:
        parts = t.split("```")
        for i in range(len(parts) - 1):
            if parts[i].strip().lower().endswith("json"):
                candidate = parts[i + 1].strip()
                try:
                    return json.loads(candidate)
                except Exception:
                    pass

    # Fallback: first {...last}
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        return json.loads(candidate)

    raise ValueError("No JSON object found in model output")


def _validate_agent_response(obj: dict) -> AgentResponse:
    return AgentResponse.model_validate(obj)


def _format_history(messages: list[ChatMessage], keep_last: int = 12) -> str:
    trimmed = messages[-keep_last:]
    lines: list[str] = []
    for m in trimmed:
        role = "USER" if m.role == "user" else "ASSISTANT"
        lines.append(f"{role}: {m.content.strip()}")
    return "\n".join(lines)


@dataclass
class Orchestrator:
    main_agent_primary: Agent[None, str]
    main_agent_fallback: Agent[None, str]

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

    async def _call_model(self, *, prompt: str, use_fallback: bool) -> AgentResponse:
        agent = self.main_agent_fallback if use_fallback else self.main_agent_primary
        raw = (await agent.run(prompt)).output

        try:
            obj = _extract_json(raw)
            return _validate_agent_response(obj)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            # One repair attempt (same agent)
            repair_prompt = (
                "You returned invalid JSON.\n"
                "Return ONLY a single JSON object matching this schema:\n"
                "{"
                '"reply_markdown": string,'
                '"action_items": [{"title": string, "why": string, "eta_minutes": int, "priority": "low|med|high"}],'
                '"follow_up_questions": [string],'
                '"warnings": [string]'
                "}\n"
                "No markdown fences. No extra keys. No extra text.\n\n"
                f"INVALID_OUTPUT:\n{raw}"
            )
            raw2 = (await agent.run(repair_prompt)).output
            obj2 = _extract_json(raw2)
            return _validate_agent_response(obj2)

    async def respond(self, messages: list[ChatMessage], mode: Mode) -> AgentResponse:
        if not messages or messages[-1].role != "user":
            raise ValueError("Last message must be from the user.")

        resolved_mode = mode

        system_context = (
            "You are PlacementSprint, a practical placement-prep agent.\n"
            "Be concise, structured, and action-oriented.\n"
            "If the prompt contains a section starting with 'RESUME_CONTEXT:' treat it as the user's resume text.\n"
            "Do not repeat the resume verbatim; extract only relevant facts.\n"
            "You MUST output ONLY JSON (one object) with keys: reply_markdown, action_items, follow_up_questions, warnings.\n"
            "Use double quotes and valid JSON. No trailing commas. No markdown fences.\n"
        )

        mode_instruction = {
            "plan": "Make a timeboxed plan (today + next 7 days). Include action_items.",
            "resume": "Improve resume bullets; provide 4-8 rewritten bullets and 3 concrete fixes.",
            "interview": "Generate interview prep: 10 questions + what a strong answer includes.",
            "auto": "Decide whether plan/resume/interview is best, then proceed with that.",
        }[resolved_mode]

        history = _format_history(messages)
        latest = messages[-1].content.strip()

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
            return await self._call_model(prompt=prompt, use_fallback=False)

        async def fallback():
            return await self._call_model(prompt=prompt, use_fallback=True)

        try:
            return await self._run_with_retries(primary)
        except Exception:
            logger.exception("primary model failed; switching to fallback")
            resp = await self._run_with_retries(fallback)
            resp.warnings.append("Primary model failed; response generated with fallback model.")
            return resp


def build_orchestrator(settings: Settings) -> Orchestrator:
    primary_model = _build_openrouter_model(settings, settings.openrouter_model)
    fallback_model = _build_openrouter_model(settings, settings.openrouter_fallback_model)

    main_agent_primary = Agent(
        model=primary_model,
        instructions="You are PlacementSprint.",
        output_type=str,  # IMPORTANT: avoids tool/function calling
    )
    main_agent_fallback = Agent(
        model=fallback_model,
        instructions="You are PlacementSprint.",
        output_type=str,  # IMPORTANT: avoids tool/function calling
    )

    return Orchestrator(
        main_agent_primary=main_agent_primary,
        main_agent_fallback=main_agent_fallback,
    )
