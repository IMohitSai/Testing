import asyncio
import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

logger = logging.getLogger("placementsprint.agent")


class ActionItem(BaseModel):
    title: str = Field(min_length=3, max_length=80)
    why: str = Field(min_length=5, max_length=280)
    eta_minutes: int = Field(ge=1, le=600)
    priority: Literal["low", "medium", "high"]


class AgentResponse(BaseModel):
    reply_markdown: str = Field(min_length=1, max_length=8000)
    action_items: list[ActionItem] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _history_to_text(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = (m.get("role") or "").upper()
        content = m.get("content") or ""
        lines.append(f"{role}:\n{content}\n")
    return "\n".join(lines).strip()


class Orchestrator:
    def __init__(self, settings):
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is missing (set it in Vercel env vars).")

        # ✅ FIX: no default_headers (your deployed pydantic-ai doesn't accept it)
        provider = OpenAIProvider(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

        self.primary_model = OpenAIChatModel(settings.openrouter_model, provider=provider)
        self.fallback_model = OpenAIChatModel(settings.openrouter_fallback_model, provider=provider)

        # IMPORTANT: output_type=str to avoid tool/function calling on OpenRouter routes
        self.intent_agent_primary = Agent(model=self.primary_model, output_type=str)
        self.intent_agent_fallback = Agent(model=self.fallback_model, output_type=str)
        self.main_agent_primary = Agent(model=self.primary_model, output_type=str)
        self.main_agent_fallback = Agent(model=self.fallback_model, output_type=str)

    async def _run_with_retries(self, fn, attempts: int = 3):
        last = None
        for i in range(1, attempts + 1):
            try:
                return await fn()
            except Exception as e:
                last = e
                wait = 1.5 * i
                logger.warning("attempt=%s failed: %r; retrying in %.1fs", i, e, wait)
                await asyncio.sleep(wait)
        raise last  # type: ignore

    async def _call_text(self, agent: Agent, prompt: str) -> str:
        r = await agent.run(prompt)
        return (r.data if isinstance(r.data, str) else str(r.data)).strip()

    async def classify_intent(self, user_text: str) -> Literal["plan", "resume", "interview"]:
        prompt = (
            "Return ONE WORD ONLY: plan OR resume OR interview.\n"
            "Rules:\n"
            "- schedule/timeline/strategy => plan\n"
            "- resume/ATS/bullets/tailor => resume\n"
            "- interview Q&A/mock => interview\n\n"
            f"USER:\n{user_text}\n"
        )

        async def p():
            return (await self._call_text(self.intent_agent_primary, prompt)).lower()

        async def f():
            return (await self._call_text(self.intent_agent_fallback, prompt)).lower()

        try:
            raw = await self._run_with_retries(p)
        except Exception:
            raw = await self._run_with_retries(f)

        if "resume" in raw:
            return "resume"
        if "interview" in raw:
            return "interview"
        return "plan"

    def _schema_instructions(self) -> str:
        return (
            "Return STRICT JSON ONLY. No markdown, no backticks.\n"
            "Schema:\n"
            "{\n"
            '  "reply_markdown": string,\n'
            '  "action_items": [{"title": string, "why": string, "eta_minutes": int, "priority": "low|medium|high"}],\n'
            '  "follow_up_questions": [string],\n'
            '  "warnings": [string]\n'
            "}\n"
        )

    async def generate(self, messages: list[dict], mode: str) -> AgentResponse:
        latest = messages[-1]["content"]

        intent: Literal["plan", "resume", "interview"]
        intent = mode if mode in ("plan", "resume", "interview") else await self.classify_intent(latest)  # type: ignore

        prompt = (
            "You are PlacementSprint, a practical placement-prep agent. Be concise and actionable.\n"
            f"MODE: {intent}\n\n"
            f"{self._schema_instructions()}\n"
            "Conversation:\n"
            f"{_history_to_text(messages)}\n"
        )

        async def p():
            return await self._call_text(self.main_agent_primary, prompt)

        async def f():
            return await self._call_text(self.main_agent_fallback, prompt)

        try:
            raw = await self._run_with_retries(p)
        except Exception:
            raw = await self._run_with_retries(f)

        # Parse strict JSON and validate with Pydantic
        try:
            return AgentResponse.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError):
            repair = (
                "Your previous output was invalid.\n"
                "Fix it and return STRICT JSON ONLY matching the schema.\n\n"
                f"{self._schema_instructions()}\n"
                f"INVALID:\n{raw}\n"
            )
            repaired = await self._call_text(self.main_agent_fallback, repair)
            return AgentResponse.model_validate(json.loads(repaired))

    async def respond(self, messages: list, mode: str) -> dict:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        res = await self.generate(msg_dicts, mode)
        return res.model_dump()
