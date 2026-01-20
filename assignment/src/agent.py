import asyncio
import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

try:
    from pydantic_ai.exceptions import ModelHTTPError
except Exception:  # very defensive; prevents import-time break if versions vary
    ModelHTTPError = Exception  # type: ignore


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
    """
    PydanticAI orchestrator:
    - Uses Agent(...) (PydanticAI) for generation
    - Enforces structured output via "strict JSON" + Pydantic validation
    - Retries & fallback model
    """

    def __init__(self, settings):
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is missing in environment variables.")

        # IMPORTANT: No `default_headers=` here (your deployed version doesn't support it)
        provider = OpenAIProvider(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

        self.primary_model = OpenAIChatModel(settings.openrouter_model, provider=provider)
        self.fallback_model = OpenAIChatModel(settings.openrouter_fallback_model, provider=provider)

        # Use plain text output to avoid tool/function-call requirements on OpenRouter routes.
        self.intent_agent_primary = Agent(model=self.primary_model, output_type=str)
        self.intent_agent_fallback = Agent(model=self.fallback_model, output_type=str)

        self.main_agent_primary = Agent(model=self.primary_model, output_type=str)
        self.main_agent_fallback = Agent(model=self.fallback_model, output_type=str)

    async def _run_with_retries(self, fn, *, attempts: int = 3):
        last_err: Exception | None = None
        for i in range(1, attempts + 1):
            try:
                return await fn()
            except Exception as e:
                last_err = e
                wait = 1.5 * i
                logger.warning("attempt=%s failed: %r; retrying in %.1fs", i, e, wait)
                await asyncio.sleep(wait)
        assert last_err is not None
        raise last_err

    async def _call_text(self, agent: Agent, prompt: str) -> str:
        r = await agent.run(prompt)
        out = r.data if isinstance(r.data, str) else str(r.data)
        return out.strip()

    async def classify_intent(self, user_text: str) -> Literal["plan", "resume", "interview"]:
        prompt = (
            "Return ONE WORD ONLY: plan OR resume OR interview.\n"
            "Rules:\n"
            "- If user asks for day-wise schedule, strategy, or preparation timeline => plan\n"
            "- If user asks to improve resume, ATS, bullets, tailoring => resume\n"
            "- If user asks interview Q&A, mock interview, questions => interview\n\n"
            f"USER:\n{user_text}\n"
        )

        async def primary():
            t = await self._call_text(self.intent_agent_primary, prompt)
            return t.lower()

        async def fallback():
            t = await self._call_text(self.intent_agent_fallback, prompt)
            return t.lower()

        try:
            raw = await self._run_with_retries(primary)
        except Exception:
            raw = await self._run_with_retries(fallback)

        if "resume" in raw:
            return "resume"
        if "interview" in raw:
            return "interview"
        return "plan"

    def _json_schema_instructions(self) -> str:
        # Provide an example to increase strict JSON compliance
        return (
            "Return STRICT JSON ONLY. No markdown, no backticks, no commentary.\n"
            "JSON schema:\n"
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

        if mode in ("plan", "resume", "interview"):
            intent = mode  # type: ignore
        else:
            intent = await self.classify_intent(latest)

        system = (
            "You are PlacementSprint, a practical placement-prep agent.\n"
            "Be concise, structured, and action-oriented.\n"
            "If resume text is provided, extract only relevant facts; never paste it fully.\n"
        )

        prompt = (
            f"{system}\n"
            f"MODE: {intent}\n\n"
            f"{self._json_schema_instructions()}\n"
            "Conversation:\n"
            f"{_history_to_text(messages)}\n"
        )

        async def primary():
            return await self._call_text(self.main_agent_primary, prompt)

        async def fallback():
            return await self._call_text(self.main_agent_fallback, prompt)

        try:
            raw = await self._run_with_retries(primary)
        except Exception:
            raw = await self._run_with_retries(fallback)

        # Parse + validate strict JSON
        try:
            data = json.loads(raw)
            return AgentResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            # One repair attempt (still using PydanticAI)
            repair_prompt = (
                "Your previous output was invalid.\n"
                "Fix it and return STRICT JSON ONLY matching the schema.\n\n"
                f"SCHEMA:\n{self._json_schema_instructions()}\n"
                f"INVALID_OUTPUT:\n{raw}\n"
            )
            repaired = await self._call_text(self.main_agent_fallback, repair_prompt)
            data = json.loads(repaired)
            return AgentResponse.model_validate(data)

    async def respond(self, messages: list, mode: str) -> dict:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        res = await self.generate(msg_dicts, mode)
        return res.model_dump()
