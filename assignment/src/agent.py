import asyncio
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


class IntentResult(BaseModel):
    intent: Literal["plan", "resume", "interview", "auto"]


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


def _format_history(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        content = m["content"]
        lines.append(f"{role}:\n{content}\n")
    return "\n".join(lines).strip()


class Orchestrator:
    def __init__(self, settings):
        provider = OpenAIProvider(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            default_headers={
                # Optional attribution headers
                **({"HTTP-Referer": settings.site_url} if settings.site_url else {}),
                **({"X-Title": settings.app_name} if settings.app_name else {}),
            },
        )

        self.primary_model = OpenAIChatModel(settings.openrouter_model, provider=provider)
        self.fallback_model = OpenAIChatModel(settings.openrouter_fallback_model, provider=provider)

        self.intent_agent_primary = Agent(model=self.primary_model, output_type=IntentResult)
        self.intent_agent_fallback = Agent(model=self.fallback_model, output_type=IntentResult)

        self.main_agent_primary = Agent(model=self.primary_model, output_type=AgentResponse)
        self.main_agent_fallback = Agent(model=self.fallback_model, output_type=AgentResponse)

    async def _run_with_retries(self, fn, *, max_attempts: int = 3):
        last = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await fn()
            except ModelHTTPError as e:
                last = e
                await asyncio.sleep(2.0 * attempt)
        raise last

    async def classify_intent(self, user_text: str) -> str:
        prompt = (
            "Classify the user's intent into one of: plan, resume, interview.\n"
            "Return ONLY the structured output.\n\n"
            f"USER:\n{user_text}"
        )

        async def primary():
            r = await self.intent_agent_primary.run(prompt)
            return r.data.intent

        async def fallback():
            r = await self.intent_agent_fallback.run(prompt)
            return r.data.intent

        try:
            return await self._run_with_retries(primary)
        except ModelHTTPError:
            return await self._run_with_retries(fallback)

    async def respond(self, messages, mode: str):
        # messages are pydantic models (ChatMessage), but we can treat as dict-like
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        latest = msg_dicts[-1]["content"]

        intent = mode if mode != "auto" else await self.classify_intent(latest)

        system_context = (
            "You are PlacementSprint, a practical placement-prep agent.\n"
            "Be concise, structured, and action-oriented.\n"
            "If the prompt contains a section starting with 'RESUME_CONTEXT:' treat it as resume text.\n"
            "Do not repeat the resume verbatim—extract only relevant facts.\n"
            "Return output strictly matching the schema.\n"
        )

        task_context = {
            "plan": "Focus on building a timeboxed plan with daily milestones.",
            "resume": "Focus on improving resume bullets, ATS keywords, and project framing.",
            "interview": "Focus on interview questions, strong-answer scaffolds, and practice drills.",
            "auto": "Choose the best of plan/resume/interview based on the user message.",
        }[intent]

        prompt = (
            f"{system_context}\n"
            f"MODE: {intent}\n"
            f"GOAL: {task_context}\n\n"
            "CONVERSATION:\n"
            f"{_format_history(msg_dicts)}"
        )

        async def primary():
            r = await self.main_agent_primary.run(prompt)
            return r.data.model_dump()

        async def fallback():
            r = await self.main_agent_fallback.run(prompt)
            return r.data.model_dump()

        try:
            return await self._run_with_retries(primary)
        except ModelHTTPError:
            return await self._run_with_retries(fallback)
