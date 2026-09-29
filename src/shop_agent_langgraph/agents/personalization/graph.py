from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel

from ...core.llm import build_deepseek_model
from ...domain.market_cache import MarketCache
from ...domain.taxonomy import get_children
from ..relation.schemas import validate_graph
from ..relation.store import RelationStore, RevisionConflict
from .schemas import (ClaimReferenceError, OutputValidationError,
                      PersonalizationContent, PersonalizationSession,
                      PersonalizedProfile, QuestionSet, validate_output)
from .store import PersonalizationStore


class PersonalizationAgent:
    def __init__(self, model: BaseChatModel | None = None, *, market_cache=None,
                 relation_store=None, store=None):
        self.model = model
        self.market_cache = market_cache if market_cache is not None else MarketCache()
        self.relation_store = relation_store if relation_store is not None else RelationStore()
        self.store = store if store is not None else PersonalizationStore()

    def _load(self, node_id):
        profile = self.market_cache.load(self.market_cache.node(node_id))
        graph = self.relation_store.load(node_id)
        if profile is None or graph is None:
            raise ValueError("persisted Market Profile and RelationGraph are required")
        validate_graph(graph, profile)
        return profile, graph

    @staticmethod
    def _messages(messages):
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, str) or not m.strip() for m in messages
        ):
            raise ValueError("user_messages must be a non-empty list of user message strings")
        return list(messages)

    def _generate(self, schema, prompt, payload, profile, graph, messages):
        agent = create_agent(model=self.model or build_deepseek_model(), tools=[],
            system_prompt=Path(__file__).with_name(prompt).read_text(encoding="utf-8"),
            response_format=ToolStrategy(schema), name="personalization_agent")
        payload = {**payload, "allowed_claim_ids": sorted(
            c.id for c in graph.relations + graph.utility_rules if c.status == "supported")}
        for attempt in range(3):
            result = agent.invoke({"messages": [{"role": "user", "content": json.dumps(
                payload, ensure_ascii=False)}]}, config={"recursion_limit": 20})["structured_response"]
            result = schema.model_validate(result)
            try:
                validate_output(result, profile, graph, messages)
                if isinstance(result, PersonalizationContent) and result.candidate_scope:
                    allowed = {n["node_id"] for n in payload.get("candidate_children", [])}
                    if result.candidate_scope.node_id not in allowed:
                        raise OutputValidationError("candidate scope must be a supplied direct child")
            except OutputValidationError as error:
                if attempt == 2:
                    error_type = (ClaimReferenceError if isinstance(error, ClaimReferenceError)
                                  else OutputValidationError)
                    raise error_type(
                        f"output validation failed after 3 attempts: {error}") from error
                # Keep corrections separate from authentic user history. Never strip bad
                # IDs silently: the associated question/preference may also be unsound.
                instruction = (
                    "Regenerate the complete output and correct the validation error. "
                    "Use each question ID and context reference at most once. Use each "
                    "source_ref at most once across criteria and attributes; choose the "
                    "single representation that best matches the user's words. Preserve "
                    "all other supported user information and obey the output contract."
                )
                if isinstance(error, ClaimReferenceError):
                    instruction += (
                        " claim_ids must use exact IDs from allowed_claim_ids, without "
                        "prefixes. Review and revise the associated text and its basis, "
                        "not just the ID. Do not substitute an unrelated supported claim. "
                        "An empty list is valid for a direct user preference or neutral "
                        "context question; remove unsupported assertions. Put unresolved "
                        "knowledge in unresolved when the output schema permits."
                    )
                payload["validation_feedback"] = {
                    "error": str(error),
                    "rejected_output": result.model_dump(mode="json"),
                    "instruction": instruction,
                }
            else:
                return result
        raise AssertionError("unreachable")

    def _check_current(self, node_id, graph):
        _, current = self._load(node_id)
        if (current.profile_hash, current.revision) != (graph.profile_hash, graph.revision):
            raise RevisionConflict("upstream inputs changed during personalization; rerun")

    def prepare_questions(self, node_id: str, user_messages: list[str], *,
                          thread_id: str, task_id: str) -> QuestionSet:
        messages = self._messages(user_messages)
        profile, graph = self._load(node_id)
        previous = self.store.load(thread_id, task_id)
        if previous and previous.node_id != node_id:
            raise ValueError("use a new task_id when switching products")
        questions = self._generate(QuestionSet, "questions_prompt.md", {
            "market_profile": profile.model_dump(mode="json"),
            "relation_graph": graph.model_dump(mode="json"), "user_messages": messages,
        }, profile, graph, messages)
        self._check_current(node_id, graph)
        base = previous.revision if previous else 0
        self.store.save(PersonalizationSession(thread_id=thread_id, task_id=task_id,
            node_id=node_id, profile_hash=graph.profile_hash, graph_revision=graph.revision,
            revision=base + 1, user_messages=messages, question_set=questions), base)
        return questions

    def personalize(self, node_id: str, user_messages: list[str], *,
                    thread_id: str, task_id: str) -> PersonalizedProfile:
        messages = self._messages(user_messages)
        session = self.store.load(thread_id, task_id)
        if session is None or session.node_id != node_id:
            raise ValueError("prepare questions for this task and node first")
        if messages[:len(session.user_messages)] != session.user_messages:
            raise ValueError("provide the full chronological user history for this task")
        if session.result is None and session.question_set.questions and len(messages) == len(session.user_messages):
            raise ValueError("wait for user feedback before personalizing")
        profile, graph = self._load(node_id)
        if (session.profile_hash, session.graph_revision) != (graph.profile_hash, graph.revision):
            raise RevisionConflict("upstream inputs changed; prepare questions again")
        content = self._generate(PersonalizationContent, "profile_prompt.md", {
            "candidate_children": get_children([node_id])[0]["children"],
            "market_profile": profile.model_dump(mode="json"),
            "relation_graph": graph.model_dump(mode="json"),
            "question_set": session.question_set.model_dump(mode="json"),
            "user_messages": messages,
        }, profile, graph, messages)
        result = PersonalizedProfile(**content.model_dump(), node_id=node_id,
            profile_hash=graph.profile_hash, graph_revision=graph.revision)
        self._check_current(node_id, graph)
        updated = session.model_copy(update={"result": result, "user_messages": messages,
                                             "revision": session.revision + 1})
        self.store.save(updated, session.revision)
        return result

    async def aprepare_questions(self, *args, **kwargs) -> QuestionSet:
        return await asyncio.to_thread(self.prepare_questions, *args, **kwargs)

    async def apersonalize(self, *args, **kwargs) -> PersonalizedProfile:
        return await asyncio.to_thread(self.personalize, *args, **kwargs)


def build_personalization_agent(model: BaseChatModel | None = None) -> PersonalizationAgent:
    return PersonalizationAgent(model)


class LazyPersonalizationAgent:
    def prepare_questions(self, *args, **kwargs):
        return build_personalization_agent().prepare_questions(*args, **kwargs)

    def personalize(self, *args, **kwargs):
        return build_personalization_agent().personalize(*args, **kwargs)

    async def aprepare_questions(self, *args, **kwargs):
        return await build_personalization_agent().aprepare_questions(*args, **kwargs)

    async def apersonalize(self, *args, **kwargs):
        return await build_personalization_agent().apersonalize(*args, **kwargs)


personalization_agent = LazyPersonalizationAgent()
