from __future__ import annotations

from typing import Any

from langgraph.graph import (
    END,
    StateGraph,
)

from src.answer_service import (
    generate_evidence_answer,
)
from src.config import (
    DEFAULT_LLM_PROVIDER,
)
from src.llm import (
    normalize_llm_provider,
)
from src.models import AgentState
from src.query_router import (
    route_question,
)
from src.semantic_route import (
    run_semantic_route,
)
from src.structured_route import (
    run_structured_route,
)
from src.table_selector import (
    select_relevant_tables,
)


def router_node(
    state: AgentState,
) -> dict[str, Any]:
    llm_provider = (
        normalize_llm_provider(
            state.get(
                "llm_provider",
                DEFAULT_LLM_PROVIDER,
            )
        )
    )

    decision, router_calls = (
        route_question(
            question=state["question"],
            chat_history=state.get(
                "chat_history",
                "",
            ),
            preferred_language=state.get(
                "preferred_language",
                "auto",
            ),
            llm_provider=llm_provider,
        )
    )

    output: dict[str, Any] = {
        "decision":
            decision.model_dump(),
        "llm_provider":
            llm_provider,
        "llm_calls": (
            int(
                state.get(
                    "llm_calls",
                    0,
                )
            )
            + router_calls
        ),
        "error":
            None,
    }

    if decision.route == "direct":
        output["answer"] = (
            decision.direct_response
            or (
                "Please clarify "
                "your question."
            )
        )

    return output


def route_after_router(
    state: AgentState,
) -> str:
    route = (
        state
        .get(
            "decision",
            {},
        )
        .get(
            "route",
            "direct",
        )
    )

    if route == "structured":
        return "table_selector"

    if route == "semantic":
        return "semantic"

    return "direct"


def table_selector_node(
    state: AgentState,
) -> dict[str, Any]:
    decision = dict(
        state.get(
            "decision",
            {},
        )
    )

    question = str(
        decision.get(
            "standalone_question"
        )
        or state["question"]
    ).strip()

    llm_provider = (
        normalize_llm_provider(
            state.get(
                "llm_provider",
                DEFAULT_LLM_PROVIDER,
            )
        )
    )

    selection, selector_calls = (
        select_relevant_tables(
            question=question,
            chat_history=state.get(
                "chat_history",
                "",
            ),
            llm_provider=llm_provider,
        )
    )

    decision["selected_tables"] = (
        selection.selected_tables
    )

    decision[
        "table_selection_reason"
    ] = selection.reason

    return {
        "decision":
            decision,
        "table_selection":
            selection.model_dump(),
        "llm_provider":
            llm_provider,
        "llm_calls": (
            int(
                state.get(
                    "llm_calls",
                    0,
                )
            )
            + selector_calls
        ),
        "error":
            None,
    }


def structured_node(
    state: AgentState,
) -> dict[str, Any]:
    return run_structured_route(
        dict(state)
    )


def semantic_node(
    state: AgentState,
) -> dict[str, Any]:
    return run_semantic_route(
        dict(state)
    )


def direct_node(
    state: AgentState,
) -> dict[str, Any]:
    decision = state.get(
        "decision",
        {},
    )

    return {
        "answer": (
            state.get("answer")
            or decision.get(
                "direct_response"
            )
            or (
                "Please clarify "
                "your question."
            )
        ),
        "llm_provider": (
            state.get(
                "llm_provider",
                DEFAULT_LLM_PROVIDER,
            )
        ),
        "llm_calls": int(
            state.get(
                "llm_calls",
                0,
            )
        ),
        "error":
            state.get("error"),
    }


def answer_node(
    state: AgentState,
) -> dict[str, Any]:
    return generate_evidence_answer(
        dict(state)
    )


def build_graph():
    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "router",
        router_node,
    )

    graph.add_node(
        "table_selector",
        table_selector_node,
    )

    graph.add_node(
        "structured",
        structured_node,
    )

    graph.add_node(
        "semantic",
        semantic_node,
    )

    graph.add_node(
        "direct",
        direct_node,
    )

    graph.add_node(
        "answer",
        answer_node,
    )

    graph.set_entry_point(
        "router"
    )

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "table_selector":
                "table_selector",
            "semantic":
                "semantic",
            "direct":
                "direct",
        },
    )

    graph.add_edge(
        "table_selector",
        "structured",
    )

    graph.add_edge(
        "structured",
        "answer",
    )

    graph.add_edge(
        "semantic",
        "answer",
    )

    graph.add_edge(
        "direct",
        END,
    )

    graph.add_edge(
        "answer",
        END,
    )

    return graph.compile()


_AGENT_GRAPH = build_graph()


def run_agent(
    question: str,
    chat_history: str = "",
    preferred_language: str = "auto",
    llm_provider: str = (
        DEFAULT_LLM_PROVIDER
    ),
) -> dict[str, Any]:
    cleaned_question = (
        question.strip()
    )

    if not cleaned_question:
        raise ValueError(
            "Question cannot be empty."
        )

    normalized_provider = (
        normalize_llm_provider(
            llm_provider
        )
    )

    initial_state: AgentState = {
        "question":
            cleaned_question,
        "chat_history":
            chat_history,
        "preferred_language":
            preferred_language,
        "llm_provider":
            normalized_provider,
        "decision": {},
        "table_selection": {},
        "schema": "",
        "relationships": "",
        "generated_sql": "",
        "evidence": {},
        "answer": "",
        "llm_calls": 0,
        "error": None,
    }

    return _AGENT_GRAPH.invoke(
        initial_state
    )