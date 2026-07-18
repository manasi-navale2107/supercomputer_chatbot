from __future__ import annotations

from typing import Any

from langgraph.graph import (
    END,
    StateGraph,
)

from src.answer_service import (
    generate_evidence_answer,
)
from src.models import AgentState
from src.query_router import route_question
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
    decision, router_calls = (
        route_question(
            question=state["question"],
            chat_history=state.get(
                "chat_history",
                "",
            ),
        )
    )

    output: dict[str, Any] = {
        "decision": decision.model_dump(),
        "llm_calls": (
            int(
                state.get(
                    "llm_calls",
                    0,
                )
            )
            + router_calls
        ),
        "error": None,
    }

    if decision.route == "direct":
        output["answer"] = (
            decision.direct_response
            or "Please clarify your question."
        )

    return output


def route_after_router(
    state: AgentState,
) -> str:
    route = (
        state
        .get("decision", {})
        .get("route", "direct")
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
        decision.get("standalone_question")
        or state["question"]
    ).strip()

    selection, selector_calls = (
        select_relevant_tables(
            question=question,
            chat_history=state.get(
                "chat_history",
                "",
            ),
        )
    )

    decision["selected_tables"] = (
        selection.selected_tables
    )

    decision["table_selection_reason"] = (
        selection.reason
    )

    return {
        "decision": decision,
        "table_selection": (
            selection.model_dump()
        ),
        "llm_calls": (
            int(
                state.get(
                    "llm_calls",
                    0,
                )
            )
            + selector_calls
        ),
        "error": None,
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
    return {
        "answer": (
            state.get("answer")
            or state.get(
                "decision",
                {},
            ).get("direct_response")
            or "Please clarify your question."
        ),
        "llm_calls": int(
            state.get(
                "llm_calls",
                0,
            )
        ),
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
            "table_selector": "table_selector",
            "semantic": "semantic",
            "direct": "direct",
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
) -> dict[str, Any]:
    initial_state: AgentState = {
        "question": question.strip(),
        "chat_history": chat_history,
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