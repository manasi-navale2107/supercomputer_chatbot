from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from src.config import TABLE_NAMES


RouteName = Literal[
    "structured",
    "semantic",
    "direct",
]


class RouteDecision(BaseModel):
    route: RouteName
    intent: str = Field(
        min_length=1,
        max_length=80,
    )
    reason: str = Field(
        min_length=1,
        max_length=500,
    )

    standalone_question: str | None = Field(
        default=None,
        max_length=2_000,
    )
    retrieval_query: str | None = Field(
        default=None,
        max_length=2_000,
    )
    direct_response: str | None = Field(
        default=None,
        max_length=2_000,
    )

    @model_validator(mode="after")
    def validate_route_payload(
        self,
    ) -> "RouteDecision":
        if self.route == "structured":
            if not (
                self.standalone_question or ""
            ).strip():
                raise ValueError(
                    "Structured route requires "
                    "a standalone question."
                )

            self.retrieval_query = None
            self.direct_response = None

        elif self.route == "semantic":
            if not (
                self.standalone_question or ""
            ).strip():
                raise ValueError(
                    "Semantic route requires "
                    "a standalone question."
                )

            if not (
                self.retrieval_query or ""
            ).strip():
                raise ValueError(
                    "Semantic route requires "
                    "a retrieval query."
                )

            self.direct_response = None

        else:
            if not (
                self.direct_response or ""
            ).strip():
                raise ValueError(
                    "Direct route requires "
                    "a direct response."
                )

            self.standalone_question = None
            self.retrieval_query = None

        return self


class TableSelection(BaseModel):
    """
    Dataset selection produced by the dedicated table-selector LLM.
    """

    selected_tables: list[str] = Field(
        min_length=1
    )
    excluded_tables: list[str] = Field(
        default_factory=list
    )
    reason: str = Field(
        min_length=1,
        max_length=1_000,
    )

    @field_validator(
        "selected_tables",
        "excluded_tables",
    )
    @classmethod
    def validate_table_names(
        cls,
        values: list[str],
    ) -> list[str]:
        allowed = set(TABLE_NAMES)
        normalized: list[str] = []

        for value in values:
            table_name = (
                value.strip().lower()
            )

            if table_name not in allowed:
                raise ValueError(
                    "Unknown dataset returned: "
                    f"{value}"
                )

            if table_name not in normalized:
                normalized.append(
                    table_name
                )

        return normalized

    @model_validator(mode="after")
    def validate_complete_partition(
        self,
    ) -> "TableSelection":
        selected = set(
            self.selected_tables
        )
        excluded = set(
            self.excluded_tables
        )

        overlap = selected & excluded

        if overlap:
            raise ValueError(
                "Datasets cannot be both selected "
                "and excluded: "
                + ", ".join(sorted(overlap))
            )

        missing = (
            set(TABLE_NAMES)
            - selected
            - excluded
        )

        if missing:
            raise ValueError(
                "Every live dataset must be evaluated. "
                "Missing: "
                + ", ".join(sorted(missing))
            )

        return self


class AgentState(TypedDict, total=False):
    question: str
    chat_history: str

    decision: dict[str, Any]
    table_selection: dict[str, Any]

    schema: str
    relationships: str
    generated_sql: str

    evidence: dict[str, Any]
    answer: str

    llm_calls: int
    error: str | None