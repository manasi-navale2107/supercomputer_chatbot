import pytest
from pydantic import ValidationError

from src.models import RouteDecision


def test_structured_route_requires_sql():
    with pytest.raises(ValidationError):
        RouteDecision(
            route="structured",
            intent="count",
            reason="The question asks for a count.",
        )


def test_semantic_route_discards_sql():
    decision = RouteDecision(
        route="semantic",
        intent="description",
        reason="The question asks for a description.",
        sql="SELECT * FROM green500_systems",
        retrieval_query="Describe the Capella supercomputer",
    )
    assert decision.sql is None
    assert decision.retrieval_query == "Describe the Capella supercomputer"


def test_non_semantic_route_discards_retrieval_query():
    decision = RouteDecision(
        route="structured",
        intent="count",
        reason="The question asks for a count.",
        sql="SELECT COUNT(*) FROM top500_systems",
        retrieval_query="This must not be used",
    )
    assert decision.retrieval_query is None


def test_direct_route_requires_response():
    with pytest.raises(ValidationError):
        RouteDecision(
            route="direct",
            intent="clarification",
            reason="The metric is ambiguous.",
        )
