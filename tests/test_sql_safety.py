import pytest

from src.sql_safety import UnsafeSQLError, validate_and_limit_sql


def test_allows_read_only_query_and_preserves_small_limit():
    sql = validate_and_limit_sql(
        "SELECT system_name FROM green500_systems ORDER BY green500_rank LIMIT 5"
    )
    assert "LIMIT 5" in sql.upper()


def test_adds_hard_limit_when_missing():
    sql = validate_and_limit_sql("SELECT country FROM country_stats")
    assert "LIMIT 201" in sql.upper()


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE green500_systems",
        "DELETE FROM green500_systems",
        "UPDATE green500_systems SET country = 'x'",
        "SELECT * FROM unknown_table",
        "SELECT SLEEP(5) FROM green500_systems",
        "SELECT * FROM information_schema.green500_systems",
    ],
)
def test_blocks_unsafe_queries(sql):
    with pytest.raises(UnsafeSQLError):
        validate_and_limit_sql(sql)

