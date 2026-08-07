from __future__ import annotations

from src.data_loader import (
    get_dataset_table_names,
)


GENERIC_DATA_RULES = """
DATA AND RELATIONSHIP RULES:
- Infer table grain only from the live schema,
  column names, samples, and the current question.
  Never invent a relationship.
- Treat every selected table as an independent
  evidence source unless the schema contains
  compatible keys and the question truly
  requires a join.
- Never join historical row-level tables on a
  descriptive name alone because repeated
  observations can create many-to-many
  duplication.
- Prefer independent SELECT queries when selected
  tables have different schemas or grains.
  The application will combine their result sets.
- Use a UNION only when its branches project
  genuinely compatible values.
- Apply user filters independently to every
  selected evidence source.
- Text matching must be case-insensitive and
  allow partial matching.
- Treat NULL, empty strings, Unknown, and
  Not Available as missing values.
- Do not interpret a missing value as numeric zero.
- Do not infer a total from a SQL LIMIT.
- Preserve ties for ranking and extreme-value
  questions.
- Use only columns present in the supplied
  live schema.
""".strip()


def get_relationship_context(
    selected_tables: (
        list[str]
        | tuple[str, ...]
    ),
) -> str:
    live_tables = set(
        get_dataset_table_names()
    )

    normalized_tables = tuple(
        dict.fromkeys(
            str(table_name)
            .strip()
            .lower()
            for table_name
            in selected_tables
            if str(table_name).strip()
        )
    )

    invalid = sorted(
        set(normalized_tables)
        - live_tables
    )

    if invalid:
        raise ValueError(
            "Unknown tables: "
            + ", ".join(
                invalid
            )
        )

    if not normalized_tables:
        raise ValueError(
            "At least one selected "
            "table is required."
        )

    rendered = "\n".join(
        f"- {name}"
        for name in normalized_tables
    )

    return (
        "SELECTED EVIDENCE SOURCES:\n"
        f"{rendered}\n\n"
        f"{GENERIC_DATA_RULES}"
    )