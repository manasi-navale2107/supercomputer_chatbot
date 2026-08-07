from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4


STOP_WORDS = {
    "a", "about", "all", "also", "an", "and", "are", "as", "at", "be",
    "because", "been", "both", "but", "by", "can", "current", "do", "does",
    "each", "for", "from", "has", "have", "how", "in", "into", "is", "it",
    "its", "latest", "list", "many", "more", "most", "of", "on", "only",
    "or", "other", "same", "system", "systems", "than", "that", "the",
    "their", "there", "these", "they", "this", "to", "use", "used", "uses",
    "using", "was", "were", "what", "when", "where", "which", "while",
    "who", "why", "with",
}


def parse_qa_file(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"QA file not found: {path}"
        )

    records: list[dict[str, Any]] = []
    section: str | None = None
    question_lines: list[str] = []
    answer_lines: list[str] = []
    reading_answer = False

    def save_current() -> None:
        nonlocal question_lines
        nonlocal answer_lines
        nonlocal reading_answer

        question = " ".join(
            line.strip()
            for line in question_lines
            if line.strip()
        )

        expected_answer = "\n".join(
            answer_lines
        ).strip()

        if (
            question
            and expected_answer
            and section
        ):
            expected_route = (
                "structured"
                if section == "structured"
                else "semantic"
            )

            records.append(
                {
                    "question_id":
                        len(records) + 1,
                    "section":
                        section,
                    "expected_route":
                        expected_route,
                    "question":
                        question,
                    "expected_answer":
                        expected_answer,
                }
            )

        question_lines = []
        answer_lines = []
        reading_answer = False

    content = path.read_text(
        encoding="utf-8-sig"
    )

    for raw_line in content.splitlines():
        line = raw_line.strip()
        upper_line = line.upper()

        if upper_line.startswith(
            "SECTION A"
        ):
            save_current()
            section = "structured"
            continue

        if upper_line.startswith(
            "SECTION B"
        ):
            save_current()
            section = "semantic"
            continue

        if line.startswith("Q:"):
            save_current()

            question_lines = [
                line[2:].strip()
            ]

            continue

        if line.startswith("A:"):
            if question_lines:
                answer_lines = [
                    line[2:].strip()
                ]

                reading_answer = True

            continue

        if (
            question_lines
            and reading_answer
        ):
            answer_lines.append(
                raw_line.rstrip()
            )

        elif (
            question_lines
            and line
            and not line.startswith("---")
        ):
            question_lines.append(
                line
            )

    save_current()

    if not records:
        raise ValueError(
            "No Q:/A: records found "
            "in the QA file."
        )

    return records


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    data = None

    headers = {
        "Accept": "application/json",
    }

    if payload is not None:
        data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        headers[
            "Content-Type"
        ] = "application/json"

    request = Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            text = (
                response
                .read()
                .decode("utf-8")
            )

            if not text.strip():
                return {}

            return json.loads(
                text
            )

    except HTTPError as error:
        detail = (
            error
            .read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        raise RuntimeError(
            f"HTTP {error.code}: "
            f"{detail[:800]}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Connection failed: "
            f"{error.reason}"
        ) from error


def extract_answer(
    response: dict[str, Any],
) -> str:
    assistant_message = (
        response.get(
            "assistant_message",
            {},
        )
    )

    if isinstance(
        assistant_message,
        dict,
    ):
        for key in (
            "content",
            "answer",
            "message",
            "text",
        ):
            value = (
                assistant_message.get(
                    key
                )
            )

            if value:
                return str(
                    value
                ).strip()

    if isinstance(
        assistant_message,
        str,
    ):
        return (
            assistant_message.strip()
        )

    return str(
        response.get("answer")
        or ""
    ).strip()


def delete_test_conversation(
    api_url: str,
    conversation_id: Any,
    client_id: str,
    timeout: int,
) -> None:
    if not conversation_id:
        return

    query = urlencode(
        {
            "client_id": client_id,
        }
    )

    url = (
        f"{api_url}/conversations/"
        f"{conversation_id}?{query}"
    )

    try:
        request_json(
            method="DELETE",
            url=url,
            timeout=timeout,
        )

    except Exception as error:
        print(
            "  Conversation cleanup "
            f"warning: {error}"
        )


def normalise_text(
    text: str,
) -> str:
    text = text.casefold()

    replacements = {
        "united states of america":
            "usa",
        "united states":
            "usa",
        "u.s.":
            "usa",
        "pflop/s":
            "pflops",
        "tflop/s":
            "tflops",
        "gflops/watt":
            "gflopsw",
        "gflops/w":
            "gflopsw",
    }

    for old, new in (
        replacements.items()
    ):
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"[^a-z0-9.+%_-]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def word_tokens(
    text: str,
) -> set[str]:
    words = re.findall(
        r"[a-z][a-z0-9_-]{2,}",
        normalise_text(text),
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
    }


def numeric_values(
    text: str,
) -> list[float]:
    values: list[float] = []

    cleaned_text = text.replace(
        ",",
        "",
    )

    raw_values = re.findall(
        (
            r"(?<![a-zA-Z])"
            r"[-+]?\d+(?:\.\d+)?"
        ),
        cleaned_text,
    )

    for raw_value in raw_values:
        try:
            values.append(
                float(raw_value)
            )

        except ValueError:
            pass

    return values


def numbers_equal(
    expected: float,
    actual: float,
) -> bool:
    if math.isclose(
        expected,
        actual,
        rel_tol=0.02,
        abs_tol=0.05,
    ):
        return True

    # Allows values such as
    # 29.7 MW and 29,700 kW.
    for scale in (
        1000.0,
        0.001,
    ):
        if math.isclose(
            expected,
            actual * scale,
            rel_tol=0.02,
            abs_tol=0.05,
        ):
            return True

    return False


def calculate_answer_score(
    section: str,
    expected_answer: str,
    generated_answer: str,
) -> tuple[float, str]:
    if not generated_answer.strip():
        return (
            0.0,
            "Empty chatbot answer",
        )

    expected_words = word_tokens(
        expected_answer
    )

    generated_words = word_tokens(
        generated_answer
    )

    overlapping_words = (
        expected_words
        & generated_words
    )

    if generated_words:
        precision = (
            len(overlapping_words)
            / len(generated_words)
        )
    else:
        precision = 0.0

    if expected_words:
        recall = (
            len(overlapping_words)
            / min(
                len(expected_words),
                15,
            )
        )
    else:
        recall = 0.0

    keyword_score = min(
        1.0,
        (
            0.55 * precision
            + 0.45 * recall
        ),
    )

    expected_numbers = (
        numeric_values(
            expected_answer
        )
    )

    generated_numbers = (
        numeric_values(
            generated_answer
        )
    )

    used_generated_numbers: set[int] = (
        set()
    )

    matching_numbers = 0

    for expected_number in (
        expected_numbers
    ):
        for index, generated_number in (
            enumerate(
                generated_numbers
            )
        ):
            if (
                index
                not in used_generated_numbers
                and numbers_equal(
                    expected_number,
                    generated_number,
                )
            ):
                matching_numbers += 1

                used_generated_numbers.add(
                    index
                )

                break

    if expected_numbers:
        numeric_score = (
            matching_numbers
            / len(expected_numbers)
        )
    else:
        numeric_score = None

    if numeric_score is None:
        final_score = keyword_score

    elif section == "structured":
        final_score = (
            0.45 * keyword_score
            + 0.55 * numeric_score
        )

    else:
        final_score = (
            0.80 * keyword_score
            + 0.20 * numeric_score
        )

    reason = (
        f"keyword={keyword_score:.2f}"
    )

    if numeric_score is not None:
        reason += (
            f", numeric="
            f"{numeric_score:.2f}"
        )

    return (
        round(
            final_score * 100,
            2,
        ),
        reason,
    )


def count_citations(
    citations: Any,
) -> int:
    if not isinstance(
        citations,
        dict,
    ):
        return 0

    total = 0

    for key in (
        "datasets",
        "documents",
        "sources",
        "references",
    ):
        items = citations.get(
            key
        )

        if isinstance(
            items,
            list,
        ):
            total += len(
                [
                    item
                    for item in items
                    if item
                ]
            )

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "HPC chatbot QA evaluator"
        )
    )

    parser.add_argument(
        "--qa-file",
        default=(
            "HPC_chatbot_test_QA.txt"
        ),
    )

    parser.add_argument(
        "--api-url",
        default=(
            "http://127.0.0.1:8000"
        ),
    )

    parser.add_argument(
        "--provider",
        choices=(
            "ollama",
            "groq",
        ),
        default="ollama",
    )

    parser.add_argument(
        "--section",
        choices=(
            "all",
            "structured",
            "semantic",
        ),
        default="all",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--parse-only",
        action="store_true",
    )

    parser.add_argument(
        "--keep-conversations",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "runtime/qa_reports"
        ),
    )

    args = parser.parse_args()

    qa_path = (
        Path(args.qa_file)
        .expanduser()
        .resolve()
    )

    records = parse_qa_file(
        qa_path
    )

    section_counts = Counter(
        record["section"]
        for record in records
    )

    if args.parse_only:
        print(
            "Total questions:",
            len(records),
        )

        print(
            "Structured questions:",
            section_counts[
                "structured"
            ],
        )

        print(
            "Semantic questions:",
            section_counts[
                "semantic"
            ],
        )

        print("QA_PARSE_PASSED")
        return

    if args.section != "all":
        records = [
            record
            for record in records
            if record["section"]
            == args.section
        ]

    if args.limit > 0:
        records = records[
            :args.limit
        ]

    if not records:
        raise ValueError(
            "No questions selected."
        )

    api_url = (
        args.api_url.rstrip("/")
    )

    client_id = str(
        uuid4()
    )

    results: list[
        dict[str, Any]
    ] = []

    for position, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"[{position}/{len(records)}] "
            f"Q{record['question_id']}: "
            f"{record['question'][:90]}"
        )

        started_at = (
            time.perf_counter()
        )

        response: dict[str, Any] = {}
        error_message = ""

        try:
            response = request_json(
                method="POST",
                url=(
                    f"{api_url}/messages"
                ),
                payload={
                    "client_id":
                        client_id,
                    "conversation_id":
                        None,
                    "content":
                        record["question"],
                    "preferred_language":
                        "en-IN",
                    "llm_provider":
                        args.provider,
                },
                timeout=args.timeout,
            )

        except Exception as error:
            error_message = str(
                error
            )

        response_time = round(
            (
                time.perf_counter()
                - started_at
            ),
            3,
        )

        generated_answer = (
            extract_answer(
                response
            )
        )

        actual_route = str(
            response.get("route")
            or "error"
        )

        api_status = str(
            response.get("status")
            or "error"
        )

        answer_score, score_reason = (
            calculate_answer_score(
                section=record[
                    "section"
                ],
                expected_answer=record[
                    "expected_answer"
                ],
                generated_answer=(
                    generated_answer
                ),
            )
        )

        route_correct = (
            actual_route
            == record["expected_route"]
        )

        citation_total = (
            count_citations(
                response.get(
                    "citations"
                )
            )
        )

        sql = str(
            response.get("sql")
            or ""
        )

        sql_success = (
            record["expected_route"]
            != "structured"
            or (
                actual_route
                == "structured"
                and api_status
                == "success"
                and bool(sql)
            )
        )

        if (
            error_message
            or api_status == "error"
        ):
            final_status = "ERROR"

        elif (
            answer_score >= 70
            and route_correct
        ):
            final_status = "PASS"

        elif answer_score >= 40:
            final_status = "PARTIAL"

        else:
            final_status = "FAIL"

        results.append(
            {
                **record,
                "generated_answer":
                    generated_answer,
                "actual_route":
                    actual_route,
                "route_correct":
                    route_correct,
                "estimated_answer_score_percent":
                    answer_score,
                "score_reason":
                    score_reason,
                "citations_present":
                    citation_total > 0,
                "citation_count":
                    citation_total,
                "sql_success":
                    sql_success,
                "sql":
                    sql,
                "api_status":
                    api_status,
                "response_time_seconds":
                    response_time,
                "final_status":
                    final_status,
                "manual_review_required":
                    record["section"]
                    == "semantic",
                "error":
                    error_message,
            }
        )

        print(
            f"  -> {final_status} | "
            f"route={actual_route} | "
            f"score={answer_score}% | "
            f"{response_time:.2f}s"
        )

        if not args.keep_conversations:
            delete_test_conversation(
                api_url=api_url,
                conversation_id=(
                    response.get(
                        "conversation_id"
                    )
                ),
                client_id=client_id,
                timeout=args.timeout,
            )

    total_questions = len(
        results
    )

    passed_questions = sum(
        result["final_status"]
        == "PASS"
        for result in results
    )

    summary = {
        "total_questions":
            total_questions,
        "provider":
            args.provider,
        "pass":
            passed_questions,
        "partial":
            sum(
                result["final_status"]
                == "PARTIAL"
                for result in results
            ),
        "fail":
            sum(
                result["final_status"]
                == "FAIL"
                for result in results
            ),
        "errors":
            sum(
                result["final_status"]
                == "ERROR"
                for result in results
            ),
        "estimated_strict_accuracy_percent":
            round(
                (
                    100
                    * passed_questions
                    / total_questions
                ),
                2,
            ),
        "route_accuracy_percent":
            round(
                (
                    100
                    * sum(
                        bool(
                            result[
                                "route_correct"
                            ]
                        )
                        for result
                        in results
                    )
                    / total_questions
                ),
                2,
            ),
        "citation_coverage_percent":
            round(
                (
                    100
                    * sum(
                        bool(
                            result[
                                "citations_present"
                            ]
                        )
                        for result
                        in results
                    )
                    / total_questions
                ),
                2,
            ),
        "average_answer_score_percent":
            round(
                statistics.mean(
                    float(
                        result[
                            "estimated_answer_score_percent"
                        ]
                    )
                    for result
                    in results
                ),
                2,
            ),
        "average_response_time_seconds":
            round(
                statistics.mean(
                    float(
                        result[
                            "response_time_seconds"
                        ]
                    )
                    for result
                    in results
                ),
                3,
            ),
    }

    output_directory = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    csv_path = (
        output_directory
        / f"qa_report_{timestamp}.csv"
    )

    json_path = (
        output_directory
        / f"qa_report_{timestamp}.json"
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== FINAL SUMMARY ==="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"CSV report: {csv_path}"
    )

    print(
        f"JSON report: {json_path}"
    )


if __name__ == "__main__":
    main()