from __future__ import annotations

from typing import Any


MESSAGES = {
    "English": {
        "no_records": "No matching records were found.",
        "verified_answer": "The verified answer is {value}.",
        "no_information": (
            "No matching information was found "
            "in the indexed datasets."
        ),
        "structured_failed": (
            "The database query could not be completed safely."
        ),
        "semantic_failed": (
            "Reliable information could not be retrieved "
            "from the knowledge base."
        ),
        "answer_failed": (
            "Matching information was found, but the "
            "final answer could not be generated."
        ),
        "unexpected_error": (
            "The request could not be completed because "
            "an unexpected error occurred."
        ),
    },

    "Hindi": {
        "no_records": "कोई मेल खाने वाला रिकॉर्ड नहीं मिला।",
        "verified_answer": "सत्यापित उत्तर {value} है।",
        "no_information": (
            "इंडेक्स किए गए डेटासेट में कोई मेल "
            "खाने वाली जानकारी नहीं मिली।"
        ),
        "structured_failed": (
            "डेटाबेस क्वेरी सुरक्षित रूप से पूरी "
            "नहीं की जा सकी।"
        ),
        "semantic_failed": (
            "ज्ञान-संग्रह से विश्वसनीय जानकारी "
            "प्राप्त नहीं की जा सकी।"
        ),
        "answer_failed": (
            "मेल खाने वाली जानकारी मिली, लेकिन "
            "अंतिम उत्तर तैयार नहीं किया जा सका।"
        ),
        "unexpected_error": (
            "एक अप्रत्याशित त्रुटि के कारण अनुरोध "
            "पूरा नहीं किया जा सका।"
        ),
    },

    "Marathi": {
        "no_records": "कोणतीही जुळणारी नोंद सापडली नाही.",
        "verified_answer": "पडताळलेले उत्तर {value} आहे.",
        "no_information": (
            "अनुक्रमित डेटासेटमध्ये कोणतीही "
            "जुळणारी माहिती सापडली नाही."
        ),
        "structured_failed": (
            "डेटाबेस क्वेरी सुरक्षितपणे पूर्ण "
            "करता आली नाही."
        ),
        "semantic_failed": (
            "ज्ञानसंग्रहातून विश्वसनीय माहिती "
            "मिळवता आली नाही."
        ),
        "answer_failed": (
            "जुळणारी माहिती मिळाली, परंतु अंतिम "
            "उत्तर तयार करता आले नाही."
        ),
        "unexpected_error": (
            "अनपेक्षित त्रुटीमुळे विनंती पूर्ण "
            "करता आली नाही."
        ),
    },
}


def normalize_language(
    language: str | None,
) -> str:
    value = str(
        language or ""
    ).strip().casefold()

    if (
        value in {"mr", "mr-in", "मराठी"}
        or "marathi" in value
    ):
        return "Marathi"

    if (
        value in {
            "hi",
            "hi-in",
            "हिंदी",
            "हिन्दी",
        }
        or "hindi" in value
    ):
        return "Hindi"

    return "English"


def localized_message(
    message_key: str,
    language: str | None,
    **values: Any,
) -> str:
    selected_language = (
        normalize_language(language)
    )

    language_messages = MESSAGES[
        selected_language
    ]

    template = language_messages.get(
        message_key,
        MESSAGES["English"][
            "unexpected_error"
        ],
    )

    return template.format(
        **values
    )