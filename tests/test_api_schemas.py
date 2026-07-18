from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.api_schemas import MessageRequest


def test_message_request_strips_content():
    request = MessageRequest(
        client_id=uuid4(),
        conversation_id=None,
        content="  Tell me about Capella.  ",
    )
    assert request.content == "Tell me about Capella."


def test_message_request_rejects_blank_content():
    with pytest.raises(ValidationError):
        MessageRequest(client_id=uuid4(), content="   ")


def test_message_request_rejects_invalid_conversation_id():
    with pytest.raises(ValidationError):
        MessageRequest(
            client_id=uuid4(),
            conversation_id=0,
            content="Hello",
        )

