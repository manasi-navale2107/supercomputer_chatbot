from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_frontend_contains_chat_history_controls():
    html = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="newChatBtn"' in html
    assert 'id="conversationList"' in html
    assert 'id="chatBox"' in html


def test_frontend_calls_delete_conversation_endpoint():
    javascript = (PROJECT_ROOT / "static" / "script.js").read_text(
        encoding="utf-8"
    )
    assert 'method: "DELETE"' in javascript
    assert "deleteChat(" in javascript

