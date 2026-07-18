let currentConversationId = null;
let isSending = false;
let activeController = null;

const chatBox = document.getElementById("chatBox");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const conversationList = document.getElementById("conversationList");
const toast = document.getElementById("toast");

function getClientId() {
    const storageKey = "supercomputer_qa_client_id";
    let clientId = localStorage.getItem(storageKey);
    if (!clientId) {
        clientId = typeof crypto.randomUUID === "function"
            ? crypto.randomUUID()
            : "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (character) =>
                (character ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> character / 4)
                    .toString(16)
            );
        localStorage.setItem(storageKey, clientId);
    }
    return clientId;
}

const clientId = getClientId();

function showToast(message, isError = false) {
    toast.textContent = message;
    toast.classList.toggle("error", isError);
    toast.classList.add("show");
    window.setTimeout(() => toast.classList.remove("show"), 2600);
}

function setSendingState(sending) {
    isSending = sending;
    sendBtn.disabled = sending;
    input.disabled = sending;
    newChatBtn.disabled = sending;
    sendBtn.textContent = sending ? "Working..." : "Send";
}

function renderMarkdown(element, markdownText) {
    const markdown = String(markdownText || "").trim();

    if (!markdown) {
        element.replaceChildren();
        return;
    }

    if (
        typeof window.marked === "undefined" ||
        typeof window.DOMPurify === "undefined"
    ) {
        element.textContent = markdown;
        return;
    }

    window.marked.setOptions({
        gfm: true,
        breaks: true
    });

    const unsafeHtml = window.marked.parse(markdown);
    const safeHtml = window.DOMPurify.sanitize(unsafeHtml, {
        USE_PROFILES: {html: true}
    });

    element.innerHTML = safeHtml;

    // Every generated link should open safely.
    for (const link of element.querySelectorAll("a")) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
    }
}


function scrollChatToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendSql(container, sql) {
    if (!sql) return;
    const sqlBox = document.createElement("pre");
    sqlBox.className = "sql-box";
    sqlBox.textContent = sql;
    container.appendChild(sqlBox);
}

function appendMessage(role, content, sql = null) {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    const text = document.createElement("div");
    text.className = "message-text";

    if (role === "assistant") {
        text.classList.add("answer-text");
        renderMarkdown(text, content);
    } else {
        // User input must always remain plain text.
        text.textContent = content;
    }

    message.appendChild(text);
    appendSql(message, sql);

    chatBox.appendChild(message);
    scrollChatToBottom();
}

async function loadConversations() {
    try {
        const response = await fetch(
            `/conversations?client_id=${encodeURIComponent(clientId)}&page=1&limit=100`
        );
        if (!response.ok) throw new Error("Could not load conversations.");
        const data = await response.json();
        conversationList.replaceChildren();

        if (!data.conversations.length) {
            const empty = document.createElement("div");
            empty.className = "empty-conversations";
            empty.textContent = "No saved chats yet.";
            conversationList.appendChild(empty);
            return;
        }

        for (const conversation of data.conversations) {
            const item = document.createElement("div");
            item.className = "conversation";
            if (conversation.id === currentConversationId) {
                item.classList.add("active");
            }
            item.title = conversation.title;
            item.addEventListener("click", () => loadConversation(conversation.id));

            const title = document.createElement("div");
            title.className = "conversation-title";
            title.textContent = conversation.title;

            const deleteButton = document.createElement("button");
            deleteButton.type = "button";
            deleteButton.className = "delete-chat-btn";
            deleteButton.textContent = "🗑";
            deleteButton.title = "Delete chat";
            deleteButton.setAttribute("aria-label", `Delete ${conversation.title}`);
            deleteButton.addEventListener("click", (event) => {
                event.stopPropagation();
                deleteChat(conversation.id, conversation.title);
            });

            item.append(title, deleteButton);
            conversationList.appendChild(item);
        }
    } catch (error) {
        console.error("Error loading conversations:", error);
        showToast("Could not load saved chats.", true);
    }
}

async function loadConversation(conversationId) {
    if (isSending) return;
    currentConversationId = conversationId;
    chatBox.replaceChildren();

    try {
        const response = await fetch(
            `/conversations/${conversationId}/messages` +
            `?client_id=${encodeURIComponent(clientId)}&page=1&limit=100`
        );
        if (!response.ok) throw new Error("Could not load messages.");
        const data = await response.json();

        for (const message of data.messages) {
            appendMessage(message.role, message.content, message.sql);
        }
        await loadConversations();
    } catch (error) {
        console.error("Error loading messages:", error);
        currentConversationId = null;
        showToast("Could not open this chat.", true);
    }
}

async function deleteChat(conversationId, title) {
    if (isSending) return;
    const confirmed = window.confirm(
        `Delete “${title}”? This will permanently remove all messages in this chat.`
    );
    if (!confirmed) return;

    try {
        const response = await fetch(
            `/conversations/${conversationId}?client_id=${encodeURIComponent(clientId)}`,
            {method: "DELETE"}
        );
        if (!response.ok) throw new Error("Delete failed.");

        if (currentConversationId === conversationId) {
            currentConversationId = null;
            chatBox.replaceChildren();
        }
        await loadConversations();
        showToast("Chat deleted.");
    } catch (error) {
        console.error("Error deleting conversation:", error);
        showToast("Could not delete the chat.", true);
    }
}

function appendActivity(activityBox, message) {
    const line = document.createElement("div");
    line.textContent = message;
    activityBox.appendChild(line);
}

async function processSsePayload(payload, uiState) {
    if (payload.type === "step") {
        appendActivity(
            uiState.activityBox,
            payload.message
        );
    }

    else if (payload.type === "route") {
        uiState.route = payload.route;
    }

    else if (payload.type === "sql") {
        uiState.finalSql = payload.sql;
    }

    else if (payload.type === "answer_chunk") {
        uiState.answerMarkdown += payload.chunk || "";

        /*
         * While streaming, keep the partial content as text.
         * Incomplete Markdown is rendered only after the done event.
         */
        uiState.answerText.textContent =
            uiState.answerMarkdown;

        scrollChatToBottom();
    }

    else if (payload.type === "error") {
        uiState.answerMarkdown =
            payload.message || "An unexpected error occurred.";

        uiState.answerText.textContent =
            uiState.answerMarkdown;
    }

    else if (payload.type === "done") {
        if (
            !uiState.answerMarkdown.trim() &&
            payload.answer
        ) {
            uiState.answerMarkdown =
                String(payload.answer);
        }

        // Render completed Markdown only once.
        renderMarkdown(
            uiState.answerText,
            uiState.answerMarkdown
        );

        uiState.finalSql =
            payload.sql || uiState.finalSql;

        if (
            uiState.finalSql &&
            !uiState.sqlRendered
        ) {
            appendSql(
                uiState.assistantMessage,
                uiState.finalSql
            );

            uiState.sqlRendered = true;
        }

        scrollChatToBottom();
    }

    else if (payload.type === "saved") {
        currentConversationId =
            payload.conversation_id;

        await loadConversations();
    }
}
async function sendMessage() {
    const content = input.value.trim();
    if (!content || isSending) return;

    appendMessage("user", content);
    input.value = "";
    setSendingState(true);

    const assistantMessage = document.createElement("div");
    assistantMessage.className = "message assistant";

    const activityBox = document.createElement("div");
    activityBox.className = "activity-box";
    assistantMessage.appendChild(activityBox);

    const answerText = document.createElement("div");
    answerText.className = "answer-text";
    assistantMessage.appendChild(answerText);
    chatBox.appendChild(assistantMessage);

    const uiState = {
    assistantMessage,
    activityBox,
    answerText,
    answerMarkdown: "",
    finalSql: "",
    route: null,
    sqlRendered: false,
};
    activeController = new AbortController();

    try {
        const response = await fetch("/messages/stream", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                client_id: clientId,
                conversation_id: currentConversationId,
                content,
            }),
            signal: activeController.signal,
        });

        if (!response.ok || !response.body) {
            let message = "Something went wrong while processing your question.";
            try {
                const error = await response.json();
                if (error.detail) message = error.detail;
            } catch (_) {
                // Keep the safe fallback message.
            }
            answerText.textContent = message;
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const {value, done} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});
            const events = buffer.split("\n\n");
            buffer = events.pop() || "";

            for (const event of events) {
                const dataLine = event
                    .split("\n")
                    .find((line) => line.startsWith("data: "));
                if (!dataLine) continue;
                const payload = JSON.parse(dataLine.slice(6));
                await processSsePayload(payload, uiState);
            }
        }

        if (buffer.trim().startsWith("data: ")) {
            const payload = JSON.parse(buffer.trim().slice(6));
            await processSsePayload(payload, uiState);
        }
    } catch (error) {
        if (error.name !== "AbortError") {
            console.error("Streaming error:", error);
            answerText.textContent = "Error while connecting to the backend.";
        }
    } finally {
        activeController = null;
        setSendingState(false);
        input.focus();
    }
}

newChatBtn.addEventListener("click", () => {
    if (isSending) return;
    currentConversationId = null;
    chatBox.replaceChildren();
    loadConversations();
    input.focus();
});

sendBtn.addEventListener("click", sendMessage);

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

loadConversations();
input.focus();
