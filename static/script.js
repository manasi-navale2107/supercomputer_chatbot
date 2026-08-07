let recognition = null;
let isListening = false;
let recognitionBaseText = "";
let recognitionHasFinalText = false;

let currentConversationId = null;
let isSending = false;
let activeController = null;
let selectedImageBase64 = null;
let selectedImageMimeType = null;
let speechSessionId = 0;

const modelSelect = document.getElementById("modelSelect");
const languageSelect = document.getElementById("languageSelect");
const micBtn = document.getElementById("micBtn");
const voiceOutputBtn = document.getElementById("voiceOutputBtn");
const chatBox = document.getElementById("chatBox");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const sendButtonIcon = sendBtn.querySelector(".send-button-icon");
const sendButtonText = sendBtn.querySelector(".send-button-text");
const newChatBtn = document.getElementById("newChatBtn");
const conversationList = document.getElementById("conversationList");
const toast = document.getElementById("toast");
const imageInput = document.getElementById("imageInput");
const imageUploadBtn = document.getElementById("imageUploadBtn");
const imagePreviewContainer = document.getElementById(
    "imagePreviewContainer"
);
const imagePreview = document.getElementById("imagePreview");
const imageFileName = document.getElementById("imageFileName");
const removeImageBtn = document.getElementById("removeImageBtn");

const preferredLanguageKey =
    "supercomputer_preferred_language";

const llmProviderKey =
    "supercomputer_llm_provider";

const voiceOutputKey =
    "supercomputer_voice_output";

let voiceOutputEnabled =
    localStorage.getItem(voiceOutputKey) === "true";

const savedLanguage =
    localStorage.getItem(preferredLanguageKey);

if (
    ["auto", "en-IN", "hi-IN", "mr-IN"]
        .includes(savedLanguage)
) {
    languageSelect.value = savedLanguage;
}

const savedLlmProvider =
    localStorage.getItem(llmProviderKey);

if (
    ["ollama", "groq"]
        .includes(savedLlmProvider)
) {
    modelSelect.value = savedLlmProvider;
}


function getClientId() {
    const storageKey =
        "supercomputer_qa_client_id";

    let storedClientId =
        localStorage.getItem(storageKey);

    if (!storedClientId) {
        storedClientId =
            typeof crypto.randomUUID === "function"
                ? crypto.randomUUID()
                : (
                    "10000000-1000-4000-" +
                    "8000-100000000000"
                ).replace(
                    /[018]/g,
                    (character) =>
                        (
                            character ^
                            (
                                crypto.getRandomValues(
                                    new Uint8Array(1)
                                )[0] &
                                (
                                    15 >>
                                    (
                                        character / 4
                                    )
                                )
                            )
                        ).toString(16)
                );

        localStorage.setItem(
            storageKey,
            storedClientId
        );
    }

    return storedClientId;
}


const clientId = getClientId();


function showToast(
    message,
    isError = false
) {
    toast.textContent = message;

    toast.classList.toggle(
        "error",
        isError
    );

    toast.classList.add("show");

    window.setTimeout(
        () =>
            toast.classList.remove(
                "show"
            ),
        2600
    );
}


function autoResizeInput() {
    input.style.height = "auto";

    const maximumHeight = 160;

    input.style.height =
        `${Math.min(
            input.scrollHeight,
            maximumHeight
        )}px`;

    input.style.overflowY =
        input.scrollHeight >
        maximumHeight
            ? "auto"
            : "hidden";
}


function clearSelectedImage() {
    selectedImageBase64 = null;
    selectedImageMimeType = null;

    imageInput.value = "";

    imagePreview.removeAttribute(
        "src"
    );

    imageFileName.textContent = "";

    imagePreviewContainer.hidden =
        true;
}


function handleImageSelection() {
    const file =
        imageInput.files[0];

    if (!file) {
        clearSelectedImage();
        return;
    }

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (
        !allowedTypes.includes(
            file.type
        )
    ) {
        clearSelectedImage();

        showToast(
            "Only JPEG, PNG, and WebP images are supported.",
            true
        );

        return;
    }

    if (
        file.size >
        10 * 1024 * 1024
    ) {
        clearSelectedImage();

        showToast(
            "The image must be smaller than 10 MB.",
            true
        );

        return;
    }

    const reader =
        new FileReader();

    reader.addEventListener(
        "load",
        () => {
            const dataUrl =
                String(
                    reader.result || ""
                );

            const separatorIndex =
                dataUrl.indexOf(",");

            if (
                separatorIndex < 0
            ) {
                clearSelectedImage();

                showToast(
                    "The selected image could not be read.",
                    true
                );

                return;
            }

            selectedImageBase64 =
                dataUrl.slice(
                    separatorIndex + 1
                );

            selectedImageMimeType =
                file.type;

            imagePreview.src =
                dataUrl;

            imageFileName.textContent =
                file.name;

            imagePreviewContainer.hidden =
                false;

            input.focus();
        }
    );

    reader.addEventListener(
        "error",
        () => {
            clearSelectedImage();

            showToast(
                "The selected image could not be read.",
                true
            );
        }
    );

    reader.readAsDataURL(file);
}


function setSendingState(
    sending
) {
    isSending = sending;

    sendBtn.disabled = sending;
    input.disabled = sending;
    newChatBtn.disabled = sending;

    languageSelect.disabled =
        sending || isListening;

    modelSelect.disabled =
        sending;

    micBtn.disabled =
        sending || !recognition;

    imageInput.disabled =
        sending;

    imageUploadBtn.disabled =
        sending;

    removeImageBtn.disabled =
        sending;

    sendButtonIcon.textContent =
        sending
            ? "…"
            : "↑";

    sendButtonText.textContent =
        sending
            ? "Working"
            : "Send";

    sendBtn.classList.toggle(
        "sending",
        sending
    );

    updateVoiceOutputButton();
}


function renderMarkdown(
    element,
    markdownText
) {
    const markdown =
        String(
            markdownText || ""
        ).trim();

    if (!markdown) {
        element.replaceChildren();
        return;
    }

    if (
        typeof window.marked ===
            "undefined" ||
        typeof window.DOMPurify ===
            "undefined"
    ) {
        element.textContent =
            markdown;

        return;
    }

    window.marked.setOptions({
        gfm: true,
        breaks: true
    });

    const unsafeHtml =
        window.marked.parse(
            markdown
        );

    element.innerHTML =
        window.DOMPurify.sanitize(
            unsafeHtml,
            {
                USE_PROFILES: {
                    html: true
                }
            }
        );

    for (
        const link of
        element.querySelectorAll("a")
    ) {
        link.target = "_blank";

        link.rel =
            "noopener noreferrer";
    }
}


function scrollChatToBottom() {
    chatBox.scrollTop =
        chatBox.scrollHeight;
}


function updateMicButton() {
    micBtn.classList.toggle(
        "listening",
        isListening
    );

    micBtn.textContent =
        isListening
            ? "⏹️"
            : "🎙️";

    micBtn.setAttribute(
        "aria-pressed",
        String(isListening)
    );

    micBtn.setAttribute(
        "aria-label",
        isListening
            ? "Stop voice input"
            : "Start voice input"
    );

    micBtn.title =
        isListening
            ? "Stop listening"
            : "Start voice input";

    languageSelect.disabled =
        isSending ||
        isListening;
}


function updateVoiceOutputButton() {
    const speechSupported =
        "speechSynthesis" in window;

    if (!speechSupported) {
        voiceOutputBtn.disabled =
            true;

        voiceOutputBtn.textContent =
            "🔇";

        voiceOutputBtn.title =
            "Voice output is not supported by this browser";

        return;
    }

    voiceOutputBtn.disabled =
        isSending;

    voiceOutputBtn.textContent =
        voiceOutputEnabled
            ? "🔊"
            : "🔇";

    voiceOutputBtn.setAttribute(
        "aria-pressed",
        String(
            voiceOutputEnabled
        )
    );

    voiceOutputBtn.setAttribute(
        "aria-label",
        voiceOutputEnabled
            ? "Disable voice responses"
            : "Enable voice responses"
    );

    voiceOutputBtn.title =
        voiceOutputEnabled
            ? "Disable voice responses"
            : "Enable voice responses";
}


function toggleVoiceOutput() {
    if (
        !(
            "speechSynthesis"
            in window
        )
    ) {
        showToast(
            "Voice output is not supported by this browser.",
            true
        );

        return;
    }

    voiceOutputEnabled =
        !voiceOutputEnabled;

    localStorage.setItem(
        voiceOutputKey,
        String(
            voiceOutputEnabled
        )
    );

    if (!voiceOutputEnabled) {
        stopSpeech();
    }

    updateVoiceOutputButton();

    showToast(
        voiceOutputEnabled
            ? "Voice responses enabled."
            : "Voice responses disabled."
    );
}


function stopSpeech() {
    speechSessionId += 1;

    if (
        "speechSynthesis"
        in window
    ) {
        window
            .speechSynthesis
            .cancel();
    }
}


function speechLocale(
    responseLanguage,
    text
) {
    const normalizedLanguage =
        String(
            responseLanguage || ""
        )
            .trim()
            .toLowerCase();

    if (
        normalizedLanguage === "mr-in" ||
        normalizedLanguage === "mr" ||
        normalizedLanguage.includes(
            "marathi"
        ) ||
        normalizedLanguage.includes(
            "मराठी"
        )
    ) {
        return "mr-IN";
    }

    if (
        normalizedLanguage === "hi-in" ||
        normalizedLanguage === "hi" ||
        normalizedLanguage.includes(
            "hindi"
        ) ||
        normalizedLanguage.includes(
            "हिन्दी"
        ) ||
        normalizedLanguage.includes(
            "हिंदी"
        )
    ) {
        return "hi-IN";
    }

    if (
        normalizedLanguage === "en-in" ||
        normalizedLanguage === "en" ||
        normalizedLanguage.includes(
            "english"
        )
    ) {
        return "en-IN";
    }

    if (
        languageSelect.value !==
        "auto"
    ) {
        return languageSelect.value;
    }

    const answerText =
        String(text || "");

    const marathiWords =
        /(?:आहे|आहेत|नाही|मध्ये|आणि|करा|केले|झाले|माहिती|संगणक|यामध्ये|त्यामुळे)/;

    if (
        marathiWords.test(
            answerText
        )
    ) {
        return "mr-IN";
    }

    if (
        /[\u0900-\u097F]/.test(
            answerText
        )
    ) {
        return "hi-IN";
    }

    return "en-IN";
}


function selectSpeechVoice(
    voices,
    language
) {
    const normalizedLanguage =
        language
            .toLowerCase()
            .replace("_", "-");

    const languagePrefix =
        normalizedLanguage
            .split("-")[0];

    const exactVoice =
        voices.find(
            (voice) =>
                voice.lang
                    .toLowerCase()
                    .replace("_", "-") ===
                normalizedLanguage
        );

    if (exactVoice) {
        return exactVoice;
    }

    const relatedVoice =
        voices.find(
            (voice) =>
                voice.lang
                    .toLowerCase()
                    .startsWith(
                        `${languagePrefix}-`
                    )
        );

    if (relatedVoice) {
        return relatedVoice;
    }

    if (
        languagePrefix === "mr"
    ) {
        const hindiVoice =
            voices.find(
                (voice) =>
                    voice.lang
                        .toLowerCase()
                        .startsWith(
                            "hi"
                        )
            );

        if (hindiVoice) {
            return hindiVoice;
        }
    }

    return (
        voices.find(
            (voice) =>
                voice.default
        ) ||
        null
    );
}


function textForSpeech(text) {
    return String(text || "")
        .replace(
            /```[\s\S]*?```/g,
            " "
        )
        .replace(
            /https?:\/\/\S+/g,
            " "
        )
        .replace(
            /[*_#`>|]/g,
            " "
        )
        .replace(
            /\s+/g,
            " "
        )
        .trim();
}


function splitSpeechIntoChunks(
    text,
    maximumLength = 220
) {
    const sentences =
        String(text || "")
            .match(
                /[^.!?।]+[.!?।]?/g
            )
        ||
        [String(text || "")];

    const chunks = [];
    let currentChunk = "";

    for (
        const sentence
        of sentences
    ) {
        const cleanedSentence =
            String(
                sentence
            ).trim();

        if (!cleanedSentence) {
            continue;
        }

        const combined =
            currentChunk
                ? (
                    `${currentChunk} ` +
                    cleanedSentence
                )
                : cleanedSentence;

        if (
            combined.length <=
            maximumLength
        ) {
            currentChunk =
                combined;

            continue;
        }

        if (currentChunk) {
            chunks.push(
                currentChunk
            );
        }

        if (
            cleanedSentence.length <=
            maximumLength
        ) {
            currentChunk =
                cleanedSentence;

            continue;
        }

        currentChunk = "";

        for (
            const word
            of cleanedSentence
                .split(/\s+/)
        ) {
            const candidate =
                currentChunk
                    ? (
                        `${currentChunk} ` +
                        word
                    )
                    : word;

            if (
                candidate.length >
                maximumLength
            ) {
                if (currentChunk) {
                    chunks.push(
                        currentChunk
                    );
                }

                currentChunk =
                    word;

            } else {
                currentChunk =
                    candidate;
            }
        }
    }

    if (currentChunk) {
        chunks.push(
            currentChunk
        );
    }

    return chunks;
}


function speakAnswer(
    text,
    responseLanguage
) {
    if (
        !(
            "speechSynthesis"
            in window
        )
    ) {
        showToast(
            "Voice output is not supported by this browser.",
            true
        );

        return;
    }

    const spokenText =
        textForSpeech(text);

    if (!spokenText) {
        return;
    }

    stopSpeech();

    const currentSessionId =
        speechSessionId;

    const language =
        speechLocale(
            responseLanguage,
            spokenText
        );

    const speechChunks =
        splitSpeechIntoChunks(
            spokenText
        );

    function startSpeech(
        attempt = 0
    ) {
        if (
            currentSessionId !==
            speechSessionId
        ) {
            return;
        }

        const voices =
            window
                .speechSynthesis
                .getVoices();

        if (
            voices.length === 0 &&
            attempt < 10
        ) {
            window.setTimeout(
                () =>
                    startSpeech(
                        attempt + 1
                    ),
                200
            );

            return;
        }

        const selectedVoice =
            selectSpeechVoice(
                voices,
                language
            );

        function speakChunk(
            chunkIndex
        ) {
            if (
                currentSessionId !==
                    speechSessionId ||
                chunkIndex >=
                    speechChunks.length
            ) {
                return;
            }

            const utterance =
                new SpeechSynthesisUtterance(
                    speechChunks[
                        chunkIndex
                    ]
                );

            utterance.lang =
                language;

            utterance.rate = 0.92;
            utterance.pitch = 1;
            utterance.volume = 1;

            if (selectedVoice) {
                utterance.voice =
                    selectedVoice;
            }

            utterance.addEventListener(
                "end",
                () =>
                    speakChunk(
                        chunkIndex + 1
                    )
            );

            utterance.addEventListener(
                "error",
                (event) => {
                    if (
                        ![
                            "canceled",
                            "interrupted"
                        ].includes(
                            event.error
                        )
                    ) {
                        console.error(
                            "Speech error:",
                            event.error
                        );
                    }
                }
            );

            window
                .speechSynthesis
                .speak(
                    utterance
                );
        }

        speakChunk(0);
    }

    window.setTimeout(
        () =>
            startSpeech(),
        100
    );
}


function initializeSpeechRecognition() {
    const SpeechRecognition =
        window.SpeechRecognition ||
        window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        micBtn.disabled = true;

        micBtn.title =
            "Voice input is not supported by this browser";

        return;
    }

    recognition =
        new SpeechRecognition();

    recognition.continuous =
        false;

    recognition.interimResults =
        true;

    recognition.maxAlternatives =
        1;

    recognition.addEventListener(
        "start",
        () => {
            isListening = true;

            recognitionHasFinalText =
                false;

            updateMicButton();

            showToast(
                "Listening… Speak your question."
            );
        }
    );

    recognition.addEventListener(
        "result",
        (event) => {
            let finalText = "";
            let interimText = "";

            for (
                let index = 0;
                index <
                event.results.length;
                index += 1
            ) {
                const transcript =
                    event
                        .results[index][0]
                        .transcript;

                if (
                    event
                        .results[index]
                        .isFinal
                ) {
                    finalText +=
                        transcript;

                } else {
                    interimText +=
                        transcript;
                }
            }

            recognitionHasFinalText =
                Boolean(
                    finalText.trim()
                );

            input.value = [
                recognitionBaseText,
                finalText,
                interimText
            ]
                .filter(Boolean)
                .join(" ")
                .replace(
                    /\s+/g,
                    " "
                )
                .trim();

            autoResizeInput();
        }
    );

    recognition.addEventListener(
        "error",
        (event) => {
            recognitionHasFinalText =
                false;

            if (
                event.error !==
                "aborted"
            ) {
                showToast(
                    event.error ===
                    "not-allowed"
                        ? (
                            "Microphone " +
                            "permission was denied."
                        )
                        : (
                            "Voice input could " +
                            "not be captured."
                        ),
                    true
                );
            }
        }
    );

    recognition.addEventListener(
        "end",
        () => {
            const shouldSend =
                recognitionHasFinalText &&
                Boolean(
                    input.value.trim()
                );

            isListening = false;

            updateMicButton();

            recognitionBaseText = "";
            recognitionHasFinalText =
                false;

            if (
                shouldSend &&
                !isSending
            ) {
                window.setTimeout(
                    () =>
                        sendMessage(
                            "voice"
                        ),
                    0
                );

            } else {
                input.focus();
            }
        }
    );

    micBtn.disabled = false;

    updateMicButton();
}


function toggleVoiceInput() {
    if (
        !recognition ||
        isSending
    ) {
        return;
    }

    if (isListening) {
        recognition.stop();
        return;
    }

    stopSpeech();

    recognition.lang =
        languageSelect.value ===
        "auto"
            ? (
                navigator.language ||
                "en-IN"
            )
            : languageSelect.value;

    recognitionBaseText =
        input.value.trim();

    try {
        recognition.start();

    } catch (error) {
        console.error(
            "Could not start voice input:",
            error
        );

        showToast(
            "Voice input is already active.",
            true
        );
    }
}


function appendSql(
    container,
    sql
) {
    if (!sql) {
        return;
    }

    const sqlBox =
        document.createElement(
            "pre"
        );

    sqlBox.className =
        "sql-box";

    sqlBox.textContent =
        sql;

    container.appendChild(
        sqlBox
    );
}


function appendCitations(
    container,
    citations
) {
    if (
        !citations ||
        Object.keys(
            citations
        ).length === 0
    ) {
        return;
    }

    const card =
        document.createElement(
            "div"
        );

    card.className =
        "citations-box";

    const heading =
        document.createElement(
            "h4"
        );

    heading.textContent =
        "📚 Evidence Used";

    card.appendChild(
        heading
    );

    function addValue(
        label,
        value
    ) {
        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return;
        }

        const paragraph =
            document.createElement(
                "p"
            );

        const strong =
            document.createElement(
                "strong"
            );

        strong.textContent =
            `${label}: `;

        paragraph.append(
            strong,
            document.createTextNode(
                String(value)
            )
        );

        card.appendChild(
            paragraph
        );
    }

    addValue(
        "Route",
        citations.route
    );

    if (
        Array.isArray(
            citations.datasets
        ) &&
        citations.datasets.length
    ) {
        const label =
            document.createElement(
                "p"
            );

        const strong =
            document.createElement(
                "strong"
            );

        strong.textContent =
            "Datasets";

        label.appendChild(
            strong
        );

        card.appendChild(
            label
        );

        const list =
            document.createElement(
                "ul"
            );

        for (
            const dataset
            of citations.datasets
        ) {
            const item =
                document.createElement(
                    "li"
                );

            item.textContent =
                dataset;

            list.appendChild(
                item
            );
        }

        card.appendChild(
            list
        );
    }

    if (
        Array.isArray(
            citations.documents
        ) &&
        citations.documents.length
    ) {
        const label =
            document.createElement(
                "p"
            );

        const strong =
            document.createElement(
                "strong"
            );

        strong.textContent =
            "Retrieved sources";

        label.appendChild(
            strong
        );

        card.appendChild(
            label
        );

        const list =
            document.createElement(
                "ul"
            );

        for (
            const documentCitation
            of citations.documents
        ) {
            const item =
                document.createElement(
                    "li"
                );

            const parts = [
                documentCitation.title ||
                documentCitation.source ||
                "Unknown source"
            ];

            if (
                documentCitation.table &&
                documentCitation.table !==
                documentCitation.source
            ) {
                parts.push(
                    `Dataset: ${
                        documentCitation.table
                    }`
                );
            }

            if (
                documentCitation.score !==
                    null &&
                documentCitation.score !==
                    undefined
            ) {
                parts.push(
                    `Score: ${
                        Number(
                            documentCitation.score
                        ).toFixed(4)
                    }`
                );
            }

            if (
                documentCitation
                    .retrieval_type
            ) {
                parts.push(
                    `Match: ${
                        documentCitation
                            .retrieval_type
                    }`
                );
            }

            item.textContent =
                parts.join(" | ");

            list.appendChild(
                item
            );
        }

        card.appendChild(
            list
        );
    }

    if (citations.sql) {
        const label =
            document.createElement(
                "p"
            );

        const strong =
            document.createElement(
                "strong"
            );

        strong.textContent =
            "Verified SQL";

        label.appendChild(
            strong
        );

        card.appendChild(
            label
        );

        appendSql(
            card,
            citations.sql
        );
    }

    addValue(
        "Search query",
        citations.retrieval_query
    );

    addValue(
        "Rows/Documents",
        citations.row_count
    );

    container.appendChild(
        card
    );
}


function removeMessageEditor(
    message,
    text
) {
    const editor =
        message.querySelector(
            ".message-editor"
        );

    if (editor) {
        editor.remove();
    }

    text.hidden = false;

    const actions =
        message.querySelector(
            ".message-actions"
        );

    if (actions) {
        actions.hidden = false;
    }
}


async function saveEditedMessage(
    message,
    text,
    messageId,
    editedContent,
    saveButton
) {
    const cleanedContent =
        editedContent.trim();

    if (!cleanedContent) {
        showToast(
            "The question cannot be empty.",
            true
        );

        return;
    }

    if (
        cleanedContent ===
        text.textContent.trim()
    ) {
        removeMessageEditor(
            message,
            text
        );

        return;
    }

    const confirmed =
        window.confirm(
            "Edit this question and generate a new answer? " +
            "Messages after this question will be replaced."
        );

    if (!confirmed) {
        return;
    }

    saveButton.disabled = true;
    saveButton.textContent =
        "Saving…";

    setSendingState(true);
    stopSpeech();

    try {
        const response =
            await fetch(
                `/messages/${messageId}`,
                {
                    method: "PATCH",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            client_id:
                                clientId,

                            content:
                                cleanedContent,

                            preferred_language:
                                languageSelect.value,

                            llm_provider:
                                modelSelect.value
                        })
                }
            );

        if (!response.ok) {
            let errorMessage =
                "The question could not be edited.";

            try {
                const errorPayload =
                    await response.json();

                if (
                    errorPayload.detail
                ) {
                    errorMessage =
                        String(
                            errorPayload.detail
                        );
                }

            } catch (_) {
                // Use fallback.
            }

            throw new Error(
                errorMessage
            );
        }

        const result =
            await response.json();

        currentConversationId =
            result.conversation_id;

        await loadConversation(
  result.conversation_id,
  true,
);

        showToast(
            "Question edited and answer regenerated."
        );

    } catch (error) {
        console.error(
            "Message edit failed:",
            error
        );

        removeMessageEditor(
            message,
            text
        );

        showToast(
            error.message ||
            "The question could not be edited.",
            true
        );

    } finally {
        setSendingState(false);
        input.focus();
    }
}


function beginMessageEdit(
    message,
    text,
    messageId
) {
    if (
        isSending ||
        message.querySelector(
            ".message-editor"
        )
    ) {
        return;
    }

    stopSpeech();

    const actions =
        message.querySelector(
            ".message-actions"
        );

    if (actions) {
        actions.hidden = true;
    }

    text.hidden = true;

    const editor =
        document.createElement(
            "div"
        );

    editor.className =
        "message-editor";

    const editorInput =
        document.createElement(
            "textarea"
        );

    editorInput.className =
        "message-editor-input";

    editorInput.maxLength = 4000;
    editorInput.rows = 3;
    editorInput.value =
        text.textContent;

    const editorActions =
        document.createElement(
            "div"
        );

    editorActions.className =
        "message-editor-actions";

    const cancelButton =
        document.createElement(
            "button"
        );

    cancelButton.type = "button";

    cancelButton.className =
        "message-edit-cancel";

    cancelButton.textContent =
        "Cancel";

    const saveButton =
        document.createElement(
            "button"
        );

    saveButton.type = "button";

    saveButton.className =
        "message-edit-save";

    saveButton.textContent =
        "Save & submit";

    cancelButton.addEventListener(
        "click",
        () =>
            removeMessageEditor(
                message,
                text
            )
    );

    saveButton.addEventListener(
        "click",
        () =>
            saveEditedMessage(
                message,
                text,
                messageId,
                editorInput.value,
                saveButton
            )
    );

    editorInput.addEventListener(
        "keydown",
        (event) => {
            if (
                event.key ===
                "Escape"
            ) {
                event.preventDefault();

                removeMessageEditor(
                    message,
                    text
                );
            }

            if (
                event.key ===
                    "Enter" &&
                !event.shiftKey
            ) {
                event.preventDefault();

                saveButton.click();
            }
        }
    );

    editorActions.append(
        cancelButton,
        saveButton
    );

    editor.append(
        editorInput,
        editorActions
    );

    message.appendChild(
        editor
    );

    editorInput.focus();

    editorInput.setSelectionRange(
        editorInput.value.length,
        editorInput.value.length
    );
}


function enableUserMessageEditing(
    message,
    text,
    messageId,
    metadata = {}
) {
    if (
        !messageId ||
        metadata?.has_image
    ) {
        return;
    }

    message.dataset.messageId =
        String(messageId);

    let actions =
        message.querySelector(
            ".message-actions"
        );

    if (!actions) {
        actions =
            document.createElement(
                "div"
            );

        actions.className =
            "message-actions";

        message.appendChild(
            actions
        );
    }

    if (
        actions.querySelector(
            ".edit-message-btn"
        )
    ) {
        return;
    }

    const editButton =
        document.createElement(
            "button"
        );

    editButton.type = "button";

    editButton.className =
        "edit-message-btn";

    editButton.textContent = "✎";

    editButton.title =
        "Edit question";

    editButton.setAttribute(
        "aria-label",
        "Edit this question"
    );

    editButton.addEventListener(
        "click",
        () =>
            beginMessageEdit(
                message,
                text,
                messageId
            )
    );

    actions.appendChild(
        editButton
    );
}


function appendMessage(
  role,
  content,
  sql = null,
  citations = null,
  messageId = null,
  metadata = {},
) {
  const message = document.createElement("div");

  message.className = `message ${role}`;

  // User ने upload केलेली image chat bubble मध्ये दाखवणे.
  if (
    role === "user" &&
    metadata?.has_image &&
    metadata?.image_data_url
  ) {
    const uploadedImage = document.createElement("img");

    uploadedImage.className = "chat-message-image";
    uploadedImage.src = metadata.image_data_url;
    uploadedImage.alt = "Uploaded image";
    uploadedImage.loading = "lazy";

    uploadedImage.addEventListener("click", () => {
      window.open(
        metadata.image_data_url,
        "_blank",
        "noopener,noreferrer",
      );
    });

    message.appendChild(uploadedImage);
  }

  const text = document.createElement("div");

  text.className = "message-text";

  if (role === "assistant") {
    text.classList.add("answer-text");

    renderMarkdown(text, content);
  } else {
    text.textContent = content;
  }

  message.appendChild(text);

  if (role === "user") {
    enableUserMessageEditing(
      message,
      text,
      messageId,
      metadata,
    );
  }

  appendSql(message, sql);

  if (role === "assistant") {
    appendCitations(
      message,
      citations,
    );
  }

  chatBox.appendChild(message);

  scrollChatToBottom();

  return message;
}


async function loadConversations() {
    try {
        const response =
            await fetch(
                (
                    `/conversations?client_id=${
                        encodeURIComponent(
                            clientId
                        )
                    }`
                    +
                    "&page=1&limit=100"
                )
            );

        if (!response.ok) {
            throw new Error(
                "Could not load conversations."
            );
        }

        const data =
            await response.json();

        conversationList
            .replaceChildren();

        if (
            !data.conversations.length
        ) {
            const empty =
                document.createElement(
                    "div"
                );

            empty.className =
                "empty-conversations";

            empty.textContent =
                "No saved chats yet.";

            conversationList
                .appendChild(
                    empty
                );

            return;
        }

        for (
            const conversation
            of data.conversations
        ) {
            const item =
                document.createElement(
                    "div"
                );

            item.className =
                "conversation";

            if (
                conversation.id ===
                currentConversationId
            ) {
                item.classList.add(
                    "active"
                );
            }

            item.title =
                conversation.title;

            item.addEventListener(
                "click",
                () =>
                    loadConversation(
                        conversation.id
                    )
            );

            const title =
                document.createElement(
                    "div"
                );

            title.className =
                "conversation-title";

            title.textContent =
                conversation.title;

            const deleteButton =
                document.createElement(
                    "button"
                );

            deleteButton.type =
                "button";

            deleteButton.className =
                "delete-chat-btn";

            deleteButton.textContent =
                "🗑";

            deleteButton.title =
                "Delete chat";

            deleteButton.setAttribute(
                "aria-label",
                `Delete ${
                    conversation.title
                }`
            );

            deleteButton
                .addEventListener(
                    "click",
                    (event) => {
                        event
                            .stopPropagation();

                        deleteChat(
                            conversation.id,
                            conversation.title
                        );
                    }
                );

            item.append(
                title,
                deleteButton
            );

            conversationList
                .appendChild(
                    item
                );
        }

    } catch (error) {
        console.error(
            "Error loading conversations:",
            error
        );

        showToast(
            "Could not load saved chats.",
            true
        );
    }
}


async function loadConversation(
  conversationId,
  forceReload = false,
) {
  if (
    isSending &&
    !forceReload
  ) {
    return;
  }

    stopSpeech();

    currentConversationId =
        conversationId;

    chatBox.replaceChildren();

    try {
        const response =
            await fetch(
                (
                    `/conversations/${
                        conversationId
                    }/messages`
                    +
                    `?client_id=${
                        encodeURIComponent(
                            clientId
                        )
                    }`
                    +
                    "&page=1&limit=100"
                )
            );

        if (!response.ok) {
            throw new Error(
                "Could not load messages."
            );
        }

        const data =
            await response.json();

        for (
            const message
            of data.messages
        ) {
            appendMessage(
                message.role,
                message.content,
                message.sql,
                (
                    message
                        .metadata
                        ?.citations ||
                    {}
                ),
                message.id,
                message.metadata || {}
            );
        }

        await loadConversations();

    } catch (error) {
        console.error(
            "Error loading messages:",
            error
        );

        currentConversationId =
            null;

        showToast(
            "Could not open this chat.",
            true
        );
    }
}


async function deleteChat(
    conversationId,
    title
) {
    if (isSending) {
        return;
    }

    const confirmed =
        window.confirm(
            `Delete “${title}”? ` +
            (
                "This will permanently " +
                "remove all messages " +
                "in this chat."
            )
        );

    if (!confirmed) {
        return;
    }

    try {
        const response =
            await fetch(
                (
                    `/conversations/${
                        conversationId
                    }`
                    +
                    `?client_id=${
                        encodeURIComponent(
                            clientId
                        )
                    }`
                ),
                {
                    method: "DELETE"
                }
            );

        if (!response.ok) {
            throw new Error(
                "Delete failed."
            );
        }

        if (
            currentConversationId ===
            conversationId
        ) {
            currentConversationId =
                null;

            chatBox.replaceChildren();
        }

        await loadConversations();

        showToast(
            "Chat deleted."
        );

    } catch (error) {
        console.error(
            "Error deleting conversation:",
            error
        );

        showToast(
            "Could not delete the chat.",
            true
        );
    }
}


function appendActivity(
    activityBox,
    message
) {
    const line =
        document.createElement(
            "div"
        );

    line.textContent =
        message;

    activityBox.appendChild(
        line
    );
}


async function processSsePayload(
    payload,
    uiState
) {
    const payloadType =
        payload.type ||
        payload.event ||
        "";

    if (
        payloadType === "step"
    ) {
        if (payload.message) {
            appendActivity(
                uiState.activityBox,
                payload.message
            );
        }

    } else if (
        payloadType === "route"
    ) {
        uiState.route =
            payload.route;

    } else if (
        payloadType === "sql"
    ) {
        uiState.finalSql =
            payload.sql;

    } else if (
        payloadType ===
        "answer_chunk"
    ) {
        uiState.answerMarkdown +=
            payload.chunk || "";

        uiState
            .answerText
            .textContent =
                uiState.answerMarkdown;

        scrollChatToBottom();

    } else if (
        payloadType === "error"
    ) {
        uiState.errorReceived =
            true;

        uiState.answerMarkdown =
            payload.message ||
            (
                "An unexpected " +
                "error occurred."
            );

        uiState
            .answerText
            .textContent =
                uiState.answerMarkdown;

        if (
            uiState
                .activityBox
                .isConnected
        ) {
            uiState
                .activityBox
                .remove();
        }

        scrollChatToBottom();

    } else if (
        payloadType === "done"
    ) {
        uiState.doneReceived =
            true;

        if (payload.answer) {
            uiState.answerMarkdown =
                String(
                    payload.answer
                );
        }

        renderMarkdown(
            uiState.answerText,
            uiState.answerMarkdown
        );

        uiState.finalSql =
            payload.sql ||
            uiState.finalSql;

        uiState.finalCitations =
            payload.citations ||
            {};

        if (
            uiState.finalSql &&
            !uiState.sqlRendered
        ) {
            appendSql(
                uiState.assistantMessage,
                uiState.finalSql
            );

            uiState.sqlRendered =
                true;
        }

        uiState.completedAnswer =
            payload.answer ||
            uiState.answerMarkdown ||
            uiState
                .answerText
                .textContent;

        appendCitations(
            uiState.assistantMessage,
            uiState.finalCitations
        );

        uiState.responseLanguage =
            payload.response_language ||
            "";

        if (
            uiState
                .activityBox
                .isConnected
        ) {
            uiState
                .activityBox
                .remove();
        }

        scrollChatToBottom();

    } else if (
        payloadType === "saved"
    ) {
        currentConversationId =
            payload.conversation_id;

        if (
            uiState.userMessageElement &&
            payload.user_message?.id
        ) {
            const userText =
                uiState
                    .userMessageElement
                    .querySelector(
                        ".message-text"
                    );

            if (userText) {
                enableUserMessageEditing(
                    uiState.userMessageElement,
                    userText,
                    payload.user_message.id,
                    (
                        payload
                            .user_message
                            .metadata ||
                        {}
                    )
                );
            }
        }

        await loadConversations();
    }
}


async function sendMessage(
    inputMode = "text"
) {
    const enteredContent =
        input.value.trim();

    const hasImage =
        Boolean(
            selectedImageBase64 &&
            selectedImageMimeType
        );

    const content =
        enteredContent ||
        (
            hasImage
                ? (
                    "Describe this image " +
                    "and explain its " +
                    "important details."
                )
                : ""
        );

    if (
        !content ||
        isSending
    ) {
        return;
    }

    const imageBase64 =
        hasImage
            ? selectedImageBase64
            : null;

    const imageMimeType =
        hasImage
            ? selectedImageMimeType
            : null;
    
    const imageDataUrl = hasImage
  ? `data:${imageMimeType};base64,${imageBase64}`
  : null;

    stopSpeech();

    const userMessageElement = appendMessage(
  "user",
  content,
  null,
  null,
  null,
  {
    has_image: hasImage,
    image_mime_type: imageMimeType,
    image_data_url: imageDataUrl,
  },
);

    input.value = "";

    autoResizeInput();
    clearSelectedImage();
    setSendingState(true);

    const assistantMessage =
        document.createElement(
            "div"
        );

    assistantMessage.className =
        "message assistant";

    const activityBox =
        document.createElement(
            "div"
        );

    activityBox.className =
        "activity-box";

    assistantMessage.appendChild(
        activityBox
    );

    const answerText =
        document.createElement(
            "div"
        );

    answerText.className =
        "answer-text";

    assistantMessage.appendChild(
        answerText
    );

    chatBox.appendChild(
        assistantMessage
    );

    const uiState = {
        userMessageElement,
        assistantMessage,
        activityBox,
        answerText,
        answerMarkdown: "",
        finalSql: "",
        finalCitations: {},
        route: null,
        sqlRendered: false,
        completedAnswer: "",
        responseLanguage: "",
        doneReceived: false,
        errorReceived: false
    };

    activeController =
        new AbortController();

    try {
        const response =
            await fetch(
                "/messages/stream",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            client_id:
                                clientId,

                            conversation_id:
                                currentConversationId,

                            content,

                            preferred_language:
                                languageSelect.value,

                            llm_provider:
                                modelSelect.value,

                            image_base64:
                                imageBase64,

                            image_mime_type:
                                imageMimeType
                        }),

                    signal:
                        activeController.signal
                }
            );

        if (
            !response.ok ||
            !response.body
        ) {
            let message =
                (
                    "Something went wrong " +
                    "while processing " +
                    "your question."
                );

            try {
                const errorResponse =
                    await response.json();

                if (
                    errorResponse.detail
                ) {
                    message =
                        errorResponse.detail;
                }

            } catch (_) {
                // Use fallback.
            }

            answerText.textContent =
                message;

            if (
                activityBox.isConnected
            ) {
                activityBox.remove();
            }

            if (
                inputMode === "voice" ||
                voiceOutputEnabled
            ) {
                speakAnswer(
                    message,
                    languageSelect.value
                );
            }

            return;
        }

        const reader =
            response.body.getReader();

        const decoder =
            new TextDecoder(
                "utf-8"
            );

        let buffer = "";

        while (true) {
            const {
                value,
                done
            } =
                await reader.read();

            if (done) {
                break;
            }

            buffer +=
                decoder.decode(
                    value,
                    {
                        stream: true
                    }
                );

            const events =
                buffer.split(
                    "\n\n"
                );

            buffer =
                events.pop() ||
                "";

            for (
                const event
                of events
            ) {
                const dataLine =
                    event
                        .split("\n")
                        .find(
                            (line) =>
                                line.startsWith(
                                    "data: "
                                )
                        );

                if (!dataLine) {
                    continue;
                }

                const payload =
                    JSON.parse(
                        dataLine.slice(6)
                    );

                await processSsePayload(
                    payload,
                    uiState
                );
            }
        }

        const remainingEvent =
            buffer.trim();

        if (
            remainingEvent.startsWith(
                "data: "
            )
        ) {
            const payload =
                JSON.parse(
                    remainingEvent.slice(6)
                );

            await processSsePayload(
                payload,
                uiState
            );
        }

        if (
            !uiState.doneReceived &&
            !uiState.errorReceived
        ) {
            throw new Error(
                "The response stream ended " +
                "before the final event."
            );
        }

        if (
            inputMode === "voice" ||
            voiceOutputEnabled
        ) {
            speakAnswer(
                (
                    uiState.completedAnswer ||
                    uiState.answerMarkdown ||
                    answerText.textContent
                ),
                uiState.responseLanguage
            );
        }

    } catch (error) {
        if (
            error.name !==
            "AbortError"
        ) {
            console.error(
                "Streaming error:",
                error
            );

            const errorMessage =
                (
                    "Error while connecting " +
                    "to the backend."
                );

            answerText.textContent =
                errorMessage;

            if (
                activityBox.isConnected
            ) {
                activityBox.remove();
            }

            if (
                inputMode === "voice" ||
                voiceOutputEnabled
            ) {
                speakAnswer(
                    errorMessage,
                    languageSelect.value
                );
            }
        }

    } finally {
        activeController = null;

        setSendingState(false);

        input.focus();
    }
}


newChatBtn.addEventListener(
    "click",
    () => {
        if (isSending) {
            return;
        }

        stopSpeech();

        if (
            isListening &&
            recognition
        ) {
            recognition.abort();
        }

        currentConversationId =
            null;

        chatBox.replaceChildren();

        input.value = "";

        autoResizeInput();
        clearSelectedImage();
        loadConversations();
        input.focus();
    }
);


imageUploadBtn.addEventListener(
    "click",
    () => {
        if (!isSending) {
            imageInput.click();
        }
    }
);


imageInput.addEventListener(
    "change",
    handleImageSelection
);


removeImageBtn.addEventListener(
    "click",
    clearSelectedImage
);


sendBtn.addEventListener(
    "click",
    () =>
        sendMessage("text")
);


micBtn.addEventListener(
    "click",
    toggleVoiceInput
);


voiceOutputBtn.addEventListener(
    "click",
    toggleVoiceOutput
);


input.addEventListener(
    "input",
    autoResizeInput
);


languageSelect.addEventListener(
    "change",
    () => {
        localStorage.setItem(
            preferredLanguageKey,
            languageSelect.value
        );
    }
);


modelSelect.addEventListener(
    "change",
    () => {
        localStorage.setItem(
            llmProviderKey,
            modelSelect.value
        );

        const providerName =
            modelSelect.value ===
            "groq"
                ? "Groq Cloud"
                : "Ollama Local";

        showToast(
            `AI provider changed to ${providerName}.`
        );
    }
);


input.addEventListener(
    "keydown",
    (event) => {
        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();

            sendMessage("text");
        }
    }
);


updateVoiceOutputButton();
initializeSpeechRecognition();
loadConversations();
autoResizeInput();
input.focus();