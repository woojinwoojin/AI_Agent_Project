const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatBox = document.getElementById("chat-box");
const sendButton = document.getElementById("send-button");
const quickQuestions = document.querySelectorAll(".quick-question");

let loadingMessage = null;

function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
}

function createMessageElement(sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message");

    if (sender === "user") {
        messageDiv.classList.add("user-message");
    } else {
        messageDiv.classList.add("bot-message");
    }

    return messageDiv;
}

function addMessage(text, sender) {
    const messageDiv = createMessageElement(sender);

    const labelDiv = document.createElement("div");
    labelDiv.classList.add("message-label");
    labelDiv.textContent = sender === "user" ? "나" : "AI 학과 길잡이";

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("message-content");
    contentDiv.textContent = text;

    messageDiv.appendChild(labelDiv);
    messageDiv.appendChild(contentDiv);

    chatBox.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

function showLoading() {
    loadingMessage = createMessageElement("bot");

    const labelDiv = document.createElement("div");
    labelDiv.classList.add("message-label");
    labelDiv.textContent = "AI 학과 길잡이";

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("message-content");
    contentDiv.innerHTML = `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span> 자료를 찾고 답변을 정리하는 중이에요.`;

    loadingMessage.appendChild(labelDiv);
    loadingMessage.appendChild(contentDiv);

    chatBox.appendChild(loadingMessage);
    scrollToBottom();
}

function hideLoading() {
    if (loadingMessage) {
        loadingMessage.remove();
        loadingMessage = null;
    }
}

function addBotResponse(data) {
    const messageDiv = createMessageElement("bot");

    const labelDiv = document.createElement("div");
    labelDiv.classList.add("message-label");
    labelDiv.textContent = data.type === "guardrail" ? "확인 필요" : "AI 학과 길잡이";

    const contentDiv = document.createElement("div");
    contentDiv.classList.add("message-content");
    contentDiv.textContent = data.answer;

    messageDiv.appendChild(labelDiv);
    messageDiv.appendChild(contentDiv);

    if (data.sources && data.sources.length > 0) {
        const sourcesDiv = document.createElement("div");
        sourcesDiv.classList.add("sources-box");

        const titleDiv = document.createElement("div");
        titleDiv.classList.add("sources-title");
        titleDiv.textContent = "참고한 자료";
        sourcesDiv.appendChild(titleDiv);

        data.sources.forEach((source, index) => {
            const itemDiv = document.createElement("div");
            itemDiv.classList.add("source-item");

            const sourceName = source.source || "출처 없음";
            const page = source.page ? ` / ${source.page}` : "";
            const score = source.score !== undefined
                ? ` / 관련도 ${Number(source.score).toFixed(3)}`
                : "";

            itemDiv.textContent = `${index + 1}. ${sourceName}${page}${score}`;
            sourcesDiv.appendChild(itemDiv);
        });

        messageDiv.appendChild(sourcesDiv);
    }

    chatBox.appendChild(messageDiv);
    scrollToBottom();
}

async function sendMessage(message) {
    if (!message) {
        return;
    }

    addMessage(message, "user");
    messageInput.value = "";
    sendButton.disabled = true;
    showLoading();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message }),
        });

        if (!response.ok) {
            throw new Error("서버 응답 오류");
        }

        const data = await response.json();

        hideLoading();
        addBotResponse(data);
    } catch (error) {
        hideLoading();
        addMessage("오류가 발생했어요. 서버 상태와 API Key, DB 연결 상태를 확인해주세요.", "bot");
        console.error(error);
    } finally {
        sendButton.disabled = false;
        messageInput.focus();
    }
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = messageInput.value.trim();
    await sendMessage(message);
});

quickQuestions.forEach((button) => {
    button.addEventListener("click", async () => {
        const message = button.textContent.trim();
        await sendMessage(message);
    });
});