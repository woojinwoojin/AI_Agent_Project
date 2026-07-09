const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatBox = document.getElementById("chat-box");
const sendButton = document.getElementById("send-button");
const quickQuestions = document.querySelectorAll(".quick-question");

let loadingMessage = null;

function nowText() {
    const now = new Date();
    return now.toLocaleTimeString("ko-KR", {
        hour: "2-digit",
        minute: "2-digit",
    });
}

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

function createBotAvatar() {
    const avatar = document.createElement("img");
    avatar.src = "/static/img/mascot.png";
    avatar.alt = "가천이";
    avatar.classList.add("avatar");
    return avatar;
}

function addUserMessage(text) {
    const messageDiv = createMessageElement("user");

    const wrap = document.createElement("div");
    wrap.classList.add("bubble-wrap");

    const bubble = document.createElement("div");
    bubble.classList.add("bubble");
    bubble.textContent = text;

    const time = document.createElement("span");
    time.classList.add("timestamp");
    time.textContent = nowText();

    wrap.appendChild(bubble);
    wrap.appendChild(time);
    messageDiv.appendChild(wrap);
    chatBox.appendChild(messageDiv);
    scrollToBottom();
}

function addBotMessage(text, options = {}) {
    const messageDiv = createMessageElement("bot");
    messageDiv.appendChild(createBotAvatar());

    const wrap = document.createElement("div");
    wrap.classList.add("bubble-wrap");

    const bubble = document.createElement("div");
    bubble.classList.add("bubble");
    bubble.textContent = text;

    wrap.appendChild(bubble);

    if (options.sources && options.sources.length > 0) {
        const sourcesBox = document.createElement("div");
        sourcesBox.classList.add("sources-box");

        const title = document.createElement("div");
        title.classList.add("sources-title");
        title.textContent = "참고한 자료";
        sourcesBox.appendChild(title);

        options.sources.forEach((source, index) => {
            const item = document.createElement("div");
            item.classList.add("source-item");

            const sourceName = source.source || source.title || "출처 없음";
            const page = source.page || source.source_page || "";
            const score = source.score !== undefined
                ? ` · 관련도 ${Number(source.score).toFixed(3)}`
                : "";

            item.textContent = `${index + 1}. ${sourceName}${page ? ` / ${page}` : ""}${score}`;
            sourcesBox.appendChild(item);
        });

        wrap.appendChild(sourcesBox);
    }

    const time = document.createElement("span");
    time.classList.add("timestamp");
    time.textContent = nowText();
    wrap.appendChild(time);

    messageDiv.appendChild(wrap);
    chatBox.appendChild(messageDiv);
    scrollToBottom();

    return messageDiv;
}

function showLoading() {
    const messageDiv = createMessageElement("bot");
    messageDiv.appendChild(createBotAvatar());

    const wrap = document.createElement("div");
    wrap.classList.add("bubble-wrap");

    const bubble = document.createElement("div");
    bubble.classList.add("bubble");
    bubble.innerHTML = `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span> 자료를 찾고 답변을 정리하는 중이에요.`;

    wrap.appendChild(bubble);
    messageDiv.appendChild(wrap);

    loadingMessage = messageDiv;
    chatBox.appendChild(messageDiv);
    scrollToBottom();
}

function hideLoading() {
    if (loadingMessage) {
        loadingMessage.remove();
        loadingMessage = null;
    }
}

async function sendMessage(message) {
    const trimmed = message.trim();
    if (!trimmed) {
        return;
    }

    addUserMessage(trimmed);
    messageInput.value = "";
    sendButton.disabled = true;
    showLoading();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message: trimmed }),
        });

        if (!response.ok) {
            throw new Error("서버 응답 오류");
        }

        const data = await response.json();

        hideLoading();
        addBotMessage(data.answer, {
            sources: data.sources || [],
            type: data.type,
        });
    } catch (error) {
        hideLoading();
        addBotMessage("오류가 발생했어요. 서버 상태, DB 연결, API Key를 확인해주세요.");
        console.error(error);
    } finally {
        sendButton.disabled = false;
        messageInput.focus();
    }
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendMessage(messageInput.value);
});

quickQuestions.forEach((button) => {
    button.addEventListener("click", async () => {
        const question = button.dataset.question || button.textContent;
        await sendMessage(question);
    });
});
