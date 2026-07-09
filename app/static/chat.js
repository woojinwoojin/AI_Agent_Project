const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const chatBox = document.getElementById("chat-box");

function addMessage(text, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message");

    if (sender === "user") {
        messageDiv.classList.add("user-message");
    } else {
        messageDiv.classList.add("bot-message");
    }

    messageDiv.textContent = text;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addBotResponse(data) {
    let text = data.answer;

    if (data.sources && data.sources.length > 0) {
    text += "\n\n[출처]";
    data.sources.forEach((source, index) => {
        const title = source.title || source.source || "출처 없음";
        const page = source.source_page || source.page || "";

        text += `\n${index + 1}. ${title}`;

        if (page) {
            text += ` / ${page}`;
        }

        if (source.score !== undefined) {
            text += ` / 유사도: ${Number(source.score).toFixed(3)}`;
        }
    });
}

    addMessage(text, "bot");
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = messageInput.value.trim();

    if (!message) {
        return;
    }

    addMessage(message, "user");
    messageInput.value = "";

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
        addBotResponse(data);
    } catch (error) {
        addMessage("오류가 발생했어요. 서버 상태를 확인해주세요.", "bot");
        console.error(error);
    }
});