// Функция для отправки сообщения
function sendMessage() {
    const inputField = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const userMessage = inputField.value.trim();

    if (userMessage) {
        // Добавить сообщение пользователя в чат
        const userMessageElement = document.createElement("div");
        userMessageElement.className = "message user-message";
        userMessageElement.textContent = userMessage;
        chatBox.appendChild(userMessageElement);

        // Прокрутка вниз после добавления сообщения
        chatBox.scrollTop = chatBox.scrollHeight;

        // Очистка поля ввода
        inputField.value = "";

        // Создаём FormData и отправляем на сервер
        const formData = new FormData();
        formData.append('message', userMessage);

        fetch('/chat', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json()) // Получаем JSON ответ
        .then(data => {
            // Добавить ответ бота в чат
            const botMessageElement = document.createElement("div");
            botMessageElement.className = "message bot-message";
            botMessageElement.textContent = data.response_text;  // Используем response_text из JSON
            chatBox.appendChild(botMessageElement);

            // Прокрутка вниз после ответа бота
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(error => {
            console.error('Ошибка:', error);
        });
    }
}

// Отправка сообщения при нажатии Enter
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});
