from flask import Flask, render_template, request, jsonify
from neuro_use import assistant_response  # Импортируем модуль нейросети

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message')
    response_data = assistant_response(user_message)  # Обработка сообщения с помощью нейросети

    # Преобразуем ответ в текст, который можно отправить обратно на клиент
    response_text = format_response(response_data)
    return jsonify(response_text=response_text)  # Отправляем JSON ответ

def format_response(response_data):
    # Форматирование ответа для отображения на веб-странице
    response_text = "Сервисы:\n"
    for service in response_data['services']:
        response_text += f"{service['service']} ({service['probability']} вероятность)\n"

    response_text += "\nПохожие обращения:\n"
    if response_data['similar_topics']:
        for topic in response_data['similar_topics']:
            response_text += f"{topic['topic']} (Схожесть: {topic['similarity']}) - Решение: {topic['solution']}\n"
    else:
        response_text += "Похожие обращения не найдены.\n"

    response_text += "\nРелевантные инструкции:\n"
    if response_data['relevant_instructions']:
        for instr in response_data['relevant_instructions']:
            response_text += f"{instr['title']} (Схожесть: {instr['similarity']}) - {instr['content']}\n"
    else:
        response_text += "Релевантные инструкции не найдены.\n"

    return response_text

if __name__ == '__main__':
    app.run(debug=True)
