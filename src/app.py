from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from src.llm_factory import get_chat_model
from src.agent import create_agent

load_dotenv()

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    agent = create_agent()
    response = agent.run(user_input)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)