from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Simple in-memory storage for chat history
chat_sessions = {}

@app.route('/')
def home():
    return "Portfolio Chatbot API is running! 🚀"

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Simple chatbot API is working'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Simple hardcoded responses for testing
        responses = {
            'hello': "Hi! I'm your portfolio assistant. Ask me about projects, skills, or experience!",
            'hi': "Hello! How can I help you learn more about this portfolio?",
            'projects': "I've worked on several exciting projects including web applications, AI chatbots, and data analysis tools.",
            'skills': "My skills include React.js, Python, Flask, AI/ML, and full-stack development.",
            'experience': "I have experience in software development, AI integration, and building scalable web applications.",
            'contact': "You can reach out through the contact section of this portfolio website.",
            'default': "That's an interesting question! I'm a simple chatbot right now, but I'm learning more about this portfolio every day."
        }
        
        # Simple keyword matching
        response_key = 'default'
        user_lower = user_message.lower()
        
        for key in responses.keys():
            if key in user_lower:
                response_key = key
                break
        
        bot_response = responses[response_key]
        
        return jsonify({
            'response': bot_response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'message': 'API is working perfectly!',
        'endpoints': [
            'GET /',
            'GET /api/health',
            'POST /api/chat',
            'GET /api/test'
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
