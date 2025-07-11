from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

class SimpleChatbot:
    def __init__(self, knowledge_base_path="knowledge_base.pkl"):
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.chat_history = []
        self.embeddings = None
        self.system_prompt_template = self._load_prompt_template()
        self.index = None
        # Load knowledge base
        self.load_knowledge_base(knowledge_base_path)

    def _load_prompt_template(self):
        return """
## Identity
You are the Assistant AI Agent for SOHAM's portfolio website. Your role is to interact with visitors, address their inquiries, and provide assistance with common details about Soham.

## Scope
- Focus on visitor inquiries about Soham's education, career, projects, and contact support.
- Do not handle personal questions about Soham, you can share email and phone number if the visitor asks.
- Redirect or escalate issues outside your expertise to the Contact Me section.

## Responsibility
- Initiate interactions with a friendly greeting.
- Guide the conversation based on visitors' interest.
- Provide accurate and concise information.
- Redirect or escalate issues outside your expertise to the Contact Me section.

## Response Style
- Maintain a friendly, clear, and professional tone.
- Keep responses brief and to the point.
- Use buttons for quick replies and easy navigation whenever possible.

## Ability
- Delegate specialized tasks to AI-Associates or escalate to a human when needed.

## Guardrails
- **Privacy**: Respect customer privacy; only request personal data if absolutely necessary.
- **Accuracy**: Provide verified and factual responses coming from Knowledge Base or official sources. Avoid speculation.

## Instructions
- **Greeting**: Start every conversation with a friendly welcome.
  _Example_: "Hi, greetings from Portfolio Assistant, what do you want to know about Soham?"

- **Escalation**: When a visitor's query becomes too complex or sensitive, notify the visitor that you'll escalate the conversation to the Contact Me section.
  _Example_: "I’m having trouble answering this. Please visit the Contact Me section and submit your question."

- **Closing**: End interactions by confirming that the visitor's question has been answered.
  _Example_: "Is there anything else I can help you with today?"

Context:
{context}

Instructions:
- Answer based on the provided context when relevant
- If the context doesn't contain relevant information, politely say so
- Be conversational and helpful
- Keep responses concise but informative
"""

    def load_knowledge_base(self, path):
        """Load the knowledge base"""
        try:
            with open(path, 'rb') as f:
                knowledge_base = pickle.load(f)

            # Reconstruct Document objects
            self.documents = [
                Document(page_content=doc["page_content"], metadata=doc["metadata"])
                for doc in knowledge_base.get('documents', [])
            ]

            self.embeddings = knowledge_base.get('embeddings')
            if self.embeddings is None:
                print("⚠️ Embeddings not found in knowledge base.")

            index_data = knowledge_base.get('index')
            if index_data is not None:
                self.index = faiss.deserialize_index(index_data)
            else:
                print("⚠️ No FAISS index found in knowledge base.")

            print(f"✅ Loaded {len(self.documents)} documents from knowledge base.")

        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")

    def get_relevant_context(self, query, top_k=10, max_chars=4000):
        """Get relevant context for the query, limited to max_chars"""
        if not self.index or not self.documents:
            return "No knowledge base available."

        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        context_parts = []
        total_chars = 0

        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.documents) or score < 0.1:
                continue

            doc = self.documents[idx]
            title = doc.metadata.get("title", "Document")
            content = doc.page_content.strip()

            # If adding this chunk would exceed limit, break
            if total_chars + len(content) > max_chars:
                break

            context_parts.append(f"**{title}**: {content}")
            total_chars += len(content)

        return "\n\n".join(context_parts) if context_parts else "No relevant information found."

    def chat(self, message):
        try:
            # If first message, build context and system prompt
            if not self.chat_history:
                context = self.get_relevant_context(message)
                prompt = self.system_prompt_template.format(context=context)
                self.chat_history.append({"role": "system", "content": prompt})

            # Append user message
            self.chat_history.append({"role": "user", "content": message})

            # Call Groq LLM
            completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=self.chat_history,
                temperature=0.5,
                max_tokens=512
            )

            # Get reply and append to history
            reply = completion.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": reply})

            return reply

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def reset_history(self):
        self.chat_history = []

# Initialize chatbot
chatbot = SimpleChatbot()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Single chat endpoint"""
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400

        user_message = data['message'].strip()

        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Generate response
        response = chatbot.chat(user_message)

        return jsonify({
            'response': response,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'documents_loaded': len(chatbot.documents)
    })

@app.route('/api/debug-context', methods=['POST'])
def debug_context():
    try:
        data = request.get_json()
        message = data.get('message', '')
        context = chatbot.get_relevant_context(message)
        return jsonify({'context': context})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    chatbot.reset_history()
    return jsonify({'status': 'chat history cleared'})

port = int(os.environ.get("PORT", 9001))
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
