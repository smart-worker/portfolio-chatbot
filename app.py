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
import psutil
import gc

# Load environment variables
load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

class SimpleChatbot:
    def __init__(self, knowledge_base_path="knowledge_base.pkl"):
        # Use lazy loading - don't initialize heavy models at startup
        self._groq_client = None
        self._embedding_model = None
        self.documents = []
        self.chat_history = []
        self.embeddings = None
        self.system_prompt_template = self._load_prompt_template()
        self.index = None
        self.knowledge_base_path = knowledge_base_path
        self._knowledge_base_loaded = False
        
        # Load only document metadata at startup (lightweight)
        self.load_knowledge_base_metadata()

    @property
    def groq_client(self):
        """Lazy load Groq client only when needed"""
        if self._groq_client is None:
            self._groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            print("🔧 Groq client initialized")
        return self._groq_client

    @property
    def embedding_model(self):
        """Lazy load embedding model only when needed"""
        if self._embedding_model is None:
            print("🔧 Loading SentenceTransformer model...")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ SentenceTransformer model loaded")
        return self._embedding_model

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
  _Example_: "I'm having trouble answering this. Please visit the Contact Me section and submit your question."

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

    def load_knowledge_base_metadata(self):
        """Load only document metadata, not embeddings or FAISS index"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                print(f"⚠️ Knowledge base file {self.knowledge_base_path} not found. Running without knowledge base.")
                return
                
            with open(self.knowledge_base_path, 'rb') as f:
                knowledge_base = pickle.load(f)

            # Load only document content and metadata (lightweight)
            self.documents = [
                Document(page_content=doc["page_content"], metadata=doc["metadata"])
                for doc in knowledge_base.get('documents', [])
            ]
            
            print(f"✅ Loaded {len(self.documents)} documents metadata")
            
        except Exception as e:
            print(f"❌ Error loading knowledge base metadata: {e}")

    def load_full_knowledge_base(self):
        """Load embeddings and FAISS index only when first chat request comes"""
        if self._knowledge_base_loaded:
            return  # Already loaded
            
        try:
            if not os.path.exists(self.knowledge_base_path):
                print("⚠️ Knowledge base file not found for full loading")
                return
                
            print("🔧 Loading full knowledge base (embeddings + FAISS index)...")
            
            with open(self.knowledge_base_path, 'rb') as f:
                knowledge_base = pickle.load(f)
            
            self.embeddings = knowledge_base.get('embeddings')
            if self.embeddings is None:
                print("⚠️ Embeddings not found in knowledge base.")
                return

            index_data = knowledge_base.get('index')
            if index_data is not None:
                self.index = faiss.deserialize_index(index_data)
                print("✅ FAISS index loaded")
            else:
                print("⚠️ No FAISS index found in knowledge base.")
                
            self._knowledge_base_loaded = True
            print("✅ Full knowledge base loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading full knowledge base: {e}")

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
            # Load full knowledge base on first chat request
            self.load_full_knowledge_base()
            
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
        # Force garbage collection
        gc.collect()

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / 1024 / 1024, 2)

# Initialize chatbot with minimal memory footprint
print("🚀 Initializing chatbot...")
chatbot = SimpleChatbot()
print(f"📊 Initial memory usage: {chatbot.get_memory_usage()} MB")

@app.route('/api/chat', methods=['POST'])
def chat():
    """Single chat endpoint with memory monitoring"""
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400

        user_message = data['message'].strip()

        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Check memory before processing
        memory_before = chatbot.get_memory_usage()
        
        # Generate response
        response = chatbot.chat(user_message)
        
        # Check memory after processing
        memory_after = chatbot.get_memory_usage()

        return jsonify({
            'response': response,
            'status': 'success',
            'memory_info': {
                'before_mb': memory_before,
                'after_mb': memory_after,
                'change_mb': round(memory_after - memory_before, 2)
            }
        })

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint with memory info"""
    memory_usage = chatbot.get_memory_usage()
    return jsonify({
        'status': 'healthy',
        'documents_loaded': len(chatbot.documents),
        'knowledge_base_loaded': chatbot._knowledge_base_loaded,
        'memory_usage_mb': memory_usage,
        'memory_percentage_of_limit': round((memory_usage / 512) * 100, 2)
    })

@app.route('/api/memory', methods=['GET'])
def memory_status():
    """Detailed memory status endpoint"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return jsonify({
        'memory_usage_mb': round(memory_info.rss / 1024 / 1024, 2),
        'virtual_memory_mb': round(memory_info.vms / 1024 / 1024, 2),
        'render_limit_mb': 512,
        'percentage_used': round((memory_info.rss / 1024 / 1024 / 512) * 100, 2),
        'models_loaded': {
            'groq_client': chatbot._groq_client is not None,
            'embedding_model': chatbot._embedding_model is not None,
            'knowledge_base': chatbot._knowledge_base_loaded
        }
    })

@app.route('/api/debug-context', methods=['POST'])
def debug_context():
    try:
        data = request.get_json()
        message = data.get('message', '')
        context = chatbot.get_relevant_context(message)
        return jsonify({
            'context': context,
            'memory_usage_mb': chatbot.get_memory_usage()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    chatbot.reset_history()
    return jsonify({
        'status': 'chat history cleared',
        'memory_usage_mb': chatbot.get_memory_usage()
    })

@app.route('/')
def index():
    return f'Portfolio Chatbot API up & running! Memory usage: {chatbot.get_memory_usage()} MB'

if __name__ == '__main__':
    import os
    
    # Debug all environment variables
    print("🔍 All Environment Variables:")
    for key, value in os.environ.items():
        if key in ['PORT', 'GROQ_API_KEY']:
            print(f"   {key}: {value}")
    
    # Multiple ways to get PORT
    port = None
    
    # Method 1: Direct os.environ access
    if 'PORT' in os.environ:
        port = int(os.environ['PORT'])
        print(f"✅ Found PORT via os.environ: {port}")
    
    # Method 2: os.getenv fallback
    elif os.getenv('PORT'):
        port = int(os.getenv('PORT'))
        print(f"✅ Found PORT via os.getenv: {port}")
    
    # Method 3: Default fallback
    else:
        port = 5000
        print(f"⚠️ PORT not found, using default: {port}")
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"📊 Startup memory usage: {chatbot.get_memory_usage()} MB")
    app.run(host='0.0.0.0', port=port)
