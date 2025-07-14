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
import time

# Load environment variables with override
load_dotenv(override=True, verbose=True)

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
        self._models_loading = False
        self._embedding_model_loading = False
        
        # Load only document metadata at startup (lightweight)
        self.load_knowledge_base_metadata()

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / 1024 / 1024, 2)
        except:
            return 0

    @property
    def groq_client(self):
        """Lazy load Groq client only when needed"""
        if self._groq_client is None:
            try:
                self._groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                print(f"🔧 Groq client initialized - Memory: {self.get_memory_usage()} MB")
            except Exception as e:
                print(f"❌ Error initializing Groq client: {e}")
        return self._groq_client

    @property
    def embedding_model(self):
        """Lazy load embedding model only when needed"""
        if self._embedding_model is None and not self._embedding_model_loading:
            try:
                self._embedding_model_loading = True
                print(f"🔧 Loading SentenceTransformer model... Current memory: {self.get_memory_usage()} MB")
                
                # Use lighter model to save memory
                self._embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                print(f"✅ SentenceTransformer model loaded - Memory: {self.get_memory_usage()} MB")
                self._embedding_model_loading = False
            except Exception as e:
                print(f"❌ Error loading embedding model: {e}")
                self._embedding_model_loading = False
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
                
            print(f"🔧 Loading knowledge base metadata... Memory: {self.get_memory_usage()} MB")
            
            with open(self.knowledge_base_path, 'rb') as f:
                knowledge_base = pickle.load(f)

            # Load only document content and metadata (lightweight)
            self.documents = [
                Document(page_content=doc["page_content"], metadata=doc["metadata"])
                for doc in knowledge_base.get('documents', [])
            ]
            
            print(f"✅ Loaded {len(self.documents)} documents metadata - Memory: {self.get_memory_usage()} MB")
            
        except Exception as e:
            print(f"❌ Error loading knowledge base metadata: {e}")

    def load_full_knowledge_base(self):
        """Load embeddings and FAISS index only when first chat request comes"""
        if self._knowledge_base_loaded:
            return True  # Already loaded
            
        try:
            if not os.path.exists(self.knowledge_base_path):
                print("⚠️ Knowledge base file not found for full loading")
                return False
                
            current_memory = self.get_memory_usage()
            print(f"🔧 Loading full knowledge base... Current memory: {current_memory} MB")
            
            # Safety check - don't load if memory is already high
            if current_memory > 450:
                print(f"⚠️ Memory too high ({current_memory} MB) to safely load knowledge base")
                return False
            
            with open(self.knowledge_base_path, 'rb') as f:
                knowledge_base = pickle.load(f)
            
            self.embeddings = knowledge_base.get('embeddings')
            if self.embeddings is None:
                print("⚠️ Embeddings not found in knowledge base.")
                return False

            index_data = knowledge_base.get('index')
            if index_data is not None:
                self.index = faiss.deserialize_index(index_data)
                print(f"✅ FAISS index loaded - Memory: {self.get_memory_usage()} MB")
            else:
                print("⚠️ No FAISS index found in knowledge base.")
                return False
                
            self._knowledge_base_loaded = True
            print(f"✅ Full knowledge base loaded successfully - Memory: {self.get_memory_usage()} MB")
            return True
            
        except Exception as e:
            print(f"❌ Error loading full knowledge base: {e}")
            return False

    def load_models_gradually(self):
        """Load models one at a time with memory monitoring"""
        print("load models gradually called")
        if self._models_loading:
            return {"status": "Models are currently loading, please wait...", "loading": True}
        
        self._models_loading = True
        results = {"steps": [], "final_memory": 0, "status": "success", "loading": False}
        
        try:
            # Step 1: Load Groq client first (lightweight)
            if self._groq_client is None:
                memory_before = self.get_memory_usage()
                _ = self.groq_client  # Trigger lazy loading
                memory_after = self.get_memory_usage()
                results["steps"].append({
                    "step": "groq_client",
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                    "success": self._groq_client is not None
                })
            
            # Step 2: Load knowledge base if memory permits
            current_memory = self.get_memory_usage()
            if current_memory < 450 and not self._knowledge_base_loaded:
                memory_before = current_memory
                success = self.load_full_knowledge_base()
                memory_after = self.get_memory_usage()
                results["steps"].append({
                    "step": "knowledge_base",
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                    "success": success
                })
                
                if not success:
                    results["status"] = "Knowledge base loading failed due to memory constraints"
            
            # Step 3: Load embedding model if memory permits
            current_memory = self.get_memory_usage()
            if current_memory < 400 and self._embedding_model is None:
                memory_before = current_memory
                _ = self.embedding_model  # Trigger lazy loading
                memory_after = self.get_memory_usage()
                results["steps"].append({
                    "step": "embedding_model",
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                    "success": self._embedding_model is not None
                })
                
                if memory_after > 450:
                    results["status"] = "Memory limit approached after loading embedding model"
            
            results["final_memory"] = self.get_memory_usage()
            self._models_loading = False
            
            return results
            
        except Exception as e:
            self._models_loading = False
            results["status"] = f"Error loading models: {str(e)}"
            results["final_memory"] = self.get_memory_usage()
            return results

    def get_relevant_context(self, query, top_k=5, max_chars=2000):
        """Get relevant context for the query, limited to max_chars"""
        if not self.index or not self.documents or not self.embedding_model:
            return "No knowledge base available."

        try:
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
        except Exception as e:
            print(f"Error in get_relevant_context: {e}")
            return "Error retrieving context."

    def simple_chat_response(self, message):
        """Fallback responses when models can't be loaded due to memory constraints"""
        try:
            if not self.groq_client:
                return "I'm currently experiencing technical difficulties. Please try again later or contact Soham directly."
            
            # Use only Groq client without RAG
            simple_prompt = """You are a helpful assistant for Soham's portfolio website. 
            Provide brief, helpful responses about general portfolio topics like projects, skills, and experience.
            Keep responses concise and professional. If asked about specific details, suggest contacting Soham directly."""
            
            completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": simple_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return "I'm currently experiencing high memory usage. Please try again in a moment, or contact Soham directly for detailed information."

    def chat(self, message):
        try:
            current_memory = self.get_memory_usage()
            
            # If memory is already very high, use simple response
            if current_memory > 450:
                return self.simple_chat_response(message)
            
            # Try to load models if not loaded
            if not self._knowledge_base_loaded or not self.embedding_model:
                loading_result = self.load_models_gradually()
                if loading_result.get("loading"):
                    return "I'm initializing my knowledge base. Please wait a moment and try again."
            
            # Check if we can use full RAG
            can_use_rag = (self._knowledge_base_loaded and 
                          self.embedding_model is not None and 
                          self.get_memory_usage() < 450)
            
            if not can_use_rag:
                return self.simple_chat_response(message)
            
            # Full RAG response
            if not self.chat_history:
                context = self.get_relevant_context(message)
                prompt = self.system_prompt_template.format(context=context)
                self.chat_history.append({"role": "system", "content": prompt})

            self.chat_history.append({"role": "user", "content": message})

            completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=self.chat_history,
                temperature=0.5,
                max_tokens=300
            )

            reply = completion.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            print(f"Error in chat: {e}")
            return self.simple_chat_response(message)

    def reset_history(self):
        self.chat_history = []
        # Force garbage collection
        gc.collect()

# Initialize chatbot with minimal memory footprint
print("🚀 Initializing chatbot...")
chatbot = SimpleChatbot()
print(f"📊 Initial memory usage: {chatbot.get_memory_usage()} MB")

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with memory monitoring and fallback"""
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
        memory_after = chatbot.get_memory_usage()

        # Determine mode based on loaded models
        mode = "full" if (chatbot._knowledge_base_loaded and chatbot._embedding_model) else "simple"

        return jsonify({
            'response': response,
            'status': 'success',
            'mode': mode,
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

@app.route('/api/warmup', methods=['POST'])
def warmup():
    """Gradually warm up models"""
    try:
        result = chatbot.load_models_gradually()
        return jsonify({
            'loading_result': result,
            'memory_usage_mb': chatbot.get_memory_usage(),
            'models_loaded': {
                'groq_client': chatbot._groq_client is not None,
                'embedding_model': chatbot._embedding_model is not None,
                'knowledge_base': chatbot._knowledge_base_loaded
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint with memory info"""
    memory_usage = chatbot.get_memory_usage()
    return jsonify({
        'status': 'healthy',
        'documents_loaded': len(chatbot.documents),
        'knowledge_base_loaded': chatbot._knowledge_base_loaded,
        'memory_usage_mb': memory_usage,
        'memory_percentage_of_limit': round((memory_usage / 512) * 100, 2),
        'models_status': {
            'groq_client': chatbot._groq_client is not None,
            'embedding_model': chatbot._embedding_model is not None,
            'knowledge_base': chatbot._knowledge_base_loaded
        }
    })

@app.route('/api/memory', methods=['GET'])
def memory_status():
    """Detailed memory status endpoint"""
    try:
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
            },
            'models_loading': chatbot._models_loading
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-context', methods=['POST'])
def debug_context():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not chatbot._knowledge_base_loaded or not chatbot.embedding_model:
            return jsonify({
                'context': 'Knowledge base not loaded due to memory constraints',
                'memory_usage_mb': chatbot.get_memory_usage()
            })
        
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
    port = int(os.getenv("PORT", 5000))
    
    print(f"🚀 Starting Flask app on port {port}")
    print(f"📊 Startup memory usage: {chatbot.get_memory_usage()} MB")
    
    # Remove debug=True for production
    app.run(host='0.0.0.0', port=port)
