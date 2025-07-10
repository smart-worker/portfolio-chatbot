import os
import pickle
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentProcessor:
    def __init__(self, knowledge_base_path="knowledge_base.pkl"):
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        self.documents = []
        self.embeddings = None
        self.index = None

    def add_text_document(self, title, content):
        doc = Document(
            page_content=content,
            metadata={"title": title, "source": "manual"}
        )
        chunks = self.text_splitter.split_documents([doc])
        self.documents.extend(chunks)
        print(f"✅ Added '{title}' with {len(chunks)} chunks. Total documents: {len(self.documents)}")

    def add_pdf(self, file_path):
        try:
            pdfreader = PdfReader(file_path)
            raw_text = ''
            for page in pdfreader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content

            if not raw_text.strip():
                print(f"⚠️ No text could be extracted from {file_path}. Skipping.")
                return

            texts = self.text_splitter.split_text(raw_text)
            chunks = [Document(page_content=t, metadata={"source": file_path}) for t in texts]
            self.documents.extend(chunks)
            print(f"✅ Added {file_path} with {len(chunks)} chunks. Total documents: {len(self.documents)}")
        except Exception as e:
            print(f"❌ Error loading PDF {file_path}: {e}")

    def add_website(self, url):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            if not docs:
                print(f"⚠️ No documents loaded from {url}")
                return
            chunks = self.text_splitter.split_documents(docs)
            self.documents.extend(chunks)
            print(f"✅ Added {url} with {len(chunks)} chunks. Total documents: {len(self.documents)}")
        except Exception as e:
            print(f"❌ Error loading {url}: {e}")

    def build_index(self):
        if not self.documents:
            print("⚠️ No documents to index! Aborting index build.")
            return False

        texts = [doc.page_content for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts)
        if len(self.embeddings) == 0:
            print("⚠️ No embeddings created! Aborting index build.")
            return False

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))

        print(f"✅ Built FAISS index with {len(self.documents)} document chunks.")
        return True

    def save_knowledge_base(self):
        if not self.documents or self.index is None:
            print("⚠️ Nothing to save. Aborting.")
            return

        serializable_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]

        knowledge_base = {
            'documents': serializable_docs,
            'embeddings': self.embeddings,
            'index': faiss.serialize_index(self.index)
        }

        with open(self.knowledge_base_path, 'wb') as f:
            pickle.dump(knowledge_base, f)

        print(f"💾 Knowledge base saved to {self.knowledge_base_path} "
              f"(size: {os.path.getsize(self.knowledge_base_path)} bytes)")

    def load_knowledge_base(self):
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'rb') as f:
                    knowledge_base = pickle.load(f)

                self.documents = [
                    Document(page_content=doc_dict["page_content"], metadata=doc_dict["metadata"])
                    for doc_dict in knowledge_base.get('documents', [])
                ]
                self.embeddings = knowledge_base.get('embeddings')
                if knowledge_base.get('index'):
                    self.index = faiss.deserialize_index(knowledge_base['index'])

                print(f"✅ Loaded {len(self.documents)} documents from knowledge base.")
            except Exception as e:
                print(f"❌ Error loading knowledge base: {e}")
        else:
            print("ℹ️ No existing knowledge base found. Starting fresh.")

# ---------------- Main Driver ----------------
if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.documents = []

    # Add sources
    processor.add_pdf("kb-docs/KGP_CV.pdf")
    processor.add_website("https://sarkarsoham.vercel.app")
    processor.add_text_document(
        "About Soham",
        "Soham is a student at IITKgp pursuing M.Tech in Computer Science. He has previously worked as a software engineer. He is great at what he does."
    )

    # Build and save
    if processor.build_index():
        processor.save_knowledge_base()
        print("✅ Knowledge base created successfully!")
    else:
        print("❌ Knowledge base creation failed.")