import os
import requests
import re
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class RAGService:
    def __init__(self):
        self.api_key = os.getenv("DS_KEY")
        # You can try 'deepseek/deepseek-chat' if 'r1' is too verbose, 
        # but this code handles cleaning 'r1' output.
        self.model_name = os.getenv("MODEL", "deepseek/deepseek-r1:free")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.chroma_path = "chroma_persistent_storage"
        self.doc_path = "./Data/txtFiles"
        
        print("==== Initializing RAG Service ====")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_qa_collection",
            embedding_function=None
        )
        
        self.sync_documents()

    def get_local_embedding(self, text):
        return self.embedding_model.encode(text).tolist()

    def sync_documents(self):
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)
            return

        files_on_disk = [f for f in os.listdir(self.doc_path) if f.endswith(".txt")]
        if not files_on_disk:
            return

        for filename in files_on_disk:
            existing = self.collection.get(where={"source": filename}, limit=1)
            if len(existing['ids']) == 0:
                print(f"➕ Found new file: {filename}. Ingesting...")
                self.ingest_file(filename)

    def ingest_file(self, filename):
        file_path = os.path.join(self.doc_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            chunks = self._split_text(text)
            
            ids, docs, metadatas, embeddings = [], [], [], []
            for i, chunk in enumerate(chunks):
                ids.append(f"{filename}_chunk{i+1}")
                docs.append(chunk)
                metadatas.append({"source": filename}) 
                embeddings.append(self.get_local_embedding(chunk))

            self.collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        except Exception as e:
            print(f"❌ Error ingesting {filename}: {e}")

    def _split_text(self, text, chunk_size=1000, chunk_overlap=20):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - chunk_overlap
        return chunks

    def call_deepseek(self, prompt: str, system_instruction: str = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "FastBot"
        }
        
        # Stricter System Prompt to prevent reasoning output
        sys_content = system_instruction or "You are a helpful assistant. Output ONLY the answer. Do NOT output internal reasoning, thought processes, or <think> tags."

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1, # Lower temp for more deterministic, concise answers
            "max_tokens": 512
        }
        
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"DeepSeek Error: {e}")
            return "Error generating response."

    def clean_response(self, text: str) -> str:
        # 1. Remove <think>...</think> blocks (DeepSeek R1 standard)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 2. Aggressively remove conversational reasoning fillers if tags aren't used
        # This Regex looks for common starting phrases of reasoning
        cleaned = re.sub(r'^(Okay|Alright|Hmm|Let me|I need to|The user wants).*?(\n|$)', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # 3. Clean up leading/trailing whitespace
        return cleaned.strip()

    def generate_answer(self, question: str) -> str:
        query_embedding = self.get_local_embedding(question)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
        relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
        context = "\n\n".join(relevant_chunks)

        if not context: return "I couldn't find relevant information in the documents."

        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer directly and concisely."
        raw_answer = self.call_deepseek(prompt)
        return self.clean_response(raw_answer)

    def generate_chat_title(self, first_message: str) -> str:
        # Prompt explicitly asks for ONLY the title
        prompt = f"Summarize this into a 3-5 word title: '{first_message}'. Return ONLY the title text. Do not use quotes or prefixes."
        
        raw_title = self.call_deepseek(prompt, system_instruction="You are a title generator. Output ONLY the title. No reasoning.")
        
        # Clean and ensure it's short
        clean_title = self.clean_response(raw_title).replace('"', '').replace("Title:", "").strip()
        
        # Fallback if cleaning failed and it's still too long
        if len(clean_title) > 50: 
            return first_message[:30] + "..."
            
        return clean_title if clean_title else "New Chat"