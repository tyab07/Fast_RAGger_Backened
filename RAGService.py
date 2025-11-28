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
            print(f"✅ Successfully ingested {filename}")
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

    def call_deepseek(self, prompt: str, system_instruction: str = None, temperature: float = 0.3) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "FastBot"
        }
        
        # Friendly but strict persona
        sys_content = system_instruction or "You are a helpful assistant."

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt}
            ],
            # Temperature 0.3: Natural language, but stays on topic
            "temperature": temperature, 
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
        # 1. Remove <think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 2. Remove meta-commentary (e.g., "Based on the text")
        # Since temp is higher, we need to be careful to catch these if they slip in
        patterns = [
            r'^Based on the provided.*?(\n|$)',
            r'^According to the.*?(\n|$)',
            r'^The documents state.*?(\n|$)',
            r'^I checked the context.*?(\n|$)',
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        return cleaned.strip()

    def generate_answer(self, question: str) -> str:
        # 1. Retrieve Context
        query_embedding = self.get_local_embedding(question)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=5)
        
        relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
        context = "\n\n".join(relevant_chunks)

        if not context.strip(): 
            return "I'm sorry, I don't have information about that in the university records."

        # 2. System Instruction: Persona = Friendly Advisor, but Rules = Strict Data Use
        system_instruction = (
            "You are a helpful and friendly Student Advisor for FAST University. "
            "Answer the user's question conversationally. "
            "IMPORTANT RULES: "
            "1. You must ONLY use the information provided in the Context Block below. "
            "2. Do NOT use outside knowledge (like general politics, world leaders, or general coding). "
            "3. Do NOT mention that you are reading from a 'file', 'context', or 'document'. Just give the answer naturally."
        )

        # 3. User Prompt: Explicitly hides the mechanism
        prompt = f"""
        Internal Knowledge Base (Use this information ONLY):
        ---------------------
        {context}
        ---------------------

        User Question: {question}

        Directives:
        - If the answer is found in the Internal Knowledge Base, explain it clearly and warmly.
        - If the answer is NOT in the Internal Knowledge Base, politely say: "I'm sorry, but I can only answer questions related to the university information I have access to."
        - Do NOT describe your search process (e.g., do not say "I found this section...").
        """

        # 4. Call Model with 0.3 Temperature (Human-like but grounded)
        raw_answer = self.call_deepseek(prompt, system_instruction, temperature=0.3)
        return self.clean_response(raw_answer)

    def generate_chat_title(self, first_message: str) -> str:
        prompt = f"Summarize this query into a short 3-5 word title: '{first_message}'. No quotes."
        raw_title = self.call_deepseek(prompt, system_instruction="Title Generator", temperature=0.5)
        return self.clean_response(raw_title).replace('"', '').replace("Title:", "").strip()