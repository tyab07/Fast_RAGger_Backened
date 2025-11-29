import os
import requests
import re
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

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
        
        # Roman Urdu detection patterns
        self.roman_urdu_patterns = [
            r'\b(mai?|tum|aap|woh|yeh|hai|hain|tha|thi|the|ka|ki|ke|ko|se|mei?|par)\b',
            r'\b(kya|kyun|kahan|kaise|kitna|kab)\b',
            r'\b(acha|theek|shukriya|meherbani|salam)\b',
        ]
        
        self.sync_documents()

    def get_local_embedding(self, text):
        return self.embedding_model.encode(text).tolist()

    def sync_documents(self):
        """Sync documents from disk to ChromaDB with better detection"""
        if not os.path.exists(self.doc_path):
            print(f"❌ Document path {self.doc_path} does not exist!")
            os.makedirs(self.doc_path)
            return

        files_on_disk = [f for f in os.listdir(self.doc_path) if f.endswith(".txt")]
        print(f"📁 Found {len(files_on_disk)} text files on disk: {files_on_disk}")
        
        if not files_on_disk:
            print("⚠️ No text files found in document directory")
            return

        # Get existing documents in collection to avoid re-ingestion
        existing_docs = self.collection.get()
        existing_sources = set()
        if existing_docs['metadatas']:
            for metadata in existing_docs['metadatas']:
                if 'source' in metadata:
                    existing_sources.add(metadata['source'])
        
        print(f"📚 Existing documents in DB: {existing_sources}")

        for filename in files_on_disk:
            if filename not in existing_sources:
                print(f"➕ Found new file: {filename}. Ingesting...")
                self.ingest_file(filename)
            else:
                print(f"✅ File already in DB: {filename}")

    def ingest_file(self, filename):
        """Ingest a single file into ChromaDB"""
        file_path = os.path.join(self.doc_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
            
            if not text:
                print(f"⚠️ File {filename} is empty")
                return

            print(f"📖 Reading file {filename}, size: {len(text)} characters")
            
            chunks = self._split_text_smart(text)
            print(f"📄 Split into {len(chunks)} chunks")
            
            ids, docs, metadatas, embeddings = [], [], [], []
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                # Create unique ID using filename and chunk content hash
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                chunk_id = f"{filename}_chunk{i+1}_{chunk_hash}"
                
                ids.append(chunk_id)
                docs.append(chunk)
                metadatas.append({"source": filename, "chunk_index": i})
                embeddings.append(self.get_local_embedding(chunk))

            if not ids:
                print(f"⚠️ No valid chunks created for {filename}")
                return

            # Add to collection in batches
            batch_size = 20
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=docs[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    embeddings=embeddings[i:end_idx]
                )
                
            print(f"✅ Successfully ingested {filename} with {len(chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error ingesting {filename}: {e}")

    def _split_text_smart(self, text, chunk_size=500, chunk_overlap=50):
        """Improved text splitting that respects sentence boundaries"""
        if not text.strip():
            return []
            
        # Split by sentences, paragraphs, or fixed size
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                overlap_words = current_chunk.split()[-10:]  # Last 10 words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += sentence + " "
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        # Filter out very short chunks (likely artifacts)
        chunks = [chunk for chunk in chunks if len(chunk) > 50]
        
        return chunks

    def detect_roman_urdu(self, text: str) -> bool:
        """Detect if the text contains Roman Urdu patterns"""
        text_lower = text.lower()
        total_words = len(text_lower.split())
        
        if total_words == 0:
            return False
            
        matches = 0
        for pattern in self.roman_urdu_patterns:
            if re.search(pattern, text_lower):
                matches += 1
                
        # If we find multiple Roman Urdu patterns, consider it Roman Urdu
        return matches >= 2

    def call_deepseek(self, prompt: str, system_instruction: str = None, temperature: float = 0.2) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "FastBot"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
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

    def extract_final_answer(self, text: str) -> str:
        """Extract only the final answer by removing thinking blocks"""
        cleaned = text
        
        # Remove thinking blocks
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<\|thinking\|>.*?<\|end\|>', '', cleaned, flags=re.DOTALL)
        
        # Remove process descriptions
        process_patterns = [
            r'^Based on (the|this) (context|information|documents?),',
            r'^According to (the|this) (context|information|documents?),',
            r'^Looking at (the|this) (context|information|documents?),',
        ]
        
        for pattern in process_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def debug_collection(self):
        """Debug function to check what's in the collection"""
        print("\n=== DEBUG COLLECTION ===")
        try:
            all_docs = self.collection.get()
            print(f"Total documents in collection: {len(all_docs['ids'])}")
            
            if all_docs['ids']:
                print("Sample documents:")
                for i, (doc_id, doc_content) in enumerate(zip(all_docs['ids'][:3], all_docs['documents'][:3])):
                    print(f"{i+1}. ID: {doc_id}")
                    print(f"   Content: {doc_content[:100]}...")
                    if all_docs['metadatas']:
                        print(f"   Metadata: {all_docs['metadatas'][i]}")
                    print()
            else:
                print("❌ Collection is empty!")
                
        except Exception as e:
            print(f"Debug error: {e}")

    def generate_answer(self, question: str) -> str:
        print(f"\n🔍 Processing question: '{question}'")
        
        # Debug collection first
        self.debug_collection()
        
        # Detect if question is in Roman Urdu
        is_roman_urdu = self.detect_roman_urdu(question)
        print(f"🌐 Roman Urdu detected: {is_roman_urdu}")
        
        # 1. Retrieve Context
        try:
            query_embedding = self.get_local_embedding(question)
            results = self.collection.query(
                query_embeddings=[query_embedding], 
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"📊 Retrieved {len(results['documents'][0]) if results['documents'] else 0} chunks")
            
            # Filter results by distance threshold (more lenient)
            relevant_chunks = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 0
                    print(f"   Chunk {i+1}: distance={distance:.3f}")
                    if distance < 1.5:  # More lenient threshold
                        relevant_chunks.append(doc)
            
            context = "\n\n".join(relevant_chunks) if relevant_chunks else ""
            print(f"🎯 Using {len(relevant_chunks)} relevant chunks")

            if not context.strip(): 
                print("❌ No relevant context found")
                if is_roman_urdu:
                    return "Maaf karein, mere paas university ke records mein is bare mein koi malumat nahi hai."
                return "I'm sorry, I don't have information about that in the university records."

            # 2. Enhanced System Instruction
            base_instruction = (
                "You are a helpful and friendly Student Advisor for FAST University. "
                "Follow this process internally:\n"
                "1. THINK: Analyze the context and question carefully in your mind\n"
                "2. PLAN: Determine the best way to answer based on available information\n" 
                "3. ANSWER: Provide only the final answer naturally\n\n"
                "STRICT OUTPUT RULES:\n"
                "- Do NOT show your thinking process in the final answer\n"
                "- Do NOT mention documents, context, or search process\n"
                "- Do NOT use phrases like 'Based on...', 'According to...'\n"
                "- Provide the answer as if you naturally know it\n"
                "- Be conversational and direct\n"
                "- If information is missing, politely say you don't know\n"
            )
            
            if is_roman_urdu:
                base_instruction += "RESPOND in Roman Urdu (Urdu written in English script) since the user asked in Roman Urdu."

            # 3. Enhanced prompt
            prompt = f"""
CONTEXT FROM UNIVERSITY RECORDS:
{context}

USER QUESTION: {question}

INTERNAL PROCESS (DO NOT SHOW IN ANSWER):
1. First, analyze the context thoroughly to understand what information is available
2. Then, determine if the context contains relevant information for the question
3. Finally, formulate a clear, direct answer using only the context information

FINAL ANSWER REQUIREMENTS:
- Provide ONLY the direct answer to the question
- Do NOT explain your thinking process or analysis steps
- Do NOT reference the context or documents
- Speak naturally as if you're having a conversation
- If the answer isn't in the context, politely decline without explanation
"""

            # 4. Call Model
            print("🤖 Calling DeepSeek API...")
            raw_answer = self.call_deepseek(prompt, base_instruction, temperature=0.3)
            
            # 5. Extract final answer
            final_answer = self.extract_final_answer(raw_answer)
            
            print(f"✅ Generated answer: {final_answer[:100]}...")
            return final_answer if final_answer else "I'm sorry, I don't have that information available."

        except Exception as e:
            print(f"❌ Error in generate_answer: {e}")
            if is_roman_urdu:
                return "Technical issue aa raha hai. Thori der baad try karein."
            return "I'm experiencing technical issues. Please try again later."

    def generate_chat_title(self, first_message: str) -> str:
        prompt = f"Summarize this query into a short 3-5 word title: '{first_message}'. No quotes, no thinking process."
        raw_title = self.call_deepseek(
            prompt, 
            system_instruction="Create very short titles. No explanations, just the title.",
            temperature=0.3
        )
        return self.extract_final_answer(raw_title).replace('"', '').replace("Title:", "").strip()

    def force_reingest_all(self):
        """Force re-ingest all documents (useful for debugging)"""
        print("🔄 Force re-ingesting all documents...")
        
        # Clear the collection
        try:
            self.collection.delete(where={})
            print("🗑️ Cleared existing collection")
        except:
            print("⚠️ Could not clear collection (might be empty)")
        
        # Re-ingest all files
        files_on_disk = [f for f in os.listdir(self.doc_path) if f.endswith(".txt")]
        for filename in files_on_disk:
            self.ingest_file(filename)
        
        print(f"✅ Force re-ingested {len(files_on_disk)} files")