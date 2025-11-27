import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from dotenv import load_dotenv
import bcrypt

from RAGService import RAGService

# --- CONFIGURATION ---
load_dotenv()
MONGO_URL = os.getenv("MONGO_URL") 
SECRET_KEY = os.getenv("SECRET_KEY", "your_super_secret_key_change_this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="FastBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncIOMotorClient(MONGO_URL)
db = client.fastbot_db
users_collection = db.users
chats_collection = db.chats

try:
    rag_engine = RAGService()
except Exception as e:
    print(f"⚠️ Warning: RAG Engine failed to initialize: {e}")
    rag_engine = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- AUTH HELPERS ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)
    
    user = await users_collection.find_one({"email": username})
    if user is None: raise HTTPException(status_code=401)
    return user

# --- MODELS ---
class UserSignup(BaseModel):
    fullName: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatSession(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]

class NewMessageRequest(BaseModel):
    chatId: str
    content: str

# --- ROUTES ---

@app.post("/auth/signup")
async def signup(user: UserSignup):
    if await users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = {"fullName": user.fullName, "email": user.email, "password": hashed_password}
    await users_collection.insert_one(new_user)
    return {"message": "User created successfully"}

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user["email"]})
    return {"access_token": access_token, "token_type": "bearer", "username": user["fullName"]}

@app.post("/auth/login")
async def json_login(user_data: UserLogin):
    user = await users_collection.find_one({"email": user_data.email})
    if not user or not verify_password(user_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token(data={"sub": user["email"]}), "username": user["fullName"]}

@app.get("/chats", response_model=List[ChatSession])
async def get_chats(current_user: dict = Depends(get_current_user)):
    cursor = chats_collection.find({"userId": current_user["email"]}).sort("updated_at", -1)
    chats = []
    async for chat in cursor:
        chats.append(ChatSession(id=str(chat["chatId"]), title=chat["title"], messages=chat["messages"]))
    return chats

@app.post("/chats/new")
async def create_chat(current_user: dict = Depends(get_current_user)):
    chat_id = str(int(datetime.utcnow().timestamp() * 1000))
    new_chat = {
        "userId": current_user["email"],
        "chatId": chat_id,
        "title": "New Chat",
        "messages": [{"role": "bot", "content": "Hello! I am Fast Bot. How can I help you today?"}],
        "updated_at": datetime.utcnow()
    }
    await chats_collection.insert_one(new_chat)
    return {"id": chat_id, "title": new_chat["title"], "messages": new_chat["messages"]}

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    result = await chats_collection.delete_one({"chatId": chat_id, "userId": current_user["email"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"message": "Chat deleted"}

@app.post("/chats/message")
async def send_message(request: NewMessageRequest, current_user: dict = Depends(get_current_user)):
    chat = await chats_collection.find_one({"chatId": request.chatId, "userId": current_user["email"]})
    if not chat: raise HTTPException(status_code=404, detail="Chat not found")

    user_msg = {"role": "user", "content": request.content}
    update_data = {"$push": {"messages": user_msg}, "$set": {"updated_at": datetime.utcnow()}}
    new_title = None

    # Generate Smart Title if needed
    if chat["title"] == "New Chat" and rag_engine:
        try:
            new_title = rag_engine.generate_chat_title(request.content)
            update_data["$set"]["title"] = new_title
        except Exception as e:
            print(f"Title Error: {e}")

    await chats_collection.update_one({"chatId": request.chatId}, update_data)

    # Generate Bot Response
    bot_answer = rag_engine.generate_answer(request.content) if rag_engine else "RAG unavailable."
    bot_msg = {"role": "bot", "content": bot_answer}

    await chats_collection.update_one({"chatId": request.chatId}, {"$push": {"messages": bot_msg}})

    return {"user_message": user_msg, "bot_message": bot_msg, "chat_title": new_title}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("MainService:app", host="0.0.0.0", port=8000, reload=True)