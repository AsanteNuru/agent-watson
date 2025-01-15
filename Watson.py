from __future__ import annotations
import asyncio
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from textblob import TextBlob
from telethon import TelegramClient, events
import logfire
import httpx
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
security = HTTPBearer()

# Supabase setup
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class PersonalityConfig(BaseModel):
    name: str
    traits: List[str]
    speaking_style: str
    background: Optional[str] = None

class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str
    personality: Optional[PersonalityConfig] = None

class AgentResponse(BaseModel):
    success: bool

class TelegramMessage(BaseModel):
    chat_id: int
    message_id: int
    text: str
    timestamp: datetime

    class Config:
        from_attributes = True

class WatsonDeps(BaseModel):
    client: Any
    model: OpenAIModel
    watson_token: str = Field(default_factory=lambda: os.getenv("API_BEARER_TOKEN"))
    
    class Config:23
    arbitrary_types_allowed = True

system_prompt = """
You are Watson, a helpful AI assistant with access to Telegram and Supabase. Your personality traits include:
- Helpful, knowledgeable, witty, classy, humorous
- Well-educated, intelligent, loveable, supportive
- Loyal, patient

You speak in a professional, concise, warm and modest manner.

Your primary responsibilities are:
1. Analyzing and responding to Telegram messages
2. Maintaining conversation history in Supabase
3. Providing helpful and accurate information
"""

llm = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
model = OpenAIModel(
    llm,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

watson_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=WatsonDeps,
    retries=2
)

@watson_agent.tool
async def analyze_telegram_message(ctx: RunContext[WatsonDeps], message: TelegramMessage) -> str:
    """Analyze and process incoming Telegram messages"""
    try:
        print(f"Received message from {message.chat_id}: {message.text}")
        
        analysis = TextBlob(message.text)
        print(f"Sentiment: {analysis.sentiment}")
        
        await store_message(
            session_id=f"telegram_{message.chat_id}",
            message_type="telegram",
            content=message.text,
            data={
                "chat_id": message.chat_id,
                "message_id": message.message_id,
                "analysis": {
                    "textblob": analysis.sentiment._asdict()
                }
            }
        )
        return "Message processed successfully"
    except Exception as e:
        print(f"Error processing Telegram message: {str(e)}")
        return f"Error: {str(e)}"

@watson_agent.tool
async def process_query(ctx: RunContext[WatsonDeps], query: str, message_history: List[Dict[str, Any]]) -> str:
    """Process user queries with conversation context"""
    try:
        messages = []
        for msg in message_history:
            if isinstance(msg, dict):
                msg_data = msg.get("message", {})
                msg_type = msg_data.get("type", "")
                msg_content = msg_data.get("content", "")
                if msg_type == "human":
                    messages.append(ModelRequest(parts=[UserPromptPart(content=msg_content)]))
                else:
                    messages.append(ModelResponse(parts=[TextPart(content=msg_content)]))
            else:
                messages.append(msg)

        # Add current query
        messages.append(ModelRequest(parts=[UserPromptPart(content=query)]))
        
        # Generate response using the model
        response = await ctx.deps.model.generate(messages=messages)
        
        # Return the response content
        if response and response.parts:
            return response.parts[0].content
        return "I'm sorry, I couldn't generate a response."
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return ModelResponse(parts=[TextPart(content="I apologize, but I encountered an error processing your request.")])

async def store_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the Supabase messages table."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj["data"] = data

    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")

class TelegramListener:
    def __init__(self):
        self.client = None
        self.running = False
        self.allowed_channels = self._get_allowed_channels()
        
    def _get_allowed_channels(self) -> List[str]:
        """Get list of allowed Telegram channels from environment"""
        channels = os.getenv("TELEGRAM_ALLOWED_CHANNELS", "")
        return [c.strip() for c in channels.split(",") if c.strip()]
        
    async def start(self):
        """Start listening for Telegram messages"""
        api_id = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        
        if not api_id or not api_hash:
            print("Telegram credentials not found in environment variables")
            return False
            
        if not self.allowed_channels:
            print("No allowed Telegram channels configured")
            return False
            
        try:
            self.client = TelegramClient('watson_session', api_id, api_hash)
            self.client.on(events.NewMessage(chats=self.allowed_channels))(self.message_handler)
            await self.client.start()
            self.running = True
            print(f"Telegram listener started for channels: {', '.join(self.allowed_channels)}")
            return True
        except Exception as e:
            print(f"Error starting Telegram client: {str(e)}")
            return False
            
    async def message_handler(self, event):
        """Store incoming Telegram messages"""
        message = event.message
        await store_message(
            session_id=f"telegram_{message.chat_id}",
            message_type="telegram",
            content=message.text,
            data={
                "chat_id": message.chat_id,
                "message_id": message.id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@watson_agent.tool
async def get_recent_telegram_messages(ctx: RunContext[WatsonDeps], chat_id: int, limit: int = 10) -> List[Dict]:
    """Get recent Telegram messages from storage"""
    return await fetch_conversation_history(f"telegram_{chat_id}", limit)

@watson_agent.tool
async def analyze_stored_telegram_message(ctx: RunContext[WatsonDeps], message_id: str) -> str:
    """Analyze a stored Telegram message"""
    try:
        result = supabase.table("messages") \
            .select("*") \
            .eq("data->>message_id", message_id) \
            .single() \
            .execute()
        message = result.data['message']
        
        # Process the message content
        return await process_query(
            ctx,
            query=message['content'],
            message_history=[]
        )
    except Exception as e:
        return f"Error analyzing message: {str(e)}"

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True

async def fetch_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return response.data[::-1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    async def run_all():
        telegram_listener = TelegramListener()
        telegram_initialized = await telegram_listener.start()
        
        tasks = [uvicorn.run(app, host="0.0.0.0", port=8001)]
        if telegram_initialized:
            tasks.append(telegram_listener.client.run_until_disconnected())
        
        await asyncio.gather(*tasks)
    
    asyncio.run(run_all())
