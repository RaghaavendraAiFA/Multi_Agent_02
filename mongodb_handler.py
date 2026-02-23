# mongodb_handler.py
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient   # async MongoDB driver
from datetime import datetime
import uuid
import certifi

from config import MONGODB_URI, MONGODB_DB_NAME

sync_client = MongoClient(
    MONGODB_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    tlsAllowInvalidCertificates=True,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=20000,
    socketTimeoutMS=20000,
)
sync_db         = sync_client[MONGODB_DB_NAME]
sync_collection = sync_db["chat_history"]


async_client = AsyncIOMotorClient(
    MONGODB_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    tlsAllowInvalidCertificates=True,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=20000,
    socketTimeoutMS=20000,
)
async_db         = async_client[MONGODB_DB_NAME]
async_collection = async_db["chat_history"]

print(f"MongoDB connected: {MONGODB_DB_NAME}")


# ─────────────────────────────────────────────────────
#  GENERATE SESSION ID
#  Creates a unique 8-character ID for each chat session
#  Called once when a new session starts in Fast_api.py
# ─────────────────────────────────────────────────────
def generate_session_id() -> str:
    return str(uuid.uuid4())[:8]


# ═════════════════════════════════════════════════════
#  SYNC FUNCTIONS
#  Used by: orchestrator.py → synthesizer_node()
#  Reason : LangGraph nodes run in sync context
# ═════════════════════════════════════════════════════

def save_conversation(session_id: str, question: str, answer: str, agents_used: list):
    """
    Save one Q&A pair to MongoDB (sync).
    Called by synthesizer_node in orchestrator.py after every answer.
    """
    document = {
        "session_id"  : session_id,
        "question"    : question,
        "answer"      : answer,
        "agents_used" : agents_used,
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    sync_collection.insert_one(document)
    print(f"  Saved to MongoDB (session: {session_id})")


def get_session_history(session_id: str) -> list:
    """
    Fetch all Q&A pairs for a session (sync).
    Returns records sorted oldest → newest.
    """
    records = sync_collection.find(
        {"session_id": session_id},
        {"_id": 0}                   # exclude MongoDB internal _id field
    ).sort("timestamp", 1)
    return list(records)


def get_all_conversations() -> list:
    """
    Fetch all conversations across all sessions (sync).
    Returns records sorted newest → oldest.
    """
    records = sync_collection.find(
        {},
        {"_id": 0}
    ).sort("timestamp", -1)
    return list(records)


# ═════════════════════════════════════════════════════
#  ASYNC FUNCTIONS
#  Used by: Fast_api.py endpoints
#  Reason : FastAPI is async — motor async driver fits perfectly
#           No need for run_in_threadpool when using these
# ═════════════════════════════════════════════════════

async def async_get_session_history(session_id: str) -> list:
    """
    Fetch all Q&A pairs for a session (async).
    Used directly in FastAPI GET /history endpoint.
    """
    cursor  = async_collection.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1)
    records = await cursor.to_list(length=None)
    return records


async def async_get_all_conversations() -> list:
    cursor  = async_collection.find(
        {},
        {"_id": 0}
    ).sort("timestamp", -1)
    records = await cursor.to_list(length=None)
    return records


# ─────────────────────────────────────────────────────
#  TEST — python mongodb_handler.py
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MongoDB connection...")

    try:
        sync_client.admin.command("ping")
        print("MongoDB ping successful!")
    except Exception as e:
        print(f"Connection failed: {e}")
        exit(1)

    # Test sync save and fetch
    test_session = generate_session_id()
    print(f"Session ID: {test_session}")

    save_conversation(
        session_id  = test_session,
        question    = "Test question",
        answer      = "Test answer",
        agents_used = ["ipc"],
    )

    history = get_session_history(test_session)
    print(f"Fetched {len(history)} record(s) — MongoDB working correctly!")