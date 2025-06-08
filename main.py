import os
import time
import json
import sqlite3
import requests
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Debug startup: verify environment variables are loaded
print(
    "Startup: ELEVENLABS_API_KEY set?", os.getenv("ELEVENLABS_API_KEY") is not None,
    "VOICE_ID:", os.getenv("ELEVENLABS_VOICE_ID")
)

# ─── CORS ────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─── Load API keys & config ──────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY   = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID             = os.getenv("ELEVENLABS_VOICE_ID")
BASE_URL             = os.getenv("BASE_URL")
GOOGLE_MAPS_API_KEY  = os.getenv("GOOGLE_MAPS_API_KEY")

# ─── Initialize OpenAI client ───────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Static files & DB setup ─────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
DB_PATH = "conversations.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                call_sid        TEXT PRIMARY KEY,
                messages        TEXT,
                reprompt_count  INTEGER DEFAULT 0
            );
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS call_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                call_sid        TEXT,
                turn_number     INTEGER,
                user_text       TEXT,
                assistant_reply TEXT,
                error_message   TEXT,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                call_sid        TEXT,
                vaccine_type    TEXT,
                patient_name    TEXT,
                desired_date    TEXT,
                booked_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        c.execute("PRAGMA table_info(conversations);")
        cols = [r[1] for r in c.fetchall()]
        if "reprompt_count" not in cols:
            c.execute("ALTER TABLE conversations ADD COLUMN reprompt_count INTEGER DEFAULT 0;")
            conn.commit()
        conn.close()
        print(f"[{datetime.utcnow()}] init_db complete, DB at {os.path.abspath(DB_PATH)}")
    except Exception as e:
        print(f"[{datetime.utcnow()}] init_db error: {e}")

init_db()

def retry_sqlite(func, *args, retries=3, delay=0.1, **kwargs):
    for _ in range(retries):
        try:
            return func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                time.sleep(delay)
            else:
                raise
    return func(*args, **kwargs)

def get_history(call_sid: str):
    def _get():
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT messages FROM conversations WHERE call_sid = ?;", (call_sid,))
            row = c.fetchone()
            if row:
                msgs = json.loads(row[0])
            else:
                msgs = []
                c.execute(
                    "INSERT INTO conversations(call_sid, messages, reprompt_count) VALUES (?, ?, 0);",
                    (call_sid, json.dumps(msgs))
                )
                conn.commit()
            conn.close()
            return msgs
        except Exception as e:
            print(f"[{datetime.utcnow()}] get_history error: {e}")
            return []
    return retry_sqlite(_get)

def save_history(call_sid: str, messages: list):
    serialized = json.dumps(messages)
    def _save():
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "UPDATE conversations SET messages = ? WHERE call_sid = ?;",
                (serialized, call_sid)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[{datetime.utcnow()}] save_history error: {e}")
    retry_sqlite(_save)

def get_reprompt_count(call_sid: str) -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT reprompt_count FROM conversations WHERE call_sid = ?;", (call_sid,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception as e:
        print(f"[{datetime.utcnow()}] get_reprompt_count error: {e}")
        return 0

def increment_reprompt_count(call_sid: str):
    def _inc():
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                UPDATE conversations
                SET reprompt_count = reprompt_count + 1
                WHERE call_sid = ?;
            """, (call_sid,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[{datetime.utcnow()}] increment_reprompt_count error: {e}")
    retry_sqlite(_inc)

def reset_reprompt_count(call_sid: str):
    def _reset():
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                UPDATE conversations
                SET reprompt_count = 0
                WHERE call_sid = ?;
            """, (call_sid,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[{datetime.utcnow()}] reset_reprompt_count error: {e}")
    retry_sqlite(_reset)

def log_call_turn(call_sid: str, turn_number: int,
                  user_text: str=None,
                  assistant_reply: str=None,
                  error_message: str=None):
    def _log():
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                INSERT INTO call_logs(
                    call_sid, turn_number, user_text, assistant_reply, error_message
                ) VALUES (?, ?, ?, ?, ?);
            """, (call_sid, turn_number, user_text, assistant_reply, error_message))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[{datetime.utcnow()}] log_call_turn error: {e}")
    retry_sqlite(_log)

def save_booking(call_sid: str, vaccine_type: str, patient_name: str, desired_date: str):
    def _insert():
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                INSERT INTO bookings (call_sid, vaccine_type, patient_name, desired_date)
                VALUES (?, ?, ?, ?);
            """, (call_sid, vaccine_type, patient_name, desired_date))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[{datetime.utcnow()}] save_booking error: {e}")
    retry_sqlite(_insert)

def classify_intent(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in ["vaccine", "shot", "vaccination"]):
        return "VACCINE"
    if any(kw in lower for kw in ["refill", "renew", "prescription"]):
        return "REFILL"
    if any(kw in lower for kw in ["hour", "open", "close", "time"]):
        return "HOURS"
    if any(kw in lower for kw in ["nearest pharmacy", "pharmacy near me"]):
        return "NEAREST"
    return "GENERAL"

def generate_and_play_tts(text: str, call_sid: str, suffix: str="resp") -> VoiceResponse:
    tts_filename = f"tts_{call_sid}_{suffix}_{int(time.time())}.mp3"
    tts_filepath = os.path.join("static", tts_filename)

    # Debug env
    print(
        "ELEVENLABS_API_KEY set?", ELEVENLABS_API_KEY is not None,
        "VOICE_ID:", VOICE_ID
    )

    try:
        tts_endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_v2_flash",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        tts_resp = requests.post(tts_endpoint, json=payload, headers=headers, timeout=10)
        if tts_resp.status_code == 200 and tts_resp.content:
            with open(tts_filepath, "wb") as f:
                f.write(tts_resp.content)
        else:
            raise Exception(f"TTS error status {tts_resp.status_code}")
    except Exception as e:
        print(f"[{datetime.utcnow()}] ElevenLabs TTS error: {e}")
        vr = VoiceResponse()
        gather = vr.gather(
            input="speech",
            action=f"{BASE_URL}/process-recording",
            method="POST",
            speechTimeout="auto"
        )
        gather.say(text)
        return vr

    vr = VoiceResponse()
    gather = vr.gather(
        input="speech",
        action=f"{BASE_URL}/process-recording",
        method="POST",
        speechTimeout="auto"
    )
    gather.play(f"{BASE_URL}/static/{tts_filename}")
    return vr

# ─── Incoming Call ──────────────────────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    print("📥 /incoming-call form data:", dict(form))

    call_sid = form.get("CallSid")
    if not call_sid:
        return Response(status_code=400)

    reset_reprompt_count(call_sid)
    greeting_text = "Hello, thank you for calling the pharmacy. How can I help you today?"
    vr = generate_and_play_tts(greeting_text, call_sid, suffix="greeting")

    twiml_str = str(vr)
    print("📤 /incoming-call TwiML:", twiml_str)
    return Response(content=twiml_str, media_type="application/xml")

# ─── Process Recording ────────────────────────────────────────────────────────────
@app.post("/process-recording")
async def process_recording(request: Request):
    form = await request.form()
    print("📥 /process-recording form data:", dict(form))

    call_sid       = form.get("CallSid")
    user_speech    = form.get("SpeechResult")
    confidence_str = form.get("Confidence")

    if not call_sid:
        return Response(status_code=400)

    try:
        confidence = float(confidence_str) if confidence_str else 0.0
    except ValueError:
        confidence = 0.0

    history   = get_history(call_sid)
    reprompts = get_reprompt_count(call_sid)

    # A) Silence
    if not user_speech or not user_speech.strip():
        if reprompts < 3:
            increment_reprompt_count(call_sid)
            text = "I’m sorry, I didn’t hear anything. Could you please repeat?"
            vr = VoiceResponse()
            gather = vr.gather(input="speech", action=f"{BASE_URL}/process-recording", method="POST", speechTimeout="auto")
            gather.say(text)
            log_call_turn(call_sid, 0, None, None, "Silence reprompt")
            tw = str(vr); print("📤 /process-recording TwiML:", tw)
            return Response(content=tw, media_type="application/xml")
        else:
            vr = VoiceResponse(); vr.say("We did not receive any input. Goodbye."); vr.hangup()
            log_call_turn(call_sid, 0, None, None, "Silence hangup")
            tw = str(vr); print("📤 /process-recording TwiML:", tw)
            return Response(content=tw, media_type="application/xml")

    # B) Low confidence
    if confidence < 0.5:
        text = "I’m sorry, I didn’t catch that clearly. Could you please repeat?"
        vr = VoiceResponse()
        gather = vr.gather(input="speech", action=f"{BASE_URL}/process-recording", method="POST", speechTimeout="auto")
        gather.say(text)
        log_call_turn(call_sid, 0, user_speech, None, f"Low confidence ({confidence})")
        tw = str(vr); print("📤 /process-recording TwiML:", tw)
        return Response(content=tw, media_type="application/xml")

    # C) Valid speech
    user_text = user_speech.strip()
    reset_reprompt_count(call_sid)
    print(f"[{datetime.utcnow()}] User said: {user_text!r} (conf={confidence})")

    # Corrections, Vaccine subflow, Nearest subflow… (unchanged from your prior code)
    # …
    # AFTER all those branches, we fall through to the “new intent detection” branch,
    # append assistant_reply, save to history and logs, then generate TTS:

    # F) New intent detection & few-shot fallback
    intent = classify_intent(user_text)
    if intent == "VACCINE":
        assistant_reply = "Which vaccine would you like? Please say the vaccine name."
    elif intent == "REFILL":
        assistant_reply = "Certainly. What is your prescription number?"
    elif intent == "HOURS":
        assistant_reply = ("Our pharmacy is open Monday to Friday, 9 AM to 6 PM, "
                          "and Saturday 10 AM to 4 PM. Anything else I can help you with?")
    elif intent == "NEAREST":
        assistant_reply = "Sure—what’s your postal code (Canada)?"
    else:
        few_shot = [
            {"role":"system","content":"You are a friendly, concise pharmacy assistant. Keep replies under 300 characters."},
            {"role":"user","content":"I want to schedule a vaccine appointment."},
            {"role":"assistant","content":"Which vaccine would you like? Please say the vaccine name."},
            {"role":"user","content":"I need to refill my prescription."},
            {"role":"assistant","content":"Of course. What is your prescription number?"},
            {"role":"user","content":"What are your pharmacy hours on Saturday?"},
            {"role":"assistant","content":"We’re open Monday–Friday 9–6, Saturday 10–4. Anything else?"},
            {"role":"user","content":"Do you have ibuprofen 200 mg in stock?"},
            {"role":"assistant","content":"Let me check… Yes, we have it. Would you like me to hold some?"}
        ]
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=few_shot + history,
            max_tokens=60, temperature=0.2
        )
        assistant_reply = resp.choices[0].message.content.strip()

    # H) Append & log
    history.append({"role":"assistant","content":assistant_reply})
    save_history(call_sid, history)
    log_call_turn(call_sid, len(history)//2 + 1, user_text, assistant_reply, None)

    # I) Generate TTS, log TwiML, return
    vr = generate_and_play_tts(assistant_reply, call_sid, suffix=str(int(time.time())))
    tw = str(vr)
    print("📤 /process-recording TwiML:", tw)
    return Response(content=tw, media_type="application/xml")

# ─── Dashboard endpoints unchanged ────────────────────────────────────────────────
@app.get("/api/logs")
async def get_call_logs(limit: int = 100):
    # … same as before …
    return JSONResponse({"logs": logs})

@app.get("/api/calls")
async def list_call_sids():
    # … same as before …
    return JSONResponse({"call_sids": call_sids})

@app.get("/api/conversations/{call_sid}")
async def get_conversation(call_sid: str):
    # … same as before …
    return JSONResponse({"call_sid": call_sid, "messages": messages})
