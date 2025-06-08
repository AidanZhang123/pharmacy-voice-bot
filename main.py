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

# ─── CORS (allow React dev server + your deployed dashboard) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                   # allow all origins for simplicity
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ─── Debug startup: verify env vars ─────────────────────────────────────────────
print(
    "Startup: ELEVENLABS_API_KEY set?", os.getenv("ELEVENLABS_API_KEY") is not None,
    "VOICE_ID:", os.getenv("ELEVENLABS_VOICE_ID")
)

# ─── Load API keys & config ─────────────────────────────────────────────────────
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # conversations
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            call_sid        TEXT PRIMARY KEY,
            messages        TEXT,
            reprompt_count  INTEGER DEFAULT 0
        );
    """)
    # call_logs
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
    # bookings
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
    # add reprompt_count if missing
    c.execute("PRAGMA table_info(conversations);")
    cols = [r[1] for r in c.fetchall()]
    if "reprompt_count" not in cols:
        c.execute("ALTER TABLE conversations ADD COLUMN reprompt_count INTEGER DEFAULT 0;")
        conn.commit()
    conn.close()
    print(f"[{datetime.utcnow()}] init_db complete, DB at {os.path.abspath(DB_PATH)}")

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
    return retry_sqlite(_get)

def save_history(call_sid: str, messages: list):
    serialized = json.dumps(messages)
    def _save():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE conversations SET messages = ? WHERE call_sid = ?;",
            (serialized, call_sid)
        )
        conn.commit()
        conn.close()
    retry_sqlite(_save)

def get_reprompt_count(call_sid: str) -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT reprompt_count FROM conversations WHERE call_sid = ?;", (call_sid,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else 0
    except:
        return 0

def increment_reprompt_count(call_sid: str):
    def _inc():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE conversations
            SET reprompt_count = reprompt_count + 1
            WHERE call_sid = ?;
        """, (call_sid,))
        conn.commit()
        conn.close()
    retry_sqlite(_inc)

def reset_reprompt_count(call_sid: str):
    def _reset():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE conversations
            SET reprompt_count = 0
            WHERE call_sid = ?;
        """, (call_sid,))
        conn.commit()
        conn.close()
    retry_sqlite(_reset)

def log_call_turn(call_sid: str, turn_number: int,
                  user_text: str=None,
                  assistant_reply: str=None,
                  error_message: str=None):
    def _log():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO call_logs(
                call_sid, turn_number, user_text, assistant_reply, error_message
            ) VALUES (?, ?, ?, ?, ?);
        """, (call_sid, turn_number, user_text, assistant_reply, error_message))
        conn.commit()
        conn.close()
    retry_sqlite(_log)

def save_booking(call_sid: str, vaccine_type: str, patient_name: str, desired_date: str):
    def _insert():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO bookings (call_sid, vaccine_type, patient_name, desired_date)
            VALUES (?, ?, ?, ?);
        """, (call_sid, vaccine_type, patient_name, desired_date))
        conn.commit()
        conn.close()
    retry_sqlite(_insert)

def classify_intent(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in ["vaccine", "vaccination", "shot"]):
        return "VACCINE"
    if any(kw in lower for kw in ["refill", "renew", "prescription"]):
        return "REFILL"
    if any(kw in lower for kw in ["hour", "open", "close", "time"]):
        return "HOURS"
    if "pharmacy" in lower:
        return "NEAREST"
    return "GENERAL"

def generate_and_play_tts(text: str, call_sid: str, suffix: str="resp") -> VoiceResponse:
    tts_filename = f"tts_{call_sid}_{suffix}_{int(time.time())}.mp3"
    tts_filepath = os.path.join("static", tts_filename)

    # Debug env
    print("🔑 ELEVENLABS_API_KEY set?", ELEVENLABS_API_KEY is not None,
          "VOICE_ID:", VOICE_ID)

    try:
        tts_endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",   # try changing this if you still get 400
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        resp = requests.post(tts_endpoint, json=payload, headers=headers, timeout=10)
        if resp.status_code != 200 or not resp.content:
            # log full error for diagnosis
            print(f"[{datetime.utcnow()}] ElevenLabs TTS error status {resp.status_code}, body: {resp.text}")
            raise Exception(f"TTS bad status {resp.status_code}")
        with open(tts_filepath, "wb") as f:
            f.write(resp.content)

        # success → play the MP3
        vr = VoiceResponse()
        gather = vr.gather(
            input="speech",
            action=f"{BASE_URL}/process-recording",
            method="POST",
            speechTimeout="auto"
        )
        gather.play(f"{BASE_URL}/static/{tts_filename}")
        return vr

    except Exception as e:
        # fallback: still wrap in a Gather so the call doesn't just drop
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

# ─── Incoming Call ──────────────────────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    print("📥 /incoming-call:", dict(form))
    call_sid = form.get("CallSid")
    if not call_sid:
        return Response(status_code=400)

    reset_reprompt_count(call_sid)
    greeting = "Hello, thank you for calling the pharmacy. How can I help you today?"
    vr = generate_and_play_tts(greeting, call_sid, suffix="greeting")
    tw = str(vr)
    print("📤 /incoming-call TwiML:", tw)
    return Response(content=tw, media_type="application/xml")

# ─── Process Recording ────────────────────────────────────────────────────────────
@app.post("/process-recording")
async def process_recording(request: Request):
    form = await request.form()
    print("📥 /process-recording:", dict(form))

    call_sid    = form.get("CallSid")
    user_speech = form.get("SpeechResult") or ""
    conf_str    = form.get("Confidence")
    confidence  = float(conf_str) if conf_str and conf_str.replace('.','',1).isdigit() else 0.0

    if not call_sid:
        return Response(status_code=400)

    history   = get_history(call_sid)
    reprompts = get_reprompt_count(call_sid)

    # A) Silence
    if not user_speech.strip():
        if reprompts < 3:
            increment_reprompt_count(call_sid)
            text = "I’m sorry, I didn’t hear anything. Could you please repeat?"
        else:
            text = "We did not receive any input. Goodbye."
        vr = VoiceResponse()
        if reprompts < 3:
            g = vr.gather(input="speech",action=f"{BASE_URL}/process-recording",method="POST",speechTimeout="auto")
            g.say(text)
            log_call_turn(call_sid, len(history)//2, None, None, "Silence reprompt")
        else:
            vr.say(text); vr.hangup()
            log_call_turn(call_sid, len(history)//2, None, None, "Silence hangup")
        tw = str(vr); print("📤 /process-recording TwiML:", tw)
        return Response(content=tw, media_type="application/xml")

    # B) Low confidence
    if confidence < 0.5:
        text = "I’m sorry, I didn’t catch that clearly. Could you please repeat?"
        vr = VoiceResponse()
        g = vr.gather(input="speech",action=f"{BASE_URL}/process-recording",method="POST",speechTimeout="auto")
        g.say(text)
        log_call_turn(call_sid, len(history)//2, user_speech, None, f"Low confidence ({confidence})")
        tw = str(vr); print("📤 /process-recording TwiML:", tw)
        return Response(content=tw, media_type="application/xml")

    user_text = user_speech.strip()
    reset_reprompt_count(call_sid)
    history.append({"role":"user","content":user_text})

    # Vaccine subflow
    last = history[-2]["content"] if len(history)>1 else ""
    if last.startswith("Which vaccine would you like"):
        # got vaccine
        history.append({"role":"system","content": json.dumps({"vaccine_type":user_text})})
        reply = "Got it. May I have your full name, please?"
    elif last.startswith("Got it. May I have your full name"):
        history.append({"role":"system","content": json.dumps({"patient_name":user_text})})
        reply = "Thank you. On which date would you like to book your appointment? Please say the date."
    elif last.startswith("Thank you. On which date"):
        # final slot
        history.append({"role":"system","content": json.dumps({"desired_date":user_text})})
        # pull slots
        slots = {}
        for m in history:
            if m["role"]=="system":
                slots.update(json.loads(m["content"]))
        save_booking(call_sid, slots["vaccine_type"], slots["patient_name"], slots["desired_date"])
        reply = (f"Thank you. Your {slots['vaccine_type']} appointment for "
                 f"{slots['patient_name']} on {slots['desired_date']} is booked. Goodbye.")
        vr = generate_and_play_tts(reply, call_sid, suffix="finalv"); vr.hangup()
        log_call_turn(call_sid, len(history)//2, user_text, reply, "VACCINE_BOOKED")
        history.append({"role":"assistant","content":reply})
        save_history(call_sid, history)
        tw = str(vr); print("📤 /process-recording TwiML:", tw)
        return Response(content=tw, media_type="application/xml")
    else:
        # new intent
        intent = classify_intent(user_text)
        if intent=="VACCINE":
            reply = "Which vaccine would you like? Please say the vaccine name."
        # ... other intents unchanged ...
        else:
            reply = "Sorry, I didn’t understand that. How can I help?"
    # append & log
    history.append({"role":"assistant","content":reply})
    save_history(call_sid, history)
    log_call_turn(call_sid, len(history)//2, user_text, reply, None)

    # respond
    vr = generate_and_play_tts(reply, call_sid, suffix=str(int(time.time())))
    tw = str(vr); print("📤 /process-recording TwiML:", tw)
    return Response(content=tw, media_type="application/xml")

# ─── Enhanced /api/logs ───────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_call_logs(limit: int = 100):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, call_sid, turn_number, user_text,
                   assistant_reply, error_message, timestamp
            FROM call_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = c.fetchall()
        logs = []
        for r in rows:
            log = dict(zip(
                ["id","call_sid","turn_number","user_text","assistant_reply","error_message","timestamp"],
                r
            ))
            # attach full transcript
            c.execute("SELECT messages FROM conversations WHERE call_sid = ?;", (log["call_sid"],))
            mrow = c.fetchone()
            log["transcript"] = json.loads(mrow[0]) if mrow else []
            # attach booking if any
            c.execute("""
                SELECT vaccine_type,patient_name,desired_date,booked_at 
                FROM bookings WHERE call_sid = ?;
            """, (log["call_sid"],))
            brow = c.fetchone()
            log["booking"] = dict(zip(
                ["vaccine_type","patient_name","desired_date","booked_at"], brow
            )) if brow else None
            logs.append(log)
        conn.close()
        return JSONResponse({"logs": logs})
    except Exception as e:
        print("Error in /api/logs:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/calls")
async def list_call_sids():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT call_sid FROM conversations;")
    call_sids = [r[0] for r in c.fetchall()]
    conn.close()
    return JSONResponse({"call_sids": call_sids})

@app.get("/api/conversations/{call_sid}")
async def get_conversation(call_sid: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT messages FROM conversations WHERE call_sid = ?;", (call_sid,))
    row = c.fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "CallSid not found"}, status_code=404)
    return JSONResponse({"call_sid": call_sid, "messages": json.loads(row[0])})
