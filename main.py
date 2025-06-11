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

# ─── Setup ───────────────────────────────────────────────────────────────────────
load_dotenv()
app = FastAPI()

# Allow dashboard & any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

print(
    "Startup: ELEVENLABS_API_KEY set?", os.getenv("ELEVENLABS_API_KEY") is not None,
    "VOICE_ID:", os.getenv("ELEVENLABS_VOICE_ID")
)

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY   = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID             = os.getenv("ELEVENLABS_VOICE_ID")
BASE_URL             = os.getenv("BASE_URL")
GOOGLE_MAPS_API_KEY  = os.getenv("GOOGLE_MAPS_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Serve static TTS MP3s
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_PATH = "conversations.db"

# ─── Database Initialization ────────────────────────────────────────────────────
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
    # call_metadata
    c.execute("""
        CREATE TABLE IF NOT EXISTS call_metadata (
            call_sid     TEXT PRIMARY KEY,
            from_number  TEXT,
            from_city    TEXT,
            from_state   TEXT,
            from_zip     TEXT,
            from_country TEXT
        );
    """)
    # add reprompt_count if missing
    c.execute("PRAGMA table_info(conversations);")
    cols = [r[1] for r in c.fetchall()]
    if "reprompt_count" not in cols:
        c.execute("ALTER TABLE conversations ADD COLUMN reprompt_count INTEGER DEFAULT 0;")
    conn.commit()
    conn.close()
    print(f"[{datetime.utcnow()}] init_db complete, DB at {os.path.abspath(DB_PATH)}")

init_db()

# ─── SQLite Helpers ─────────────────────────────────────────────────────────────
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

# ─── Intent Classification ───────────────────────────────────────────────────────
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

# ─── ElevenLabs TTS Helper ───────────────────────────────────────────────────────
def generate_and_play_tts(text: str, call_sid: str, suffix: str="resp") -> VoiceResponse:
    tts_filename = f"tts_{call_sid}_{suffix}_{int(time.time())}.mp3"
    tts_filepath = os.path.join("static", tts_filename)

    print("🔑 ELEVENLABS_API_KEY set?", ELEVENLABS_API_KEY is not None,
          "VOICE_ID:", VOICE_ID)

    try:
        endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        if resp.status_code != 200 or not resp.content:
            print(f"[{datetime.utcnow()}] ElevenLabs TTS error status {resp.status_code}, body: {resp.text}")
            raise Exception("TTS failed")
        with open(tts_filepath, "wb") as f:
            f.write(resp.content)

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
        print(f"[{datetime.utcnow()}] ElevenLabs TTS exception: {e}")
        vr = VoiceResponse()
        gather = vr.gather(
            input="speech",
            action=f"{BASE_URL}/process-recording",
            method="POST",
            speechTimeout="auto"
        )
        gather.say(text)
        return vr

# ─── Incoming Call: capture metadata & greet ────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    print("📥 /incoming-call form data:", dict(form))

    call_sid     = form.get("CallSid")
    from_number  = form.get("From")
    from_city    = form.get("FromCity")
    from_state   = form.get("FromState")
    from_zip     = form.get("FromZip")
    from_country = form.get("FromCountry")

    # save metadata
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO call_metadata
          (call_sid, from_number, from_city, from_state, from_zip, from_country)
        VALUES (?, ?, ?, ?, ?, ?);
    """, (call_sid, from_number, from_city, from_state, from_zip, from_country))
    conn.commit()
    conn.close()

    if not call_sid:
        return Response(status_code=400)

    reset_reprompt_count(call_sid)
    greeting = "Hello, thank you for calling the pharmacy. How can I help you today?"
    vr = generate_and_play_tts(greeting, call_sid, suffix="greeting")

    twi = str(vr)
    print("📤 /incoming-call TwiML:", twi)
    return Response(content=twi, media_type="application/xml")

# ─── Process Recording: multi-turn logic ────────────────────────────────────────
@app.post("/process-recording")
async def process_recording(request: Request):
    form = await request.form()
    print("📥 /process-recording form data:", dict(form))

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
            g = vr.gather(input="speech", action=f"{BASE_URL}/process-recording", method="POST", speechTimeout="auto")
            g.say(text)
            log_call_turn(call_sid, len(history)//2, None, None, "Silence reprompt")
        else:
            vr.say(text); vr.hangup()
            log_call_turn(call_sid, len(history)//2, None, None, "Silence hangup")
        twi = str(vr); print("📤 /process-recording TwiML:", twi)
        return Response(content=twi, media_type="application/xml")

    # B) Low confidence
    if confidence < 0.5:
        text = "I’m sorry, I didn’t catch that clearly. Could you please repeat?"
        vr = VoiceResponse()
        g = vr.gather(input="speech", action=f"{BASE_URL}/process-recording", method="POST", speechTimeout="auto")
        g.say(text)
        log_call_turn(call_sid, len(history)//2, user_speech, None, f"Low confidence ({confidence})")
        twi = str(vr); print("📤 /process-recording TwiML:", twi)
        return Response(content=twi, media_type="application/xml")

    # C) Add user to history
    history.append({"role":"user","content":user_speech.strip()})
    reset_reprompt_count(call_sid)

    # D) Vaccine booking flow
    last = history[-2]["content"] if len(history) > 1 else ""
    if last.startswith("Which vaccine would you like"):
        history.append({"role":"system","content":json.dumps({"vaccine_type":user_speech.strip()})})
        reply = "Got it. May I have your full name, please?"
    elif last.startswith("Got it. May I have your full name"):
        history.append({"role":"system","content":json.dumps({"patient_name":user_speech.strip()})})
        reply = "Thank you. On which date would you like to book your appointment? Please say the date."
    elif last.startswith("Thank you. On which date"):
        history.append({"role":"system","content":json.dumps({"desired_date":user_speech.strip()})})
        slots = {}
        for m in history:
            if m["role"]=="system":
                slots.update(json.loads(m["content"]))
        save_booking(call_sid, slots["vaccine_type"], slots["patient_name"], slots["desired_date"])
        reply = (f"Thank you. Your {slots['vaccine_type']} appointment for "
                 f"{slots['patient_name']} on {slots['desired_date']} is booked. Goodbye.")
        vr = generate_and_play_tts(reply, call_sid, suffix="finalv")
        vr.hangup()
        log_call_turn(call_sid, len(history)//2, user_speech, reply, "VACCINE_BOOKED")
        history.append({"role":"assistant","content":reply})
        save_history(call_sid, history)
        twi = str(vr); print("📤 /process-recording TwiML:", twi)
        return Response(content=twi, media_type="application/xml")
    else:
        # E) New intent detection & fallback
        intent = classify_intent(user_speech)
        if intent == "VACCINE":
            reply = "Which vaccine would you like? Please say the vaccine name."
        elif intent == "REFILL":
            reply = "Certainly. What is your prescription number?"
        elif intent == "HOURS":
            reply = ("Our pharmacy is open Monday to Friday, 9 AM to 6 PM, "
                     "and Saturday 10 AM to 4 PM. Anything else I can help you with?")
        elif intent == "NEAREST":
            reply = "Sure—what’s your postal code (Canada)?"
        else:
            few_shot = [
                {"role":"system","content":"You are a concise pharmacy assistant under 300 chars."},
                {"role":"user","content":"I want to schedule a vaccine appointment."},
                {"role":"assistant","content":"Which vaccine would you like? Please say the vaccine name."},
                {"role":"user","content":"I need to refill my prescription."},
                {"role":"assistant","content":"Of course. What is your prescription number?"},
                {"role":"user","content":"What are your pharmacy hours on Saturday?"},
                {"role":"assistant","content":"We’re open M–F 9–6, Sat 10–4."},
                {"role":"user","content":"Do you have ibuprofen 200 mg in stock?"},
                {"role":"assistant","content":"Yes. Would you like me to hold some?"}
            ]
            resp = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=few_shot + history,
                max_tokens=60,
                temperature=0.2
            )
            reply = resp.choices[0].message.content.strip()

    # F) Append assistant reply & log
    history.append({"role":"assistant","content":reply})
    save_history(call_sid, history)
    log_call_turn(call_sid, len(history)//2, user_speech, reply, None)

    # G) Respond
    vr = generate_and_play_tts(reply, call_sid, suffix=str(int(time.time())))
    twi = str(vr)
    print("📤 /process-recording TwiML:", twi)
    return Response(content=twi, media_type="application/xml")

# ─── Enhanced Logs API ───────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_call_logs(limit: int = 100):
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
        # transcript
        c.execute("SELECT messages FROM conversations WHERE call_sid = ?;", (log["call_sid"],))
        m = c.fetchone()
        log["transcript"] = json.loads(m[0]) if m else []
        # booking
        c.execute("SELECT vaccine_type,patient_name,desired_date,booked_at FROM bookings WHERE call_sid = ?;", (log["call_sid"],))
        b = c.fetchone()
        log["booking"] = dict(zip(["vaccine_type","patient_name","desired_date","booked_at"], b)) if b else None
        # metadata
        c.execute("""
            SELECT from_number,from_city,from_state,from_zip,from_country
              FROM call_metadata WHERE call_sid = ?;
        """, (log["call_sid"],))
        md = c.fetchone()
        log["metadata"] = {
            "from_number":  md[0],
            "from_city":    md[1],
            "from_state":   md[2],
            "from_zip":     md[3],
            "from_country": md[4],
        } if md else {}
        logs.append(log)

    conn.close()
    return JSONResponse({"logs": logs})

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
        return JSONResponse({"error":"CallSid not found"}, status_code=404)
    return JSONResponse({"call_sid": call_sid, "messages": json.loads(row[0])})
