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

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID           = os.getenv("ELEVENLABS_VOICE_ID")
BASE_URL           = os.getenv("BASE_URL")
GOOGLE_MAPS_API_KEY= os.getenv("GOOGLE_MAPS_API_KEY")
openai_client      = OpenAI(api_key=OPENAI_API_KEY)

# ─── Static & DB ─────────────────────────────────────────────────────────────────
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
    # metadata
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
    # ensure reprompt_count column
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
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("SELECT messages FROM conversations WHERE call_sid = ?;", (call_sid,))
        row = c.fetchone()
        if row:
            msgs = json.loads(row[0])
        else:
            msgs = []
            c.execute(
                "INSERT INTO conversations(call_sid,messages,reprompt_count) VALUES (?, ?, 0);",
                (call_sid, json.dumps(msgs))
            )
            conn.commit()
        conn.close()
        return msgs
    return retry_sqlite(_get)

def save_history(call_sid: str, messages: list):
    def _save():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute(
            "UPDATE conversations SET messages = ? WHERE call_sid = ?;",
            (json.dumps(messages), call_sid)
        )
        conn.commit(); conn.close()
    retry_sqlite(_save)

def get_reprompt_count(call_sid: str) -> int:
    try:
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("SELECT reprompt_count FROM conversations WHERE call_sid = ?;", (call_sid,))
        row = c.fetchone(); conn.close()
        return row[0] if row else 0
    except:
        return 0

def increment_reprompt_count(call_sid: str):
    def _inc():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("UPDATE conversations SET reprompt_count = reprompt_count + 1 WHERE call_sid = ?;", (call_sid,))
        conn.commit(); conn.close()
    retry_sqlite(_inc)

def reset_reprompt_count(call_sid: str):
    def _reset():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("UPDATE conversations SET reprompt_count = 0 WHERE call_sid = ?;", (call_sid,))
        conn.commit(); conn.close()
    retry_sqlite(_reset)

def log_call_turn(call_sid: str, turn_number: int, user_text: str=None,
                  assistant_reply: str=None, error_message: str=None):
    def _log():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("""
            INSERT INTO call_logs(
                call_sid, turn_number, user_text, assistant_reply, error_message
            ) VALUES (?, ?, ?, ?, ?);
        """, (call_sid, turn_number, user_text, assistant_reply, error_message))
        conn.commit(); conn.close()
    retry_sqlite(_log)

def save_booking(call_sid: str, vaccine_type: str, patient_name: str, desired_date: str):
    def _insert():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("""
            INSERT INTO bookings(call_sid, vaccine_type, patient_name, desired_date)
            VALUES (?, ?, ?, ?);
        """, (call_sid, vaccine_type, patient_name, desired_date))
        conn.commit(); conn.close()
    retry_sqlite(_insert)

# ─── Intent Classification ───────────────────────────────────────────────────────
def classify_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["vaccine","vaccination","shot"]): return "VACCINE"
    if any(k in t for k in ["refill","renew","prescription"]): return "REFILL"
    if any(k in t for k in ["hour","open","close","time"]): return "HOURS"
    if "pharmacy" in t: return "NEAREST"
    return "GENERAL"

# ─── ElevenLabs TTS Helper ───────────────────────────────────────────────────────
def generate_and_play_tts(text: str, call_sid: str, suffix: str="resp") -> VoiceResponse:
    fn = f"tts_{call_sid}_{suffix}_{int(time.time())}.mp3"
    fp = os.path.join("static", fn)
    print("🔑 ELEVENLABS_API_KEY set?", ELEVENLABS_API_KEY is not None, "VOICE_ID:", VOICE_ID)
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
        payload = {"text": text, "model_id": "eleven_multilingual_v2",
                   "voice_settings": {"stability":0.5,"similarity_boost":0.5}}
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code != 200 or not resp.content:
            print(f"[{datetime.utcnow()}] TTS error {resp.status_code}: {resp.text}")
            raise Exception("TTS failed")
        with open(fp, "wb") as f: f.write(resp.content)
        vr = VoiceResponse()
        g = vr.gather(input="speech",
                      action=f"{BASE_URL}/process-recording",
                      method="POST", speechTimeout="auto")
        g.play(f"{BASE_URL}/static/{fn}")
        return vr
    except Exception as e:
        print(f"[{datetime.utcnow()}] ElevenLabs TTS exception: {e}")
        vr = VoiceResponse()
        g = vr.gather(input="speech",
                      action=f"{BASE_URL}/process-recording",
                      method="POST", speechTimeout="auto")
        g.say(text)
        return vr

# ─── Incoming Call ──────────────────────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form() 
    print("📥 /incoming-call:", dict(form))
    cid = form.get("CallSid")
    # metadata
    md=(cid, form.get("From"),form.get("FromCity"),form.get("FromState"),
        form.get("FromZip"),form.get("FromCountry"))
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("""INSERT OR REPLACE INTO call_metadata
                 (call_sid,from_number,from_city,from_state,from_zip,from_country)
                 VALUES(?,?,?,?,?,?);""", md)
    conn.commit(); conn.close()
    if not cid: return Response(status_code=400)
    reset_reprompt_count(cid)
    greeting="Hello, thank you for calling the pharmacy. How can I help you today?"
    vr=generate_and_play_tts(greeting,cid,suffix="greeting")
    tw=str(vr); print("📤 /incoming-call TwiML:",tw)
    return Response(content=tw, media_type="application/xml")

# ─── Process Recording ────────────────────────────────────────────────────────────
@app.post("/process-recording")
async def process_recording(request: Request):
    form = await request.form()
    print("📥 /process-recording:",dict(form))
    cid=form.get("CallSid"); us=form.get("SpeechResult") or ""
    cf=form.get("Confidence"); conf=float(cf) if cf and cf.replace('.','',1).isdigit() else 0.0
    if not cid: return Response(status_code=400)
    history=get_history(cid); reps=get_reprompt_count(cid)

    # Escalation
    for kw in ("urgent","emergency","immediately","asap"):
        if kw in us.lower():
            esc="I understand this is an emergency – redirecting you to a pharmacist now. Please hold."
            log_call_turn(cid,len(history)//2+1,us,esc,
                          "EMERGENCY_OBSERVED – transferred")
            history.append({"role":"assistant","content":esc}); save_history(cid,history)
            vr=generate_and_play_tts(esc,cid,suffix="escalation"); vr.hangup()
            return Response(content=str(vr),media_type="application/xml")

    # Silence
    if not us.strip():
        msg=reps<3 and "Sorry, I didn’t hear anything. Could you repeat?" \
                 or "We did not receive any input. Goodbye."
        vr=generate_and_play_tts(msg,cid,suffix=reps<3 and "reprompt" or "hangup")
        if reps<3:
            increment_reprompt_count(cid)
            log_call_turn(cid,len(history)//2,None,None,"Silence reprompt")
        else:
            log_call_turn(cid,len(history)//2,None,None,"Silence hangup"); vr.hangup()
        tw=str(vr); print("📤 /process-recording TwiML:",tw)
        return Response(content=tw,media_type="application/xml")

    # Low confidence
    if conf<0.5:
        msg="Sorry, I didn’t catch that clearly. Could you repeat?"
        vr=generate_and_play_tts(msg,cid,suffix="reprompt_low")
        log_call_turn(cid,len(history)//2,us,None,f"Low confidence ({conf})")
        tw=str(vr); print("📤 /process-recording TwiML:",tw)
        return Response(content=tw,media_type="application/xml")

    # Add user message
    history.append({"role":"user","content":us.strip()})
    reset_reprompt_count(cid)

    # Vaccine booking flow
    last=history[-2]["content"] if len(history)>1 else ""
    if last.startswith("Sure! Which vaccine"):
        history.append({"role":"system","content":json.dumps({"vaccine_type":us.strip()})})
        reply="Sure! Which vaccine would you like?"  # updated preset
    elif last.startswith("Sure! Which vaccine"):
        # duplicate check
        pass
    elif last.startswith("Sure! Which vaccine") is False and last.startswith("Got it.") is False:
        # initial intent detection
        intent=classify_intent(us)
        if intent=="VACCINE":
            reply="Sure! Which vaccine would you like?"
        elif intent=="REFILL":
            reply="Sure! What is your prescription number?"
        elif intent=="HOURS":
            reply="We’re open Monday to Friday 9 AM to 6 PM, and Saturday 10 AM to 4 PM."
        elif intent=="NEAREST":
            reply="Sure! What’s your postal code?"
        else:
            few=[{"role":"system","content":"You are concise; keep replies under 300 characters."}]
            out=openai_client.chat.completions.create(
                model="gpt-3.5-turbo",messages=few+history,temperature=0.2
            )
            reply=out.choices[0].message.content.strip()
    # subsequent slots and confirmations unchanged...

    # Append & log
    history.append({"role":"assistant","content":reply})
    save_history(cid,history)
    log_call_turn(cid,len(history)//2,us,reply,None)

    vr=generate_and_play_tts(reply,cid,suffix=str(int(time.time())))
    tw=str(vr); print("📤 /process-recording TwiML:",tw)
    return Response(content=tw,media_type="application/xml")
