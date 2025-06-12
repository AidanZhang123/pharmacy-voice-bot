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
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True
)

print(
    "Startup:", 
    "ELEVENLABS_API_KEY?", os.getenv("ELEVENLABS_API_KEY") is not None,
    "VOICE_ID:", os.getenv("ELEVENLABS_VOICE_ID")
)

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID            = os.getenv("ELEVENLABS_VOICE_ID")
BASE_URL            = os.getenv("BASE_URL")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
openai_client       = OpenAI(api_key=OPENAI_API_KEY)

# ─── Static files & DB init ──────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
DB_PATH = "conversations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    # conversation history + reprompts
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            call_sid TEXT PRIMARY KEY,
            messages TEXT,
            reprompt_count INTEGER DEFAULT 0
        );
    """)
    # per-turn logs
    c.execute("""
        CREATE TABLE IF NOT EXISTS call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid TEXT,
            turn_number INTEGER,
            user_text TEXT,
            assistant_reply TEXT,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # bookings
    c.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid TEXT,
            vaccine_type TEXT,
            patient_name TEXT,
            desired_date TEXT,
            booked_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # caller metadata
    c.execute("""
        CREATE TABLE IF NOT EXISTS call_metadata (
            call_sid TEXT PRIMARY KEY,
            from_number TEXT,
            from_city TEXT,
            from_state TEXT,
            from_zip TEXT,
            from_country TEXT
        );
    """)
    c.execute("PRAGMA table_info(conversations);")
    if "reprompt_count" not in [r[1] for r in c.fetchall()]:
        c.execute("ALTER TABLE conversations ADD COLUMN reprompt_count INTEGER DEFAULT 0;")
    conn.commit(); conn.close()
    print(f"[{datetime.utcnow()}] init_db complete")

init_db()

# ─── SQLite helpers ───────────────────────────────────────────────────────────────
def retry_sqlite(f,*a,**k):
    for _ in range(3):
        try:
            return f(*a,**k)
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                time.sleep(0.1)
            else:
                raise
    return f(*a,**k)

def get_history(sid):
    def _():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("SELECT messages FROM conversations WHERE call_sid=?;", (sid,))
        row = c.fetchone()
        msgs = json.loads(row[0]) if row else []
        if not row:
            c.execute(
                "INSERT INTO conversations(call_sid,messages,reprompt_count) VALUES(?,?,0);",
                (sid, json.dumps([]))
            )
            conn.commit()
        conn.close()
        return msgs
    return retry_sqlite(_)

def save_history(sid, msgs):
    def _():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute(
            "UPDATE conversations SET messages=? WHERE call_sid=?;",
            (json.dumps(msgs), sid)
        )
        conn.commit(); conn.close()
    retry_sqlite(_)

def get_reprompt_count(sid):
    try:
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("SELECT reprompt_count FROM conversations WHERE call_sid=?;", (sid,))
        row = c.fetchone(); conn.close()
        return row[0] if row else 0
    except:
        return 0

def increment_reprompt_count(sid):
    def _():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute(
            "UPDATE conversations SET reprompt_count=reprompt_count+1 WHERE call_sid=?;",
            (sid,)
        )
        conn.commit(); conn.close()
    retry_sqlite(_)

def reset_reprompt_count(sid):
    def _():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute(
            "UPDATE conversations SET reprompt_count=0 WHERE call_sid=?;",
            (sid,)
        )
        conn.commit(); conn.close()
    retry_sqlite(_)

def log_call_turn(sid, turn, ut, ar, err):
    def _():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("""
            INSERT INTO call_logs(
                call_sid, turn_number, user_text,
                assistant_reply, error_message
            ) VALUES (?,?,?,?,?);
        """, (sid, turn, ut, ar, err))
        conn.commit(); conn.close()
    retry_sqlite(_)

def save_booking(sid, vt, pn, dd):
    def _():
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("""
            INSERT INTO bookings(
                call_sid, vaccine_type, patient_name, desired_date
            ) VALUES (?,?,?,?);
        """, (sid, vt, pn, dd))
        conn.commit(); conn.close()
    retry_sqlite(_)

# ─── Intent classification ────────────────────────────────────────────────────────
def classify_intent(text: str) -> str:
    l = text.lower()
    # vaccination + booking verbs
    if (("vaccine" in l or "vaccination" in l) and
        ("book" in l or "schedule" in l or "appointment" in l)) \
       or any(kw in l for kw in ["vaccine","vaccination","shot"]):
        return "VACCINE"
    if any(kw in l for kw in ["refill","renew","prescription"]):
        return "REFILL"
    if any(kw in l for kw in ["hour","open","close","time"]):
        return "HOURS"
    if "pharmacy" in l:
        return "NEAREST"
    return "GENERAL"

# ─── ElevenLabs TTS helper ───────────────────────────────────────────────────────
def generate_and_play_tts(text: str, sid: str, suffix: str="resp") -> VoiceResponse:
    fn = f"tts_{sid}_{suffix}_{int(time.time())}.mp3"
    fp = os.path.join("static", fn)
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        if r.status_code != 200 or not r.content:
            raise Exception(f"TTS error {r.status_code}")
        with open(fp, "wb") as f:
            f.write(r.content)

        vr = VoiceResponse()
        g = vr.gather(
            input="speech",
            action=f"{BASE_URL}/process-recording",
            method="POST",
            speechTimeout="auto"
        )
        g.play(f"{BASE_URL}/static/{fn}")
        return vr

    except Exception:
        vr = VoiceResponse()
        g = vr.gather(
            input="speech",
            action=f"{BASE_URL}/process-recording",
            method="POST",
            speechTimeout="auto"
        )
        g.say(text)
        return vr

# ─── /incoming-call ──────────────────────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    print("📥 incoming-call:", dict(form))
    sid = form.get("CallSid")
    # store caller metadata
    meta = (
        sid,
        form.get("From"), form.get("FromCity"),
        form.get("FromState"), form.get("FromZip"),
        form.get("FromCountry")
    )
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO call_metadata(
            call_sid, from_number, from_city,
            from_state, from_zip, from_country
        ) VALUES (?,?,?,?,?,?);
    """, meta)
    conn.commit(); conn.close()

    if not sid:
        return Response(status_code=400)

    reset_reprompt_count(sid)
    greeting = "Hello, thank you for calling the pharmacy. How can I help you today?"
    vr = generate_and_play_tts(greeting, sid, "greeting")
    tw = str(vr); print("📤 incoming-call TwiML:", tw)
    return Response(content=tw, media_type="application/xml")

# ─── /process-recording ──────────────────────────────────────────────────────────
@app.post("/process-recording")
async def process_recording(request: Request):
    form = await request.form()
    data = dict(form)
    print("📥 process-recording:", data)

    sid = form.get("CallSid")
    us  = form.get("SpeechResult") or ""
    cf  = form.get("Confidence")
    conf = float(cf) if cf and cf.replace('.', '', 1).isdigit() else 0.0

    if not sid:
        return Response(status_code=400)

    history = get_history(sid)
    reps    = get_reprompt_count(sid)

    # 1) Emergency detection
    for kw in ("urgent","emergency","immediately","asap"):
        if kw in us.lower():
            esc = "Emergency observed; transferring you to a pharmacist now."
            log_call_turn(sid, len(history)//2+1, us, esc, "EMERGENCY_OBSERVED")
            history.append({"role":"assistant","content":esc})
            save_history(sid, history)
            vr = generate_and_play_tts(esc, sid, "escalation")
            vr.hangup()
            return Response(content=str(vr), media_type="application/xml")

    # 2) Silence reprompts (up to 3)
    if not us.strip():
        if reps < 3:
            increment_reprompt_count(sid)
            msg = "Sorry, I didn’t hear anything. Could you please repeat?"
            log_call_turn(sid, len(history)//2, None, None, "Silence reprompt")
            vr = generate_and_play_tts(msg, sid, "reprompt_silence")
            tw = str(vr); print("📤 reprompt TwiML:", tw)
            return Response(content=tw, media_type="application/xml")
        else:
            msg = "We did not receive any input. Goodbye."
            log_call_turn(sid, len(history)//2, None, None, "Silence hangup")
            vr = VoiceResponse()
            vr.say(msg)
            vr.hangup()
            tw = str(vr); print("📤 hangup TwiML:", tw)
            return Response(content=tw, media_type="application/xml")

    # 3) Low-confidence reprompt
    if conf < 0.5:
        msg = "Sorry, I didn’t catch that clearly. Could you please repeat?"
        log_call_turn(sid, len(history)//2, us, None, f"Low confidence ({conf})")
        vr = generate_and_play_tts(msg, sid, "reprompt_conf")
        tw = str(vr); print("📤 low-conf TwiML:", tw)
        return Response(content=tw, media_type="application/xml")

    # record the valid user turn
    history.append({"role":"user","content":us.strip()})
    reset_reprompt_count(sid)

    # rebuild slot_data from prior system messages
    slot_data = {}
    for m in history:
        if m["role"] == "system":
            try:
                slot_data.update(json.loads(m["content"]))
            except:
                pass

    # exact dialogue strings for booking flow
    Q1 = "Sure! Which vaccine would you like?"
    Q2 = "Got it. May I have your full name?"
    Q3 = "Thank you. On which date would you like to book your appointment?"

    # find the last assistant prompt
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"),
        ""
    )

    # 4) Vaccine booking slot flow, guaranteed three steps
    if "vaccine_type" not in slot_data and classify_intent(us) == "VACCINE":
        assistant_reply = Q1

    elif last_assistant == Q1:
        history.append({"role":"system","content": json.dumps({"vaccine_type": us.strip()})})
        assistant_reply = Q2

    elif "vaccine_type" in slot_data and "patient_name" not in slot_data and last_assistant == Q2:
        history.append({"role":"system","content": json.dumps({"patient_name": us.strip()})})
        assistant_reply = Q3

    elif "vaccine_type" in slot_data and "patient_name" in slot_data and last_assistant == Q3:
        history.append({"role":"system","content": json.dumps({"desired_date": us.strip()})})
        vt = slot_data["vaccine_type"]
        pn = slot_data["patient_name"]
        dd = us.strip()
        save_booking(sid, vt, pn, dd)
        assistant_reply = (
            f"Thank you. Your {vt} appointment for {pn} on {dd} is booked. Goodbye."
        )
        log_call_turn(sid, len(history)//2+1, us, assistant_reply, "VACCINE_BOOKED")
        history.append({"role":"assistant","content":assistant_reply})
        save_history(sid, history)
        vr = generate_and_play_tts(assistant_reply, sid, "finalv")
        vr.hangup()
        tw = str(vr); print("📤 final TwiML:", tw)
        return Response(content=tw, media_type="application/xml")

    else:
        # 5) Other intents or GPT fallback
        intent = classify_intent(us)
        if intent == "REFILL":
            assistant_reply = "Sure! What is your prescription number?"
        elif intent == "HOURS":
            assistant_reply = "We’re open Monday–Friday 9 AM–6 PM, and Saturday 10 AM–4 PM."
        elif intent == "NEAREST":
            assistant_reply = "Sure! What’s your postal code?"
        else:
            few = [{"role":"system","content":"You’re a concise pharmacy assistant—keep replies under 300 characters."}]
            resp = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=few + history,
                temperature=0.2
            )
            assistant_reply = resp.choices[0].message.content.strip()

    # append & log the assistant reply
    history.append({"role":"assistant","content":assistant_reply})
    save_history(sid, history)
    log_call_turn(sid, len(history)//2, us, assistant_reply, None)

    # generate TTS and return
    vr = generate_and_play_tts(assistant_reply, sid, str(int(time.time())))
    tw = str(vr); print("📤 response TwiML:", tw)
    return Response(content=tw, media_type="application/xml")

# ─── Dashboard endpoints ─────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_call_logs(limit: int = 100):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""
        SELECT id, call_sid, turn_number, user_text,
               assistant_reply, error_message, timestamp
        FROM call_logs
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall(); logs = []
    for r in rows:
        log = dict(zip(
            ["id","call_sid","turn_number","user_text","assistant_reply","error_message","timestamp"],
            r
        ))
        c.execute("SELECT messages FROM conversations WHERE call_sid=?;", (log["call_sid"],))
        m = c.fetchone(); log["transcript"] = json.loads(m[0]) if m else []
        c.execute("""
            SELECT vaccine_type, patient_name, desired_date, booked_at
            FROM bookings WHERE call_sid=?;
        """, (log["call_sid"],))
        b = c.fetchone()
        log["booking"] = dict(zip(
            ["vaccine_type","patient_name","desired_date","booked_at"], b
        )) if b else None
        c.execute("""
            SELECT from_number, from_city, from_state, from_zip, from_country
            FROM call_metadata WHERE call_sid=?;
        """, (log["call_sid"],))
        md = c.fetchone()
        log["metadata"] = dict(zip(
            ["from_number","from_city","from_state","from_zip","from_country"], md
        )) if md else {}
        logs.append(log)
    conn.close()
    return JSONResponse({"logs": logs})

@app.get("/api/calls")
async def list_call_sids():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT call_sid FROM conversations;")
    sids = [r[0] for r in c.fetchall()]
    conn.close()
    return JSONResponse({"call_sids": sids})

@app.get("/api/conversations/{call_sid}")
async def get_conversation(call_sid: str):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT messages FROM conversations WHERE call_sid=?;", (call_sid,))
    row = c.fetchone(); conn.close()
    if not row:
        return JSONResponse({"error": "CallSid not found"}, status_code=404)
    return JSONResponse({"call_sid": call_sid, "messages": json.loads(row[0])})
