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

# ─── CORS (allow React dev server) ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─── Load API keys & config from .env ───────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY   = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID             = os.getenv("ELEVENLABS_VOICE_ID")
BASE_URL             = os.getenv("BASE_URL")
GOOGLE_MAPS_API_KEY  = os.getenv("GOOGLE_MAPS_API_KEY")

# ─── Initialize OpenAI client ───────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Ensure “static” directory exists (to serve TTS MP3s) ───────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── SQLite DB setup ────────────────────────────────────────────────────────────
DB_PATH = "conversations.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # 1) conversations table tracks history + reprompt_count
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                call_sid        TEXT PRIMARY KEY,
                messages        TEXT,
                reprompt_count  INTEGER DEFAULT 0
            );
        """)
        # 2) call_logs for analytics
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
        # 3) bookings table: store completed vaccine bookings
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

        # 4) If reprompt_count is missing (old schema), add it
        c.execute("PRAGMA table_info(conversations);")
        cols = [row[1] for row in c.fetchall()]
        if "reprompt_count" not in cols:
            c.execute("ALTER TABLE conversations ADD COLUMN reprompt_count INTEGER DEFAULT 0;")
            conn.commit()

        conn.close()
        print(f"[{datetime.utcnow()}] init_db: using SQLite path: {os.path.abspath(DB_PATH)}")
    except Exception as e:
        print(f"[{datetime.utcnow()}] init_db error: {e}")

def retry_sqlite(func, *args, retries=3, delay=0.1, **kwargs):
    for _ in range(retries):
        try:
            return func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                time.sleep(delay)
                continue
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
            print(f"[{datetime.utcnow()}] get_history error for {call_sid}: {e}")
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
            print(f"[{datetime.utcnow()}] save_history error for {call_sid}: {e}")
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
        print(f"[{datetime.utcnow()}] get_reprompt_count error for {call_sid}: {e}")
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
            print(f"[{datetime.utcnow()}] increment_reprompt_count error for {call_sid}: {e}")
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
            print(f"[{datetime.utcnow()}] reset_reprompt_count error for {call_sid}: {e}")
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
            print(f"[{datetime.utcnow()}] log_call_turn error for {call_sid}, turn {turn_number}: {e}")
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
            print(f"[{datetime.utcnow()}] save_booking error for {call_sid}: {e}")
    retry_sqlite(_insert)

init_db()

# ─── Utility: Classify intents (VACCINE, REFILL, HOURS, NEAREST, GENERAL) ─────
def classify_intent(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in ["vaccine", "shot", "vaccination", "schedule a vaccine"]):
        return "VACCINE"
    if any(kw in lower for kw in ["refill", "renew", "prescription"]):
        return "REFILL"
    if any(kw in lower for kw in ["hour", "open", "close", "time"]):
        return "HOURS"
    if any(kw in lower for kw in ["nearest pharmacy", "closest pharmacy", "find pharmacy", "pharmacy near me"]):
        return "NEAREST"
    return "GENERAL"

# ─── Helper: Generate ElevenLabs TTS MP3 and return a <Play> TwiML response ────
def generate_and_play_tts(text: str, call_sid: str, suffix: str="resp") -> VoiceResponse:
    tts_filename = f"tts_{call_sid}_{suffix}_{int(time.time())}.mp3"
    tts_filepath = os.path.join("static", tts_filename)

    # Debug: verify environment variables
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
            "model_id": "eleven_monolingual_v1",
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

# ─── Incoming Call: Play Greeting inside <Gather> ───────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    if not call_sid:
        return Response(status_code=400)

    reset_reprompt_count(call_sid)
    greeting_text = "Hello, thank you for calling the pharmacy. How can I help you today?"
    return generate_and_play_tts(greeting_text, call_sid, suffix="greeting")

# ─── Process Recording: Main multi‐turn logic ────────────────────────────────────
@app.post("/process-recording")
async def process_recording(request: Request):
    form = await request.form()
    call_sid       = form.get("CallSid")
    user_speech    = form.get("SpeechResult")
    confidence_str = form.get("Confidence")

    if not call_sid:
        return Response(status_code=400)

    try:
        confidence = float(confidence_str) if confidence_str is not None else 0.0
    except ValueError:
        confidence = 0.0

    history   = get_history(call_sid)
    reprompts = get_reprompt_count(call_sid)

    # A) SILENCE
    if not user_speech or user_speech.strip() == "":
        if reprompts < 3:
            increment_reprompt_count(call_sid)
            text = "I’m sorry, I didn’t hear anything. Could you please repeat?"
            vr = VoiceResponse()
            gather = vr.gather(
                input="speech",
                action=f"{BASE_URL}/process-recording",
                method="POST",
                speechTimeout="auto"
            )
            gather.say(text)
            log_call_turn(
                call_sid=call_sid,
                turn_number=0,
                user_text=None,
                assistant_reply=None,
                error_message="Silence reprompt"
            )
            return Response(content=str(vr), media_type="application/xml")
        else:
            vr = VoiceResponse()
            vr.say("We did not receive any input. Goodbye.")
            vr.hangup()
            log_call_turn(
                call_sid=call_sid,
                turn_number=0,
                user_text=None,
                assistant_reply=None,
                error_message="Silence hangup after 3 reprompts"
            )
            return Response(content=str(vr), media_type="application/xml")

    # B) LOW CONFIDENCE
    if confidence < 0.5:
        text = "I’m sorry, I didn’t catch that clearly. Could you please repeat?"
        vr = VoiceResponse()
        gather = vr.gather(
            input="speech",
            action=f"{BASE_URL}/process-recording",
            method="POST",
            speechTimeout="auto"
        )
        gather.say(text)
        log_call_turn(
            call_sid=call_sid,
            turn_number=0,
            user_text=user_speech,
            assistant_reply=None,
            error_message=f"Low confidence ({confidence})"
        )
        return Response(content=str(vr), media_type="application/xml")

    # C) VALID SPEECH
    user_text = user_speech.strip()
    reset_reprompt_count(call_sid)
    print(f"[{datetime.utcnow()}] CallSid={call_sid} • User said: {user_text!r} (conf={confidence})")

    # Corrections
    lower = user_text.lower()
    correction_keywords = ("actually", "i mean", "sorry", "change")
    if any(lower.startswith(k) for k in correction_keywords):
        corrected = user_text
        for k in correction_keywords:
            if lower.startswith(k):
                corrected = user_text[len(k):].strip(" ,:")
                break

        last_slot_idx = None
        last_slot_key = None
        for idx in reversed(range(len(history))):
            msg = history[idx]
            if msg["role"] == "system":
                try:
                    slot_obj = json.loads(msg["content"])
                    if isinstance(slot_obj, dict) and len(slot_obj) == 1:
                        last_slot_idx = idx
                        last_slot_key = next(iter(slot_obj.keys()))
                        break
                except Exception:
                    continue

        if last_slot_idx is not None and last_slot_key:
            history[last_slot_idx] = {"role": "system", "content": json.dumps({last_slot_key: corrected})}

            if last_slot_key == "vaccine_type":
                assistant_reply = f"Sure—so you want the {corrected} vaccine. May I have your full name, please?"
            elif last_slot_key == "patient_name":
                assistant_reply = f"Thank you. Noted your name as {corrected}. On which date would you like to book your appointment?"
            elif last_slot_key == "desired_date":
                slot_data = {}
                for msg in history:
                    if msg["role"] == "system":
                        try:
                            slot_data.update(json.loads(msg["content"]))
                        except:
                            pass
                vt = slot_data.get("vaccine_type", "your vaccine")
                pn = slot_data.get("patient_name", "the patient")
                dd = corrected
                save_booking(call_sid, vt, pn, dd)
                assistant_reply = f"Thank you. Your {vt} appointment for {pn} on {dd} is booked. Goodbye."
                log_call_turn(
                    call_sid=call_sid,
                    turn_number=(len(history)//2 + 1),
                    user_text=corrected,
                    assistant_reply=assistant_reply,
                    error_message="VACCINE_BOOKED (correction)"
                )
                history.append({"role": "assistant", "content": assistant_reply})
                save_history(call_sid, history)
                vr = generate_and_play_tts(assistant_reply, call_sid, suffix="finalv")
                vr.hangup()
                return Response(content=str(vr), media_type="application/xml")
            elif last_slot_key == "nearest_postal":
                assistant_reply = f"Got it—your postal code is {corrected}. Looking up nearest pharmacies..."
                user_text = corrected
            else:
                assistant_reply = "I’m sorry, I’m not sure what you wish to correct. Could you please restate?"
        else:
            assistant_reply = "I’m sorry, I’m not sure what you wish to correct. Could you please restate?"

        history.append({"role": "assistant", "content": assistant_reply})
        save_history(call_sid, history)
        log_call_turn(
            call_sid=call_sid,
            turn_number=(len(history)//2 + 1),
            user_text=user_text,
            assistant_reply=assistant_reply,
            error_message="Slot correction"
        )
        return Response(content=str(generate_and_play_tts(assistant_reply, call_sid, suffix="corr")), media_type="application/xml")

    # D) Vaccine subflow
    last_assistant = history[-1]["content"] if history and history[-1]["role"] == "assistant" else ""
    if last_assistant.startswith("Which vaccine would you like"):
        vaccine_type = user_text
        history.append({"role": "system", "content": json.dumps({"vaccine_type": vaccine_type})})
        assistant_reply = "Got it. May I have your full name, please?"
        history.append({"role": "assistant", "content": assistant_reply})
        save_history(call_sid, history)
        log_call_turn(
            call_sid=call_sid,
            turn_number=(len(history)//2),
            user_text=vaccine_type,
            assistant_reply=assistant_reply,
            error_message=None
        )
        return Response(content=str(generate_and_play_tts(assistant_reply, call_sid, suffix="ask_name")), media_type="application/xml")

    elif last_assistant.startswith("Got it. May I have your full name"):
        patient_name = user_text
        history.append({"role": "system", "content": json.dumps({"patient_name": patient_name})})
        assistant_reply = "Thank you. On which date would you like to book your appointment? Please say the date."
        history.append({"role": "assistant", "content": assistant_reply})
        save_history(call_sid, history)
        log_call_turn(
            call_sid=call_sid,
            turn_number=(len(history)//2),
            user_text=patient_name,
            assistant_reply=assistant_reply,
            error_message=None
        )
        return Response(content=str(generate_and_play_tts(assistant_reply, call_sid, suffix="ask_date")), media_type="application/xml")

    elif last_assistant.startswith("Thank you. On which date"):
        desired_date = user_text
        slot_data = {}
        for msg in history:
            if msg["role"] == "system":
                try:
                    slot_data.update(json.loads(msg["content"]))
                except:
                    pass
        vt = slot_data.get("vaccine_type", "your vaccine")
        pn = slot_data.get("patient_name", "the patient")
        dd = desired_date
        save_booking(call_sid, vt, pn, dd)
        assistant_reply = f"Thank you. Your {vt} appointment for {pn} on {dd} is booked. Goodbye."
        history.append({"role": "assistant", "content": assistant_reply})
        save_history(call_sid, history)
        log_call_turn(
            call_sid=call_sid,
            turn_number=(len(history)//2),
            user_text=desired_date,
            assistant_reply=assistant_reply,
            error_message="VACCINE_BOOKED"
        )
        vr = generate_and_play_tts(assistant_reply, call_sid, suffix="finalv")
        vr.hangup()
        return Response(content=str(vr), media_type="application/xml")

    # E) Nearest pharmacy subflow
    last_assistant = history[-1]["content"] if history and history[-1]["role"] == "assistant" else ""
    if last_assistant.startswith("Sure—what’s your postal code"):
        postal_code = user_text.replace(" ", "")
        history.append({"role": "system", "content": json.dumps({"nearest_postal": postal_code})})
        geo_url = (
            "https://maps.googleapis.com/maps/api/geocode/json"
            f"?address={postal_code},Canada&key={GOOGLE_MAPS_API_KEY}"
        )
        try:
            geo_resp = requests.get(geo_url, timeout=5).json()
            if geo_resp.get("status") != "OK" or not geo_resp.get("results"):
                raise Exception("Geocode failed or no results")
            loc = geo_resp["results"][0]["geometry"]["location"]
            lat, lng = loc["lat"], loc["lng"]
        except Exception as e:
            assistant_reply = (
                "I’m sorry, I couldn’t look up that postal code. "
                "Could you please repeat your postal code?"
            )
            history.append({"role": "assistant", "content": assistant_reply})
            save_history(call_sid, history)
            log_call_turn(
                call_sid=call_sid,
                turn_number=(len(history)//2),
                user_text=postal_code,
                assistant_reply=assistant_reply,
                error_message="Geocode failed"
            )
            return Response(content=str(generate_and_play_tts(assistant_reply, call_sid, suffix="err_pc")), media_type="application/xml")

        places_url = (
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            f"?location={lat},{lng}&radius=5000&type=pharmacy&key={GOOGLE_MAPS_API_KEY}"
        )
        try:
            places_resp = requests.get(places_url, timeout=5).json()
            if places_resp.get("status") != "OK" or not places_resp.get("results"):
                raise Exception("Places search failed or no results")
            results = places_resp["results"][:3]
            nearest_list = []
            for place in results:
                name = place.get("name", "Unnamed Pharmacy")
                addr = place.get("vicinity", "Address not available")
                nearest_list.append(f"{name} at {addr}")
            assistant_reply = (
                "Here are the three closest pharmacies to you: "
                + "; ".join(nearest_list)
                + ". Is there anything else I can help you with?"
            )
        except Exception as e:
            assistant_reply = (
                "I’m sorry, I couldn’t find nearby pharmacies for that postal code. "
                "Could you please provide a different postal code?"
            )
            history.append({"role": "assistant", "content": assistant_reply})
            save_history(call_sid, history)
            log_call_turn(
                call_sid=call_sid,
                turn_number=(len(history)//2),
                user_text=postal_code,
                assistant_reply=assistant_reply,
                error_message="Places API failed"
            )
            return Response(content=str(generate_and_play_tts(assistant_reply, call_sid, suffix="err_pc2")), media_type="application/xml")

        history.append({"role": "assistant", "content": assistant_reply})
        save_history(call_sid, history)
        log_call_turn(
            call_sid=call_sid,
            turn_number=(len(history)//2),
            user_text=postal_code,
            assistant_reply=assistant_reply,
            error_message=None
        )
        return Response(content=str(generate_and_play_tts(assistant_reply, call_sid, suffix="nearest")), media_type="application/xml")

    # F) New intent detection
    intent = classify_intent(user_text)

    if intent == "VACCINE":
        assistant_reply = "Which vaccine would you like? Please say the vaccine name."
    elif intent == "REFILL":
        assistant_reply = "Certainly. What is your prescription number?"
    elif intent == "HOURS":
        assistant_reply = (
            "Our pharmacy is open Monday to Friday, 9 AM to 6 PM, "
            "and Saturday 10 AM to 4 PM. Anything else I can help you with?"
        )
    elif intent == "NEAREST":
        assistant_reply = "Sure—what’s your postal code (Canada)?"
    else:
        few_shot = [
            {
                "role": "system",
                "content": (
                    "You are a friendly, concise, and accurate pharmacy assistant. "
                    "Keep replies under 300 characters."
                )
            },
            {"role": "user", "content": "I want to schedule a vaccine appointment."},
            {"role": "assistant", "content": "Which vaccine would you like? Please say the vaccine name."},
            {"role": "user", "content": "I need to refill my prescription."},
            {"role": "assistant", "content": "Of course. What is your prescription number?"},
            {"role": "user", "content": "What are your pharmacy hours on Saturday?"},
            {"role": "assistant", "content": "We’re open Monday–Friday 9 AM–6 PM, and Saturday 10 AM–4 PM. Anything else?"},
            {"role": "user", "content": "Do you have ibuprofen 200 mg in stock?"},
            {"role": "assistant", "content": "Let me check… Yes, we have ibuprofen 200 mg in stock. Would you like me to hold some for you?"},
        ]
        prompt_with_instructions = few_shot + history
        try:
            chat_resp = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt_with_instructions,
                max_tokens=60,
                temperature=0.2
            )
            assistant_reply = chat_resp.choices[0].message.content.strip()
            print(f"[{datetime.utcnow()}] CallSid={call_sid} • GPT reply: {assistant_reply!r}")
        except Exception as e:
            error_msg = f"GPT error: {e}"
            print(f"[{datetime.utcnow()}] {error_msg}")
            assistant_reply = "I’m sorry, I’m having trouble right now. Please try again later."
            log_call_turn(
                call_sid=call_sid,
                turn_number=(len(history)//2 + 1),
                user_text=user_text,
                assistant_reply=assistant_reply,
                error_message=error_msg
            )

    # H) Append assistant_reply & log
    if assistant_reply is None:
        assistant_reply = "I’m sorry, I’m not sure how to help with that. Goodbye."

    history.append({"role": "assistant", "content": assistant_reply})
    save_history(call_sid, history)
    log_call_turn(
        call_sid=call_sid,
        turn_number=(len(history)//2 + 1),
        user_text=user_text,
        assistant_reply=assistant_reply,
        error_message=None
    )

    # I) Generate TTS + <Gather>
    vr = generate_and_play_tts(assistant_reply, call_sid, suffix=str(int(time.time())))
    return Response(content=str(vr), media_type="application/xml")


# ─── API endpoints for React dashboard (unchanged) ──────────────────────────────

@app.get("/api/logs")
async def get_call_logs(limit: int = 100):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, call_sid, turn_number, user_text, assistant_reply, error_message, timestamp
        FROM call_logs
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    keys = ["id", "call_sid", "turn_number", "user_text", "assistant_reply", "error_message", "timestamp"]
    logs = [dict(zip(keys, row)) for row in rows]
    return JSONResponse({"logs": logs})

@app.get("/api/calls")
async def list_call_sids():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT call_sid FROM conversations;")
    rows = c.fetchall()
    conn.close()
    call_sids = [r[0] for r in rows]
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
    messages = json.loads(row[0])
    return JSONResponse({"call_sid": call_sid, "messages": messages})
