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
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

print("Startup:", 
      "ELEVENLABS_API_KEY set?", os.getenv("ELEVENLABS_API_KEY") is not None, 
      "VOICE_ID:", os.getenv("ELEVENLABS_VOICE_ID"))

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY   = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID             = os.getenv("ELEVENLABS_VOICE_ID")
BASE_URL             = os.getenv("BASE_URL")
GOOGLE_MAPS_API_KEY  = os.getenv("GOOGLE_MAPS_API_KEY")
openai_client        = OpenAI(api_key=OPENAI_API_KEY)

# ─── Static & DB ─────────────────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
DB_PATH = "conversations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            call_sid TEXT PRIMARY KEY, messages TEXT, reprompt_count INTEGER DEFAULT 0
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid TEXT, turn_number INTEGER,
            user_text TEXT, assistant_reply TEXT,
            error_message TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid TEXT, vaccine_type TEXT,
            patient_name TEXT, desired_date TEXT,
            booked_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS call_metadata (
            call_sid TEXT PRIMARY KEY,
            from_number TEXT, from_city TEXT,
            from_state TEXT, from_zip TEXT,
            from_country TEXT
        );
    """)
    c.execute("PRAGMA table_info(conversations);")
    if "reprompt_count" not in [r[1] for r in c.fetchall()]:
        c.execute("ALTER TABLE conversations ADD COLUMN reprompt_count INTEGER DEFAULT 0;")
    conn.commit(); conn.close()
    print(f"[{datetime.utcnow()}] init_db complete")

init_db()

# ─── SQLite Helpers ─────────────────────────────────────────────────────────────
def retry_sqlite(func, *args, **kwargs):
    for _ in range(3):
        try: return func(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "locked" in str(e): time.sleep(0.1)
            else: raise
    return func(*args, **kwargs)

def get_history(sid):
    def _():
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("SELECT messages FROM conversations WHERE call_sid=?;",(sid,))
        row=c.fetchone()
        msgs=json.loads(row[0]) if row else []
        if not row:
            c.execute("INSERT INTO conversations(call_sid,messages,reprompt_count) VALUES(?,?,0);",
                      (sid,json.dumps(msgs)))
            conn.commit()
        conn.close(); return msgs
    return retry_sqlite(_)

def save_history(sid,msgs):
    def _():
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("UPDATE conversations SET messages=? WHERE call_sid=?;",
                  (json.dumps(msgs),sid))
        conn.commit(); conn.close()
    retry_sqlite(_)

def get_reprompt_count(sid):
    try:
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("SELECT reprompt_count FROM conversations WHERE call_sid=?;",(sid,))
        row=c.fetchone(); conn.close()
        return row[0] if row else 0
    except: return 0

def increment_reprompt_count(sid):
    def _():
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("UPDATE conversations SET reprompt_count=reprompt_count+1 WHERE call_sid=?;",(sid,))
        conn.commit(); conn.close()
    retry_sqlite(_)

def reset_reprompt_count(sid):
    def _():
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("UPDATE conversations SET reprompt_count=0 WHERE call_sid=?;",(sid,))
        conn.commit(); conn.close()
    retry_sqlite(_)

def log_call_turn(sid,turn,ut,ar,err):
    def _():
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("""INSERT INTO call_logs(
                        call_sid, turn_number, user_text,
                        assistant_reply, error_message
                     ) VALUES(?,?,?,?,?);""",(sid,turn,ut,ar,err))
        conn.commit(); conn.close()
    retry_sqlite(_)

def save_booking(sid,vt,pn,dd):
    def _():
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("""INSERT INTO bookings(
                        call_sid, vaccine_type, patient_name, desired_date
                     ) VALUES(?,?,?,?);""",(sid,vt,pn,dd))
        conn.commit(); conn.close()
    retry_sqlite(_)

# ─── Intent Classification ───────────────────────────────────────────────────────
def classify_intent(text):
    t=text.lower()
    if any(k in t for k in ["vaccine","shot"]): return "VACCINE"
    if any(k in t for k in ["refill","renew","prescription"]): return "REFILL"
    if any(k in t for k in ["hour","open","close","time"]): return "HOURS"
    if "pharmacy" in t: return "NEAREST"
    return "GENERAL"

# ─── ElevenLabs TTS Helper ───────────────────────────────────────────────────────
def generate_and_play_tts(text,sid,suffix="resp"):
    fn=f"tts_{sid}_{suffix}_{int(time.time())}.mp3"
    fp=os.path.join("static",fn)
    print("TTS key?",ELEVENLABS_API_KEY is not None,"VOICE_ID:",VOICE_ID)
    try:
        url=f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers={"xi-api-key":ELEVENLABS_API_KEY,"Content-Type":"application/json"}
        payload={"text":text,"model_id":"eleven_multilingual_v2",
                 "voice_settings":{"stability":0.5,"similarity_boost":0.5}}
        r=requests.post(url,json=payload,headers=headers,timeout=10)
        if r.status_code!=200 or not r.content:
            print("TTS error",r.status_code,r.text); raise Exception()
        with open(fp,"wb") as f: f.write(r.content)
        vr=VoiceResponse()
        gr=vr.gather(input="speech",action=f"{BASE_URL}/process-recording",method="POST",speechTimeout="auto")
        gr.play(f"{BASE_URL}/static/{fn}")
        return vr
    except:
        vr=VoiceResponse()
        gr=vr.gather(input="speech",action=f"{BASE_URL}/process-recording",method="POST",speechTimeout="auto")
        gr.say(text)
        return vr

# ─── Incoming Call ──────────────────────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(req:Request):
    f=await req.form(); d=dict(f); print("📥 incoming-call:",d)
    sid=f.get("CallSid")
    # save metadata
    meta=(sid,f.get("From"),f.get("FromCity"),f.get("FromState"),f.get("FromZip"),f.get("FromCountry"))
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("""INSERT OR REPLACE INTO call_metadata
                 (call_sid,from_number,from_city,from_state,from_zip,from_country)
                 VALUES(?,?,?,?,?,?);""",meta)
    conn.commit(); conn.close()
    if not sid: return Response(status_code=400)
    reset_reprompt_count(sid)
    greeting="Hello, thank you for calling the pharmacy. How can I help you today?"
    vr=generate_and_play_tts(greeting,sid,"greeting")
    tw=str(vr); print("📤 incoming-call TwiML:",tw)
    return Response(content=tw,media_type="application/xml")

# ─── Process Recording ────────────────────────────────────────────────────────────
@app.post("/process-recording")
async def process_recording(req:Request):
    f=await req.form(); d=dict(f); print("📥 process-recording:",d)
    sid=f.get("CallSid"); us=f.get("SpeechResult") or ""; cf=f.get("Confidence")
    conf=float(cf) if cf and cf.replace('.','',1).isdigit() else 0.0
    if not sid: return Response(status_code=400)
    history=get_history(sid); reps=get_reprompt_count(sid)

    # Escalation keywords
    for kw in ("urgent","emergency","immediately","asap"):
        if kw in us.lower():
            esc="Emergency observed; transferring to pharmacist. Please hold."
            log_call_turn(sid,len(history)//2+1,us,esc,"EMERGENCY_OBSERVED")
            history.append({"role":"assistant","content":esc}); save_history(sid,history)
            vr=generate_and_play_tts(esc,sid,"escalation"); vr.hangup()
            return Response(content=str(vr),media_type="application/xml")

    # Silence
    if not us.strip():
        msg=reps<3 and "Sorry, I didn’t hear you. Could you repeat?" or "Goodbye."
        vr=generate_and_play_tts(msg,sid,reps<3 and "reprompt" or "hangup")
        if reps<3:
            increment_reprompt_count(sid); log_call_turn(sid,len(history)//2,None,None,"Silence reprompt")
        else:
            log_call_turn(sid,len(history)//2,None,None,"Silence hangup"); vr.hangup()
        tw=str(vr); print("📤 process-recording TwiML:",tw)
        return Response(content=tw,media_type="application/xml")

    # Low confidence
    if conf<0.5:
        msg="Sorry, I didn’t catch that. Could you repeat?"
        vr=generate_and_play_tts(msg,sid,"reprompt_low")
        log_call_turn(sid,len(history)//2,us,None,f"Low confidence ({conf})")
        tw=str(vr); print("📤 process-recording TwiML:",tw)
        return Response(content=tw,media_type="application/xml")

    # add user
    history.append({"role":"user","content":us.strip()})
    reset_reprompt_count(sid)

    # determine last assistant
    last_assistant=""
    for m in reversed(history):
        if m["role"]=="assistant":
            last_assistant=m["content"]
            break

    # Vaccine flow
    if last_assistant == "Sure! Which vaccine would you like?":
        vt=us.strip()
        history.append({"role":"system","content":json.dumps({"vaccine_type":vt})})
        assistant_reply="Got it. May I have your full name?"
    elif last_assistant.startswith("Got it. May I have your full name"):
        pn=us.strip()
        history.append({"role":"system","content":json.dumps({"patient_name":pn})})
        assistant_reply="Thank you. On which date would you like to book your appointment?"
    elif last_assistant.startswith("Thank you. On which date"):
        dd=us.strip()
        history.append({"role":"system","content":json.dumps({"desired_date":dd})})
        slots={}; 
        for m in history:
            if m["role"]=="system":
                slots.update(json.loads(m["content"]))
        save_booking(sid,slots["vaccine_type"],slots["patient_name"],slots["desired_date"])
        assistant_reply=(f"Thank you. Your {slots['vaccine_type']} appointment for "
                         f"{slots['patient_name']} on {slots['desired_date']} is booked. Goodbye.")
        vr=generate_and_play_tts(assistant_reply,sid,"finalv"); vr.hangup()
        log_call_turn(sid,len(history)//2,us,assistant_reply,"VACCINE_BOOKED")
        history.append({"role":"assistant","content":assistant_reply})
        save_history(sid,history)
        tw=str(vr); print("📤 process-recording TwiML:",tw)
        return Response(content=tw,media_type="application/xml")
    else:
        # New intent
        intent=classify_intent(us)
        if intent=="VACCINE":
            assistant_reply="Sure! Which vaccine would you like?"
        elif intent=="REFILL":
            assistant_reply="Sure! What is your prescription number?"
        elif intent=="HOURS":
            assistant_reply="We’re open Monday–Friday 9 AM–6 PM, and Saturday 10 AM–4 PM."
        elif intent=="NEAREST":
            assistant_reply="Sure! What’s your postal code?"
        else:
            few=[{"role":"system","content":"You are concise; keep replies under 300 characters."}]
            resp=openai_client.chat.completions.create(
                model="gpt-3.5-turbo",messages=few+history,temperature=0.2
            )
            assistant_reply=resp.choices[0].message.content.strip()

    history.append({"role":"assistant","content":assistant_reply})
    save_history(sid,history)
    log_call_turn(sid,len(history)//2,us,assistant_reply,None)

    vr=generate_and_play_tts(assistant_reply,sid,str(int(time.time())))
    tw=str(vr); print("📤 process-recording TwiML:",tw)
    return Response(content=tw,media_type="application/xml")

# ─── /api/logs ─────────────────────────────────────────────────────────────────
@app.get("/api/logs")
async def get_call_logs(limit:int=100):
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("""SELECT id,call_sid,turn_number,user_text,
                        assistant_reply,error_message,timestamp
                 FROM call_logs ORDER BY timestamp DESC LIMIT ?;""",(limit,))
    rows=c.fetchall(); logs=[]
    for r in rows:
        log=dict(zip(["id","call_sid","turn_number","user_text",
                      "assistant_reply","error_message","timestamp"],r))
        c.execute("SELECT messages FROM conversations WHERE call_sid=?;",(log["call_sid"],))
        m=c.fetchone(); log["transcript"]=json.loads(m[0]) if m else []
        c.execute("SELECT vaccine_type,patient_name,desired_date,booked_at FROM bookings WHERE call_sid=?;",(log["call_sid"],))
        b=c.fetchone(); log["booking"]=dict(zip(["vaccine_type","patient_name","desired_date","booked_at"],b)) if b else None
        c.execute("""SELECT from_number,from_city,from_state,from_zip,from_country
                     FROM call_metadata WHERE call_sid=?;""",(log["call_sid"],))
        md=c.fetchone(); log["metadata"]=dict(zip(["from_number","from_city","from_state","from_zip","from_country"],md)) if md else {}
        logs.append(log)
    conn.close(); return JSONResponse({"logs":logs})

@app.get("/api/calls")
async def list_call_sids():
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("SELECT call_sid FROM conversations;"); s=[r[0] for r in c.fetchall()]
    conn.close(); return JSONResponse({"call_sids":s})

@app.get("/api/conversations/{call_sid}")
async def get_conversation(call_sid:str):
    conn=sqlite3.connect(DB_PATH); c=conn.cursor()
    c.execute("SELECT messages FROM conversations WHERE call_sid=?;",(call_sid,))
    row=c.fetchone(); conn.close()
    if not row: return JSONResponse({"error":"CallSid not found"},status_code=404)
    return JSONResponse({"call_sid":call_sid,"messages":json.loads(row[0])})
