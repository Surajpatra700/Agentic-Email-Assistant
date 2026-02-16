import os
import uuid
from typing import TypedDict, Annotated, Literal
import operator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError
import logging

app = FastAPI(title="LangGraph Mail Agent with HITL (manager-only)")

# -------- LLM setup (replace with your model if needed) --------
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.6)

# -------- State definition --------
class MailState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    sender_email: str | None
    manager_email: str | None
    subject: str | None
    body: str | None
    approved: bool | None

    # which field we are currently waiting for (HITL)
    awaiting: Literal["manager_email", "approval", None]

# -------- SMTP config (use .env) --------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

if not SMTP_USER or not SMTP_PASSWORD:
    # It's better to fail early than attempt to send without creds.
    raise RuntimeError("Please set SMTP_USER and SMTP_PASSWORD in environment/.env")

# -------- Utilities --------
def is_valid_email(addr: str) -> bool:
    try:
        validate_email(addr)
        return True
    except EmailNotValidError:
        return False

def send_email(smtp_from: str, recipient: str, subject: str, body: str, reply_to: str | None = None) -> None:
    """
    Send email using SMTP_USER credentials. smtp_from is informational; actual From will be SMTP_USER.
    reply_to: put the real user's email in Reply-To so replies go to them.
    """
    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    if reply_to:
        msg["Reply-To"] = reply_to

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)

# -- helper to robustly extract content from messages --
def _get_message_content(msg) -> str | None:
    """
    Return message content whether msg is a langchain message object (has .content)
    or a plain dict with a 'content' key.
    """
    # langchain message classes expose `.content`
    if hasattr(msg, "content"):
        return getattr(msg, "content", None)
    # sometimes messages are serialized dicts
    if isinstance(msg, dict):
        # common keys: "content", sometimes nested
        return msg.get("content")
    return None

# -------- LangGraph nodes (only manager email is requested) --------
def ask_manager_email(state: MailState):
    """
    Capture manager email if missing.
    Always try to read latest human input when manager_email is None.
    """

    # If already have manager email, move on
    if state.get("manager_email"):
        return {}

    # Try to read the latest human message as manager email
    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", None)
        if not content:
            continue

        val = content.strip()
        if is_valid_email(val):
            return {
                "manager_email": val,
                "awaiting": None
            }
        else:
            interrupt(
                "That doesn't look like a valid email. "
                "Please share your manager’s email ID (e.g. manager@company.com)."
            )
            return {"awaiting": "manager_email"}

    # No human input yet → ask for it
    interrupt("Please share your manager’s email ID.")
    return {"awaiting": "manager_email"}


def draft_email(state: MailState):
    """
    Drafts subject and body using the model. Robust parsing with fallback.
    """
    user_request = state["messages"][0].content if state["messages"] else ""
    prompt = f"""
Write a professional email.

Requirements:
- Include a clear subject
- Polite and formal tone
- Request: {user_request}

Return format (exactly):
SUBJECT:
<one-line subject here>

BODY:
<email body here>
"""
    response = model.invoke([
        SystemMessage(content="You are a professional corporate email writer."),
        HumanMessage(content=prompt)
    ])

    raw = getattr(response, "content", str(response))

    # Robust parsing: try to split on "BODY:", else fallback
    try:
        subject_part, body_part = raw.split("BODY:", 1)
        subject = subject_part.replace("SUBJECT:", "").strip()
        body = body_part.strip()
        if not subject:
            # Use first line of body as fallback subject
            subject = (body.splitlines()[0] if body else "No subject").strip()[:80]
    except Exception:
        # fallback: whole output as body
        body = raw.strip()
        subject = (body.splitlines()[0] if body else "No subject")[:80]

    # Ensure awaiting cleared
    return {"subject": subject, "body": body, "awaiting": None}

def capture_approval(state: MailState):
    """
    Capture explicit yes/no approval from the latest human input.
    """

    if state.get("approved") is not None:
        return {}

    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", None)
        if not content:
            continue

        decision = content.lower().strip()

        if decision in ("yes", "y", "send", "ok", "sure"):
            return {"approved": True, "awaiting": None}

        if decision in ("no", "n", "cancel", "don't send", "do not send"):
            return {"approved": False, "awaiting": None}

        # ignore unrelated text (important)
        continue

    interrupt("Do you want me to send this email? (yes/no)")
    return {"awaiting": "approval"}

def send_email_node(state: MailState):
    """
    Send email using SMTP_USER as From and user's email as Reply-To for replies.
    """
    try:
        # manager_email must be present and valid
        manager = state.get("manager_email")
        if not manager or not is_valid_email(manager):
            return {"messages": [AIMessage(content="⚠️ Manager email missing or invalid. Aborting send.")]}
        send_email(
            smtp_from=state.get("sender_email") or SMTP_USER,
            recipient=manager,
            subject=state.get("subject") or "(no subject)",
            body=state.get("body") or "",
            reply_to=state.get("sender_email")
        )
        return {"messages": [AIMessage(content="✅ Email sent successfully.")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"⚠️ Failed to send email: {e}")]}  # keep graph deterministic

def should_send(state: MailState) -> Literal["send_email", END]:
    return "send_email" if state.get("approved") else END

# -------- Build the graph --------
builder = StateGraph(MailState)

# Note: we do NOT add ask_sender_email node; sender is SMTP_USER
builder.add_node("ask_manager_email", ask_manager_email)
builder.add_node("draft_email", draft_email)
builder.add_node("capture_approval", capture_approval)
builder.add_node("send_email", send_email_node)

builder.add_edge(START, "ask_manager_email")
builder.add_edge("ask_manager_email", "draft_email")
builder.add_edge("draft_email", "capture_approval")

builder.add_conditional_edges(
    "capture_approval",
    should_send,
    ["send_email", END]
)

builder.add_edge("send_email", END)

mail_agent = builder.compile()

# -------- In-memory sessions (dev only) --------
SESSIONS: dict[str, dict] = {}

# -------- Pydantic models --------
class StartRequest(BaseModel):
    user_request: str

class ContinueRequest(BaseModel):
    session_id: str
    user_input: str

# -------- Helper functions for interrupt normalization --------
def normalize_interrupt_payload(payload) -> str | None:
    if payload is None:
        return None
    return str(payload)

def strip_interrupt_from_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "__interrupt__"}

# -------- Endpoints --------
@app.post("/start")
def start_mail_agent(req: StartRequest):
    session_id = str(uuid.uuid4())

    # Set sender_email to SMTP_USER so we don't ask for it
    initial_state: MailState = {
        "messages": [HumanMessage(content=req.user_request)],
        "sender_email": SMTP_USER,
        "manager_email": None,
        "subject": None,
        "body": None,
        "approved": None,
        "awaiting": None
    }
    print(f"sender email: {SMTP_USER}")

    result = mail_agent.invoke(initial_state)

    # Normalize and store state without interrupt
    interrupt_text = normalize_interrupt_payload(result.get("__interrupt__"))
    state_only = strip_interrupt_from_result(result)
    SESSIONS[session_id] = state_only

    return {"session_id": session_id, "message": interrupt_text}

@app.post("/continue")
def continue_mail_agent(req: ContinueRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id")
    print(f"Continuing session: {req.session_id}")

    previous_state = SESSIONS[req.session_id].copy()
    print(f"Previous state before adding user input: \n{previous_state}")
    previous_state.pop("__interrupt__", None)
    print(f"Previous state after popping interrupt: \n{previous_state}")

    # Append user's reply to messages
    prev_msgs = previous_state.get("messages", [])
    print(f"Previous messages count: {len(prev_msgs)} & prv_msg: \n{prev_msgs}")

    new_msgs = prev_msgs + [HumanMessage(content=req.user_input)]
    print(f"New Msg: \n{new_msgs}")

    new_state = {**previous_state, "messages": new_msgs}
    print(f"New state before invoke: \n{new_state}")

    result = mail_agent.invoke(new_state)
    print(f"Result after invoke: \n{result}")

    # Save cleaned state (without interrupt)
    interrupt_text = normalize_interrupt_payload(result.get("__interrupt__"))
    state_only = strip_interrupt_from_result(result)
    SESSIONS[req.session_id] = state_only

    if interrupt_text is not None:
        return {"session_id": req.session_id, "message": interrupt_text}

    # If completed, return final info (last AI message + current state)
    last_msgs = result.get("messages", [])
    last_content = None
    if last_msgs:
        last = last_msgs[-1]
        last_content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else str(last))

    return {
        "session_id": req.session_id,
        "message": "Flow completed successfully.",
        "final_message": last_content,
        "state": state_only
    }