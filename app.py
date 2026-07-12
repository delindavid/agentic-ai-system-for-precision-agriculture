"""
╔══════════════════════════════════════════════════════════════════════════╗
║  AgriBot — Advanced Agentic AI for Precision Agriculture                 ║
║  Architecture: LangGraph multi-agent · LLM supervisor · Tool binding     ║
║               Memory · Guardrails · Structured output · Streamlit UI     ║
║                                                                          ║
║  Run:  streamlit run agribot_advanced.py                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ── Standard Library ──────────────────────────────────────────────────────
import pickle
import re
import json
import time
from datetime import datetime
from typing import TypedDict, Annotated, Literal, Optional, Any
from functools import lru_cache

# ── Third-Party ───────────────────────────────────────────────────────────
import cv2
import joblib
import numpy as np
import requests
import streamlit as st
import tensorflow as tf

# ── LangChain / LangGraph ─────────────────────────────────────────────────
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌾 AgriBot Advanced",
    page_icon="🌱",
    layout="wide",
)
 
# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.bubble-user {
    background: #2d6a4f; color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; margin: 6px 0 6px auto;
    max-width: 70%; width: fit-content; word-wrap: break-word;
}
.bubble-bot {
    background: #f0f4f0; color: #1a1a1a;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px; margin: 6px auto 6px 0;
    max-width: 70%; width: fit-content; word-wrap: break-word;
}
.agent-badge {
    font-size: 11px; color: #888; margin-bottom: 2px;
    display: flex; align-items: center; gap: 6px;
}
.agent-tag {
    background: #e8f5e9; color: #2d6a4f;
    border-radius: 10px; padding: 1px 8px;
    font-size: 10px; font-weight: 600; letter-spacing: 0.4px;
}
.result-card {
    background: #f9fbf9; border-left: 4px solid #2d6a4f;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}
.confidence-bar-wrap {
    background: #e0e0e0; border-radius: 6px;
    height: 8px; width: 100%; margin: 6px 0;
}
.confidence-bar {
    background: linear-gradient(90deg, #2d6a4f, #52b788);
    height: 8px; border-radius: 6px; transition: width 0.6s ease;
}
.thinking-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #fff3cd; color: #856404;
    border-radius: 12px; padding: 4px 12px; font-size: 12px;
}
.stButton > button {
    border-radius: 20px; background: #2d6a4f;
    color: white; border: none; padding: 6px 22px;
}
.stButton > button:hover { background: #1b4332; }
.metric-box {
    background: #f0f7f4; border: 1px solid #b7dfc9;
    border-radius: 8px; padding: 10px 14px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Model paths (edit as needed)
# ═══════════════════════════════════════════════════════════════════════════
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.55   # below this the validator flags low confidence

TOMATO_MODEL_PATH   = "tomato_model.keras"
TOMATO_CLASSES_PATH = "tomato_classes.pkl"

RICE_MODEL_PATH   = r"C:\Users\Hey!\Desktop\AI Projects\Portfolio\Agentic AI System for Precision Agriculture\Rice type detection\best_model.keras"
RICE_CLASSES_PATH = r"C:\Users\Hey!\Desktop\AI Projects\Portfolio\Agentic AI System for Precision Agriculture\Rice type detection\classes.pkl"

FERTILIZER_MODEL_PATH   = r"C:\Users\Hey!\Desktop\AI Projects\Portfolio\Agentic AI System for Precision Agriculture\Fertilizer prediciton\fertilizer_model.pkl"
FERTILIZER_ENCODERS_PATH = r"C:\Users\Hey!\Desktop\AI Projects\Portfolio\Agentic AI System for Precision Agriculture\Fertilizer prediciton\encoders.pkl"

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING — @st.cache_resource so they load once
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_llm():
    """Fast generation LLM (OllamaLLM) used for explanations."""
    return OllamaLLM(model="mistral", temperature=0.3)

@st.cache_resource
def load_chat_llm():
    """Chat LLM (ChatOllama) used by the supervisor for structured routing."""
    return ChatOllama(model="mistral", temperature=0.0, format="json")

@st.cache_resource
def load_tomato():
    model = tf.keras.models.load_model(TOMATO_MODEL_PATH)
    with open(TOMATO_CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)
    return model, classes

@st.cache_resource
def load_rice():
    model = tf.keras.models.load_model(RICE_MODEL_PATH)
    with open(RICE_CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)
    return model, classes

@st.cache_resource
def load_fertilizer():
    fert_model = pickle.load(open(FERTILIZER_MODEL_PATH, "rb"))
    encoders   = joblib.load(FERTILIZER_ENCODERS_PATH)
    return fert_model, encoders

# Instantiate models once
llm      = load_llm()
chat_llm = load_chat_llm()

tomato_model, tomato_classes = load_tomato()
rice_model, rice_classes     = load_rice()
fertilizer_model, encoders   = load_fertilizer()

soil_encoder  = encoders["soil"]
crop_encoder  = encoders["crop"]
fert_encoder  = encoders["fertilizer"]
SOIL_CHOICES  = list(soil_encoder.classes_)
CROP_CHOICES  = list(crop_encoder.classes_)

# ═══════════════════════════════════════════════════════════════════════════
#  LANGGRAPH — Shared State
# ═══════════════════════════════════════════════════════════════════════════

class AgricultureState(TypedDict, total=False):
    # LangGraph managed message history
    messages: Annotated[list[BaseMessage], add_messages]

    # Supervisor decision
    intent: Literal["tomato", "rice", "fertilizer", "crop_advice", "general", "unknown"]
    reasoning: str                    # why supervisor chose this intent

    # Raw inputs from UI
    raw_message: str
    image_bytes: Optional[bytes]
    fertilizer_inputs: Optional[dict]

    # Vision outputs
    label: Optional[str]              # disease name or rice type
    confidence_score: Optional[float] # 0.0 – 1.0
    low_confidence: Optional[bool]    # validator flag

    # LLM explanation
    explanation: Optional[str]
    video_url: Optional[str]
    sources: Optional[list[str]]      # cited sources

    # Fertilizer
    fertilizer_name: Optional[str]

    # Errors & retries
    error: Optional[str]
    retry_count: int

    # Final structured response for UI
    response: Optional[dict]

    # Agent trace — list of visited node names
    trace: list[str]

# ═══════════════════════════════════════════════════════════════════════════
#  LANGCHAIN TOOLS — each agent calls these
# ═══════════════════════════════════════════════════════════════════════════

@tool
def run_tomato_model(image_bytes_hex: str) -> dict:
    """Run the tomato disease CNN model on an image provided as hex string."""
    raw = bytes.fromhex(image_bytes_hex)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}
    inp  = np.reshape(cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0, (1, IMG_SIZE, IMG_SIZE, 3))
    pred = tomato_model.predict(inp, verbose=0)
    idx  = int(np.argmax(pred))
    return {
        "label":      tomato_classes[idx],
        "confidence": float(np.max(pred)),
        "all_scores": {c: float(pred[0][i]) for i, c in enumerate(tomato_classes)},
    }

@tool
def run_rice_model(image_bytes_hex: str) -> dict:
    """Run the rice type CNN model on an image provided as hex string."""
    raw = bytes.fromhex(image_bytes_hex)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}
    inp  = np.reshape(cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0, (1, IMG_SIZE, IMG_SIZE, 3))
    pred = rice_model.predict(inp, verbose=0)
    idx  = int(np.argmax(pred))
    return {
        "label":      rice_classes[idx],
        "confidence": float(np.max(pred)),
        "all_scores": {c: float(pred[0][i]) for i, c in enumerate(rice_classes)},
    }

@tool
def run_fertilizer_model(
    temp: float, humidity: float, moisture: float,
    soil_type: str, crop_type: str,
    nitrogen: float, potassium: float, phosphorous: float
) -> dict:
    """Predict the best fertilizer given soil and crop parameters."""
    if soil_type not in SOIL_CHOICES:
        return {"error": f"Unknown soil type '{soil_type}'"}
    if crop_type not in CROP_CHOICES:
        return {"error": f"Unknown crop type '{crop_type}'"}
    data = np.array([[
        temp, humidity, moisture,
        soil_encoder.transform([soil_type])[0],
        crop_encoder.transform([crop_type])[0],
        nitrogen, potassium, phosphorous
    ]])
    name = fert_encoder.inverse_transform(fertilizer_model.predict(data))[0]
    return {"fertilizer": name}

@tool
def get_youtube_video(query: str) -> dict:
    """Search YouTube for a relevant treatment/farming video and return the URL."""
    q       = query.replace(" ", "+") + "+treatment+farming"
    url     = f"https://www.youtube.com/results?search_query={q}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        ids  = re.findall(r"watch\?v=(\S{11})", resp.text)
        if ids:
            return {"url": f"https://www.youtube.com/watch?v={ids[0]}"}
    except Exception:
        pass
    return {"url": None}

@tool
def search_agricultural_info(query: str) -> dict:
    """Search DuckDuckGo for agricultural information and return a short summary."""
    try:
        url  = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        abstract = data.get("AbstractText") or data.get("Answer") or ""
        related  = [r.get("Text","") for r in data.get("RelatedTopics", [])[:3]]
        return {"abstract": abstract, "related": related}
    except Exception:
        return {"abstract": "", "related": []}

@tool
def get_current_time() -> dict:
    """Return the current date and time."""
    now = datetime.now()
    return {
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%d %B %Y"),
        "day":  now.strftime("%A"),
    }

# ═══════════════════════════════════════════════════════════════════════════
#  SUPERVISOR NODE — LLM-based routing (NOT keyword matching)
# ═══════════════════════════════════════════════════════════════════════════

SUPERVISOR_PROMPT = """\
You are the supervisor of an agricultural AI assistant. 
Analyse the user's message and choose the most relevant specialist agent.

Available agents:
- tomato       : Detect diseases in tomato leaves from an uploaded image
- rice         : Identify rice varieties from an uploaded image
- fertilizer   : Recommend fertilizer based on soil and crop parameters
- crop_advice  : Give general farming advice, crop calendar, pest management, irrigation tips
- general      : Anything else (greetings, weather, time, off-topic)

Return ONLY a JSON object:
{{
  "intent": "<one of the agents above>",
  "reasoning": "<one sentence explaining why>",
  "needs_image": <true/false>,
  "needs_form": <true/false>
}}

Do not include any other text."""

def supervisor_node(state: AgricultureState) -> AgricultureState:
    """LLM-powered intent router — no keyword matching."""
    trace = state.get("trace", []) + ["supervisor"]
    msg   = state.get("raw_message", "")

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"User message: {msg}"),
    ]

    try:
        response = chat_llm.invoke(messages)
        content  = response.content if hasattr(response, "content") else str(response)
        # Strip markdown fences if present
        clean    = re.sub(r"```json|```", "", content).strip()
        parsed   = json.loads(clean)
        intent   = parsed.get("intent", "general")
        reasoning = parsed.get("reasoning", "")
        needs_image = parsed.get("needs_image", False)
        needs_form  = parsed.get("needs_form", False)
    except Exception as e:
        intent, reasoning, needs_image, needs_form = "general", str(e), False, False

    # Guard: if user said "tomato"/"rice" but provided no image yet,
    # we still route correctly — the specialist will request the upload.
    return {
        **state,
        "intent":    intent,
        "reasoning": reasoning,
        "trace":     trace,
        "messages":  state.get("messages", []) + [HumanMessage(content=msg)],
    }

# ═══════════════════════════════════════════════════════════════════════════
#  MEMORY NODE — injects conversation history context into state
# ═══════════════════════════════════════════════════════════════════════════

def memory_node(state: AgricultureState) -> AgricultureState:
    """Prepends recent chat context so LLM agents have multi-turn awareness."""
    trace = state.get("trace", []) + ["memory"]
    history = state.get("messages", [])
    # Keep only last 10 messages to stay within context limits
    trimmed = history[-10:]
    return {**state, "messages": trimmed, "trace": trace}

# ═══════════════════════════════════════════════════════════════════════════
#  TOMATO AGENT — calls tool, then LLM for explanation
# ═══════════════════════════════════════════════════════════════════════════

def tomato_agent_node(state: AgricultureState) -> AgricultureState:
    trace = state.get("trace", []) + ["tomato_agent"]
    image_bytes = state.get("image_bytes")

    if not image_bytes:
        return {**state, "trace": trace, "response": {
            "agent":  "tomato_disease_agent",
            "reply":  "🍅 Tomato Disease Detection Agent is ready.\n\nPlease upload a clear photo of the tomato leaf below.",
            "action": "show_tomato_upload",
        }}

    # Call the tool
    result = run_tomato_model.invoke({"image_bytes_hex": image_bytes.hex()})

    if "error" in result:
        return {**state, "trace": trace, "error": result["error"], "response": {
            "agent": "tomato_disease_agent", "error": result["error"]
        }}

    label      = result["label"]
    confidence = result["confidence"]

    # Generate explanation via LLM
    explanation = llm.invoke(
        f"A tomato leaf image shows '{label}' with {confidence*100:.1f}% confidence. "
        f"In 3-4 sentences: (1) describe what this disease or condition is, "
        f"(2) what causes it, (3) the best treatment approach for a farmer."
    )

    # Get related video
    video_result = get_youtube_video.invoke({"query": f"tomato {label} treatment"})
    video_url    = video_result.get("url")

    # Get additional info from web
    web_result = search_agricultural_info.invoke({"query": f"tomato {label} disease management"})
    sources    = [s for s in web_result.get("related", []) if s][:2]

    ai_msg = AIMessage(content=f"Tomato analysis complete: {label} ({confidence*100:.1f}% confidence).")

    return {
        **state,
        "trace":            trace,
        "label":            label,
        "confidence_score": confidence,
        "explanation":      explanation,
        "video_url":        video_url,
        "sources":          sources,
        "messages":         state.get("messages", []) + [ai_msg],
    }

# ═══════════════════════════════════════════════════════════════════════════
#  RICE AGENT
# ═══════════════════════════════════════════════════════════════════════════

def rice_agent_node(state: AgricultureState) -> AgricultureState:
    trace = state.get("trace", []) + ["rice_agent"]
    image_bytes = state.get("image_bytes")

    if not image_bytes:
        return {**state, "trace": trace, "response": {
            "agent":  "rice_type_agent",
            "reply":  "🌾 Rice Type Detection Agent is ready.\n\nPlease upload a clear photo of the rice grain or plant.",
            "action": "show_rice_upload",
        }}

    result = run_rice_model.invoke({"image_bytes_hex": image_bytes.hex()})

    if "error" in result:
        return {**state, "trace": trace, "error": result["error"], "response": {
            "agent": "rice_type_agent", "error": result["error"]
        }}

    label      = result["label"]
    confidence = result["confidence"]

    explanation = llm.invoke(
        f"Rice type identified: '{label}' with {confidence*100:.1f}% confidence. "
        f"In 3-4 sentences: (1) describe this rice variety's key traits, "
        f"(2) optimal growing conditions, (3) primary uses and market value."
    )

    video_result = get_youtube_video.invoke({"query": f"{label} rice cultivation farming"})
    web_result   = search_agricultural_info.invoke({"query": f"{label} rice farming tips"})
    sources      = [s for s in web_result.get("related", []) if s][:2]

    ai_msg = AIMessage(content=f"Rice analysis: {label} ({confidence*100:.1f}% confidence).")

    return {
        **state,
        "trace":            trace,
        "label":            label,
        "confidence_score": confidence,
        "explanation":      explanation,
        "video_url":        video_result.get("url"),
        "sources":          sources,
        "messages":         state.get("messages", []) + [ai_msg],
    }

# ═══════════════════════════════════════════════════════════════════════════
#  FERTILIZER AGENT
# ═══════════════════════════════════════════════════════════════════════════

def fertilizer_agent_node(state: AgricultureState) -> AgricultureState:
    trace  = state.get("trace", []) + ["fertilizer_agent"]
    inputs = state.get("fertilizer_inputs")

    if not inputs:
        return {**state, "trace": trace, "response": {
            "agent":      "fertilizer_agent",
            "reply":      "🌱 Fertilizer Recommendation Agent is ready.\n\nPlease fill in the soil and crop parameters in the form below.",
            "action":     "show_fertilizer_form",
            "soil_types": SOIL_CHOICES,
            "crop_types": CROP_CHOICES,
        }}

    result = run_fertilizer_model.invoke(inputs)

    if "error" in result:
        return {**state, "trace": trace, "error": result["error"], "response": {
            "agent": "fertilizer_agent", "error": result["error"]
        }}

    fert_name = result["fertilizer"]

    explanation = llm.invoke(
        f"For a {inputs['crop_type']} crop on {inputs['soil_type']} soil "
        f"(N={inputs['nitrogen']}, P={inputs['phosphorous']}, K={inputs['potassium']}) "
        f"the recommended fertilizer is '{fert_name}'. "
        f"In 3-4 sentences: explain why this fertilizer is recommended, "
        f"how to apply it, and any precautions the farmer should take."
    )

    video_result = get_youtube_video.invoke({"query": f"{fert_name} fertilizer application {inputs['crop_type']}"})
    ai_msg = AIMessage(content=f"Fertilizer recommendation: {fert_name}.")

    return {
        **state,
        "trace":          trace,
        "fertilizer_name": fert_name,
        "explanation":    explanation,
        "video_url":      video_result.get("url"),
        "messages":       state.get("messages", []) + [ai_msg],
        "response": {
            "agent":      "fertilizer_agent",
            "fertilizer": fert_name,
            "explanation": explanation,
            "video_url":  video_result.get("url"),
        },
    }

# ═══════════════════════════════════════════════════════════════════════════
#  CROP ADVICE AGENT — RAG-style with web search + LLM
# ═══════════════════════════════════════════════════════════════════════════

CROP_ADVISOR_SYSTEM = """\
You are an expert agronomist with 20 years of experience.
Given the farmer's question and any web search context provided,
give a detailed, practical, actionable answer.
Use bullet points for steps. Be specific about quantities and timings.
Always end with one key warning or caution relevant to the topic."""

def crop_advice_agent_node(state: AgricultureState) -> AgricultureState:
    trace = state.get("trace", []) + ["crop_advice_agent"]
    msg   = state.get("raw_message", "")

    # Tool call 1: search for grounding context
    web_result = search_agricultural_info.invoke({"query": msg})
    context    = web_result.get("abstract", "")
    related    = web_result.get("related", [])

    # Tool call 2: YouTube resource
    video_result = get_youtube_video.invoke({"query": msg + " farming guide"})

    # Build grounded prompt
    grounding = f"\n\nWeb context: {context}" if context else ""
    prompt    = (
        f"{CROP_ADVISOR_SYSTEM}"
        f"{grounding}\n\n"
        f"Farmer's question: {msg}"
    )
    answer = llm.invoke(prompt)

    sources = [r for r in related if r][:3]
    ai_msg  = AIMessage(content=answer)

    return {
        **state,
        "trace":       trace,
        "explanation": answer,
        "video_url":   video_result.get("url"),
        "sources":     sources,
        "messages":    state.get("messages", []) + [ai_msg],
        "response": {
            "agent":   "crop_advice_agent",
            "reply":   answer,
            "video_url": video_result.get("url"),
            "sources": sources,
        },
    }

# ═══════════════════════════════════════════════════════════════════════════
#  GENERAL LLM AGENT — conversational fallback with memory
# ═══════════════════════════════════════════════════════════════════════════

GENERAL_SYSTEM = """\
You are AgriBot, a friendly AI assistant for farmers and agricultural professionals.
You have expertise in farming, crops, soil, weather, and rural life.
Answer concisely and helpfully. If the question is completely off-topic from
agriculture, answer briefly but steer the conversation back to farming topics."""

def general_llm_node(state: AgricultureState) -> AgricultureState:
    trace = state.get("trace", []) + ["general_llm"]
    msg   = state.get("raw_message", "")

    # Check for time query
    if any(w in msg.lower() for w in ["time", "date", "today", "day"]):
        t_result = get_current_time.invoke({})
        reply    = (
            f"📅 Today is {t_result['day']}, {t_result['date']}.\n"
            f"🕐 Current time: {t_result['time']}"
        )
    else:
        # Multi-turn: include recent history
        history = state.get("messages", [])[-6:]
        history_text = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'AgriBot'}: {m.content}"
            for m in history
        ])
        prompt = f"{GENERAL_SYSTEM}\n\nConversation so far:\n{history_text}\n\nUser: {msg}"
        reply  = llm.invoke(prompt)

    ai_msg = AIMessage(content=reply)
    return {
        **state,
        "trace":    trace,
        "messages": state.get("messages", []) + [ai_msg],
        "response": {"agent": "general_llm_agent", "reply": reply},
    }

# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATOR / GUARDRAIL NODE
# ═══════════════════════════════════════════════════════════════════════════

def validator_node(state: AgricultureState) -> AgricultureState:
    """
    Checks:
    1. If a vision prediction has confidence below threshold → warn user
    2. If an error exists → log it and build error response
    3. If retry_count exceeded → give up gracefully
    """
    trace = state.get("trace", []) + ["validator"]
    error = state.get("error")

    if error:
        retry = state.get("retry_count", 0)
        if retry >= 2:
            return {**state, "trace": trace, "response": {
                "agent": "validator",
                "error": f"Could not complete the request after retries: {error}",
            }}
        return {**state, "trace": trace, "retry_count": retry + 1}

    # Low-confidence warning
    conf = state.get("confidence_score")
    if conf is not None and conf < CONFIDENCE_THRESHOLD:
        reply = state.get("response", {})
        reply["warning"] = (
            f"⚠️ Confidence is low ({conf*100:.1f}%). "
            "Please upload a clearer, well-lit image for a more reliable result."
        )
        return {**state, "trace": trace, "low_confidence": True, "response": reply}

    return {**state, "trace": trace}

# ═══════════════════════════════════════════════════════════════════════════
#  RESPONSE FORMATTER NODE
# ═══════════════════════════════════════════════════════════════════════════

def formatter_node(state: AgricultureState) -> AgricultureState:
    """
    Builds the final structured response dict consumed by the Streamlit UI.
    Also appends metadata: intent, reasoning, trace.
    """
    trace    = state.get("trace", []) + ["formatter"]
    response = state.get("response") or {}

    intent  = state.get("intent", "unknown")
    label   = state.get("label")
    conf    = state.get("confidence_score")
    expl    = state.get("explanation")
    video   = state.get("video_url")
    sources = state.get("sources", [])
    fert    = state.get("fertilizer_name")

    # Enrich response based on intent
    if intent in ("tomato", "rice") and label and not response.get("error"):
        response.update({
            "type":        "vision_result",
            "intent":      intent,
            "label":       label,
            "confidence":  conf,
            "explanation": expl,
            "video_url":   video,
            "sources":     sources,
        })
    elif intent == "fertilizer" and fert and not response.get("error"):
        response.update({
            "type":        "fertilizer_result",
            "fertilizer":  fert,
            "explanation": expl,
            "video_url":   video,
        })

    # Always attach trace for sidebar debug panel
    response["trace"]     = trace
    response["reasoning"] = state.get("reasoning", "")

    return {**state, "trace": trace, "response": response}

# ═══════════════════════════════════════════════════════════════════════════
#  UNKNOWN FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

def unknown_node(state: AgricultureState) -> AgricultureState:
    trace = state.get("trace", []) + ["unknown"]
    return {**state, "trace": trace, "response": {
        "agent": "fallback",
        "reply": (
            "I'm not sure how to handle that. "
            "I can help with tomato disease detection, rice type identification, "
            "fertilizer recommendations, crop advice, and general farming questions. "
            "What would you like to know?"
        ),
    }}

# ═══════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def route_supervisor(state: AgricultureState) -> str:
    intent_map = {
        "tomato":       "tomato_agent",
        "rice":         "rice_agent",
        "fertilizer":   "fertilizer_agent",
        "crop_advice":  "crop_advice_agent",
        "general":      "general_llm",
        "unknown":      "unknown",
    }
    return intent_map.get(state.get("intent", "unknown"), "unknown")

def route_after_vision(state: AgricultureState) -> str:
    """After vision agents: skip to formatter if upload-prompt was sent."""
    if state.get("response") and not state.get("label"):
        return "formatter"
    return "validator"

def route_after_validator(state: AgricultureState) -> str:
    """If error and retries left, loop back to appropriate agent."""
    if state.get("error") and state.get("retry_count", 0) < 2:
        intent = state.get("intent", "unknown")
        return route_supervisor(state)  # re-route to same agent
    return "formatter"

# ═══════════════════════════════════════════════════════════════════════════
#  LANGGRAPH — Build & Compile Graph
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_graph():
    g = StateGraph(AgricultureState)

    # Nodes
    g.add_node("memory",            memory_node)
    g.add_node("supervisor",        supervisor_node)
    g.add_node("tomato_agent",      tomato_agent_node)
    g.add_node("rice_agent",        rice_agent_node)
    g.add_node("fertilizer_agent",  fertilizer_agent_node)
    g.add_node("crop_advice_agent", crop_advice_agent_node)
    g.add_node("general_llm",       general_llm_node)
    g.add_node("unknown",           unknown_node)
    g.add_node("validator",         validator_node)
    g.add_node("formatter",         formatter_node)

    # Entry: always pass through memory first, then supervisor
    g.set_entry_point("memory")
    g.add_edge("memory", "supervisor")

    # Supervisor → specialist agents
    g.add_conditional_edges("supervisor", route_supervisor)

    # Vision agents → validator (with early-exit for upload prompts)
    g.add_conditional_edges("tomato_agent",  route_after_vision)
    g.add_conditional_edges("rice_agent",    route_after_vision)

    # Other agents → validator directly
    g.add_edge("fertilizer_agent",  "validator")
    g.add_edge("crop_advice_agent", "validator")
    g.add_edge("general_llm",       "validator")
    g.add_edge("unknown",           "formatter")

    # Validator → formatter (or retry loop)
    g.add_conditional_edges("validator", route_after_validator)

    # Formatter → END
    g.add_edge("formatter", END)

    return g.compile()

agriculture_graph = build_graph()

# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH INVOCATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def run_graph(
    message: str,
    image_bytes: Optional[bytes] = None,
    fertilizer_inputs: Optional[dict] = None,
    history: Optional[list[BaseMessage]] = None,
) -> dict:
    initial: AgricultureState = {
        "raw_message":       message,
        "image_bytes":       image_bytes,
        "fertilizer_inputs": fertilizer_inputs,
        "messages":          history or [],
        "retry_count":       0,
        "trace":             [],
    }
    try:
        final = agriculture_graph.invoke(initial)
        return final.get("response", {"error": "No response generated."})
    except Exception as exc:
        return {"error": str(exc), "agent": "system"}

# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
if "lc_messages"     not in st.session_state: st.session_state.lc_messages     = []
if "pending_action"  not in st.session_state: st.session_state.pending_action  = None
if "soil_choices"    not in st.session_state: st.session_state.soil_choices    = SOIL_CHOICES
if "crop_choices"    not in st.session_state: st.session_state.crop_choices    = CROP_CHOICES
if "agent_traces"    not in st.session_state: st.session_state.agent_traces    = []

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌾 AgriBot Advanced")
    st.caption("LangGraph · LLM Supervisor · Tool Agents · Memory")
    st.divider()
    st.markdown("""
**Agents available**

🍅 **Tomato disease** — upload a leaf photo  
🌾 **Rice type** — upload a grain/plant photo  
🌱 **Fertilizer** — fill the soil/crop form  
🌿 **Crop advisor** — ask any farming question  
🤖 **General AI** — anything else  
""")
    st.divider()

    # Agent trace panel
    if st.session_state.agent_traces:
        st.markdown("**Last agent trace**")
        last_trace = st.session_state.agent_traces[-1]
        for i, node in enumerate(last_trace):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
                f'<span style="color:#888;font-size:11px">{i+1}.</span>'
                f'<span style="background:#e8f5e9;color:#2d6a4f;border-radius:8px;'
                f'padding:1px 8px;font-size:11px">{node}</span></div>',
                unsafe_allow_html=True
            )
        st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history   = []
        st.session_state.lc_messages    = []
        st.session_state.pending_action = None
        st.session_state.agent_traces   = []
        st.rerun()

    st.caption("Powered by LangGraph · ChatOllama · TensorFlow")

# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h2 style='color:#2d6a4f;margin-bottom:0'>🌱 AgriBot — Advanced Multi-Agent System</h2>"
    "<p style='color:#666;margin-top:2px'>LangGraph · LLM Supervisor · Tool Calling · Memory · Guardrails</p>",
    unsafe_allow_html=True,
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════
#  RENDER CHAT HISTORY
# ═══════════════════════════════════════════════════════════════════════════

def confidence_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#52b788" if score >= 0.75 else "#f4a261" if score >= 0.55 else "#e76f51"
    return (
        f'<div style="margin:6px 0">'
        f'<div style="font-size:12px;color:#555;margin-bottom:3px">Confidence: {pct}%</div>'
        f'<div style="background:#e0e0e0;border-radius:6px;height:8px;width:100%">'
        f'<div style="background:{color};width:{pct}%;height:8px;border-radius:6px"></div>'
        f'</div></div>'
    )

def render_chat():
    for msg in st.session_state.chat_history:
        role    = msg["role"]
        content = msg["content"]
        meta    = msg.get("meta", {})

        if role == "user":
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end">'
                f'<div class="bubble-user">🧑 {content}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            agent  = meta.get("agent", "agent")
            reason = meta.get("reasoning", "")
            reason_html = f'<span style="color:#aaa">· {reason[:60]}…</span>' if reason else ""
            st.markdown(
                f'<div class="agent-badge">'
                f'🤖 <span class="agent-tag">{agent}</span>'
                f'{reason_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="bubble-bot">{content}</div>',
                unsafe_allow_html=True,
            )

            # ── Vision result card (tomato / rice) ───────────────
            if meta.get("type") == "vision_result":
                conf  = meta.get("confidence", 0)
                label = meta.get("label", "")
                intent = meta.get("intent", "")
                emoji = "🍅" if intent == "tomato" else "🌾"
                expl  = meta.get("explanation", "")
                warning = meta.get("warning", "")
                warning_style = 'color:#856404;font-size:12px;background:#fff3cd;border-radius:6px;padding:5px 10px;margin:6px 0'
                warning_html = f'<div style="{warning_style}">{warning}</div>' if warning else ""
                st.markdown(
                    f'<div class="result-card">'
                    f'<div style="font-weight:700;font-size:16px;color:#2d6a4f">{emoji} {label}</div>'
                    f'{confidence_bar(conf)}'
                    f'{warning_html}'
                    f'<hr style="margin:8px 0">'
                    f'<p style="font-size:14px">{expl}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if meta.get("video_url"):
                    st.video(meta["video_url"])
                if meta.get("sources"):
                    with st.expander("📚 Related information"):
                        for s in meta["sources"]:
                            st.markdown(f"- {s}")

            # ── Fertilizer result ─────────────────────────────────
            elif meta.get("type") == "fertilizer_result":
                fert = meta.get("fertilizer", "")
                expl = meta.get("explanation", "")
                st.success(f"🌿 Recommended Fertilizer: **{fert}**")
                if expl:
                    with st.expander("Why this fertilizer?"):
                        st.write(expl)
                if meta.get("video_url"):
                    st.video(meta["video_url"])

render_chat()

# ═══════════════════════════════════════════════════════════════════════════
#  DYNAMIC WIDGETS
# ═══════════════════════════════════════════════════════════════════════════

# ── Tomato Upload ────────────────────────────────────────────────────────
if st.session_state.pending_action == "show_tomato_upload":
    st.markdown("#### 📤 Tomato Leaf Image")
    c_img, c_prev = st.columns([3, 2])
    with c_img:
        uploaded = st.file_uploader(
            "Upload a clear photo of the tomato leaf",
            type=["jpg", "jpeg", "png"], key="tomato_up"
        )
    with c_prev:
        if uploaded:
            st.image(uploaded, caption="Preview", use_container_width=True)

    ca, cb = st.columns([2, 8])
    with ca:
        detect = st.button("🔍 Detect Disease", use_container_width=True)
    with cb:
        if st.button("❌ Cancel"):
            st.session_state.pending_action = None
            st.rerun()

    if detect and uploaded:
        with st.spinner("Running tomato disease model…"):
            data = run_graph(
                "tomato",
                image_bytes=uploaded.getvalue(),
                history=st.session_state.lc_messages,
            )
        if "error" in data:
            st.session_state.chat_history.append({
                "role": "bot", "content": f"❌ {data['error']}",
                "meta": {"agent": "tomato_disease_agent"},
            })
        else:
            st.session_state.agent_traces.append(data.get("trace", []))
            st.session_state.chat_history.append({
                "role":    "bot",
                "content": f"Analysis complete: **{data.get('label','')}**",
                "meta": {
                    "agent":      "tomato_disease_agent",
                    "type":       "vision_result",
                    "intent":     "tomato",
                    "label":      data.get("label"),
                    "confidence": data.get("confidence"),
                    "explanation":data.get("explanation"),
                    "video_url":  data.get("video_url"),
                    "sources":    data.get("sources", []),
                    "warning":    data.get("warning", ""),
                    "reasoning":  data.get("reasoning", ""),
                },
            })
        st.session_state.pending_action = None
        st.rerun()

# ── Rice Upload ──────────────────────────────────────────────────────────
elif st.session_state.pending_action == "show_rice_upload":
    st.markdown("#### 📤 Rice Image")
    c_img, c_prev = st.columns([3, 2])
    with c_img:
        uploaded = st.file_uploader(
            "Upload a photo of the rice grain or plant",
            type=["jpg", "jpeg", "png"], key="rice_up"
        )
    with c_prev:
        if uploaded:
            st.image(uploaded, caption="Preview", use_container_width=True)

    ca, cb = st.columns([2, 8])
    with ca:
        detect = st.button("🔍 Detect Rice Type", use_container_width=True)
    with cb:
        if st.button("❌ Cancel"):
            st.session_state.pending_action = None
            st.rerun()

    if detect and uploaded:
        with st.spinner("Running rice type model…"):
            data = run_graph(
                "rice",
                image_bytes=uploaded.getvalue(),
                history=st.session_state.lc_messages,
            )
        if "error" in data:
            st.session_state.chat_history.append({
                "role": "bot", "content": f"❌ {data['error']}",
                "meta": {"agent": "rice_type_agent"},
            })
        else:
            st.session_state.agent_traces.append(data.get("trace", []))
            st.session_state.chat_history.append({
                "role":    "bot",
                "content": f"Rice identified: **{data.get('label','')}**",
                "meta": {
                    "agent":      "rice_type_agent",
                    "type":       "vision_result",
                    "intent":     "rice",
                    "label":      data.get("label"),
                    "confidence": data.get("confidence"),
                    "explanation":data.get("explanation"),
                    "video_url":  data.get("video_url"),
                    "sources":    data.get("sources", []),
                    "warning":    data.get("warning", ""),
                    "reasoning":  data.get("reasoning", ""),
                },
            })
        st.session_state.pending_action = None
        st.rerun()

# ── Fertilizer Form ──────────────────────────────────────────────────────
elif st.session_state.pending_action == "show_fertilizer_form":
    st.markdown("#### 🌱 Fertilizer Recommendation Form")

    with st.form("fert_form"):
        st.markdown("**Environmental conditions**")
        ca, cb, cc = st.columns(3)
        with ca:
            temp     = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
            humidity = st.number_input("Humidity (%)",     min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            moisture = st.number_input("Soil Moisture (%)",min_value=0.0, max_value=100.0, value=40.0, step=1.0)
        with cb:
            st.markdown("**Crop & soil**")
            soil_type = st.selectbox("Soil Type", options=st.session_state.soil_choices)
            crop_type = st.selectbox("Crop Type", options=st.session_state.crop_choices)
        with cc:
            st.markdown("**Nutrient levels (mg/kg)**")
            nitrogen    = st.number_input("Nitrogen (N)",    min_value=0.0, max_value=200.0, value=30.0, step=1.0)
            potassium   = st.number_input("Potassium (K)",   min_value=0.0, max_value=200.0, value=20.0, step=1.0)
            phosphorous = st.number_input("Phosphorous (P)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)

        submitted = st.form_submit_button("🌿 Get Fertilizer Recommendation", use_container_width=True)

    if submitted:
        payload = {
            "temp": temp, "humidity": humidity, "moisture": moisture,
            "soil_type": soil_type, "crop_type": crop_type,
            "nitrogen": nitrogen, "potassium": potassium, "phosphorous": phosphorous,
        }
        with st.spinner("Analysing soil parameters and predicting fertilizer…"):
            data = run_graph(
                "fertilizer",
                fertilizer_inputs=payload,
                history=st.session_state.lc_messages,
            )
        if "error" in data:
            st.session_state.chat_history.append({
                "role": "bot", "content": f"❌ {data['error']}",
                "meta": {"agent": "fertilizer_agent"},
            })
        else:
            st.session_state.agent_traces.append(data.get("trace", []))
            st.session_state.chat_history.append({
                "role":    "bot",
                "content": f"Fertilizer recommendation ready.",
                "meta": {
                    "agent":       "fertilizer_agent",
                    "type":        "fertilizer_result",
                    "fertilizer":  data.get("fertilizer"),
                    "explanation": data.get("explanation"),
                    "video_url":   data.get("video_url"),
                    "reasoning":   data.get("reasoning", ""),
                },
            })
        st.session_state.pending_action = None
        st.rerun()

    if st.button("❌ Cancel"):
        st.session_state.pending_action = None
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
#  CHAT INPUT BAR
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
col_in, col_btn = st.columns([8, 1])
with col_in:
    user_input = st.text_input(
        label="",
        placeholder="Ask about tomato disease, rice type, fertilizer, crop advice, or anything farming-related…",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("Send ➤", use_container_width=True)

if send and user_input.strip():
    raw = user_input.strip()

    # Add to UI history
    st.session_state.chat_history.append({"role": "user", "content": raw})

    with st.spinner("AgriBot is thinking…"):
        data = run_graph(raw, history=st.session_state.lc_messages)

    # Update LangChain message memory
    st.session_state.lc_messages.append(HumanMessage(content=raw))
    if data.get("reply") or data.get("explanation"):
        bot_text = data.get("reply") or data.get("explanation", "")
        st.session_state.lc_messages.append(AIMessage(content=bot_text))

    # Save trace
    if data.get("trace"):
        st.session_state.agent_traces.append(data["trace"])

    action = data.get("action")
    reply  = data.get("reply", "")
    agent  = data.get("agent", "agent")

    # If action needed, show widget and skip adding a duplicate message
    if action:
        st.session_state.pending_action = action
        st.session_state.chat_history.append({
            "role": "bot", "content": reply,
            "meta": {"agent": agent, "reasoning": data.get("reasoning", "")},
        })
        if action == "show_fertilizer_form":
            st.session_state.soil_choices = data.get("soil_types", SOIL_CHOICES)
            st.session_state.crop_choices = data.get("crop_types", CROP_CHOICES)
    elif reply:
        st.session_state.chat_history.append({
            "role": "bot", "content": reply,
            "meta": {
                "agent":     agent,
                "reasoning": data.get("reasoning", ""),
                "type":      data.get("type", ""),
            },
        })
    elif "error" in data:
        st.session_state.chat_history.append({
            "role": "bot", "content": f"❌ {data['error']}",
            "meta": {"agent": agent},
        })

    st.rerun()