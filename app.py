"""
AI Freelancing Success Toolkit
Domain: Freelancing / Gig Economy
API: Google Gemini (Free Tier)
Framework: Flask
"""

from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "freelance-ai-secret-2024")

# ── Gemini Configuration ────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "Your_key")
genai.configure(api_key=GEMINI_API_KEY)

# Model config  (Temperature=0.7 for creative proposals; top_p=0.9)
generation_config = genai.GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    max_output_tokens=1024,
)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",          # Free-tier model
    generation_config=generation_config,
    system_instruction=(
        "You are an expert AI assistant for freelancers. "
        "You help generate professional proposals and guide client communication. "
        "Always be concise, professional, and actionable. "
        "Focus specifically on freelancing best practices."
    ),
)

# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/proposal")
def proposal_page():
    return render_template("proposal.html")


@app.route("/chatbot")
def chatbot_page():
    session.setdefault("chat_history", [])
    return render_template("chatbot.html")


# ── API: Generate Proposal ───────────────────────────────────────────────────

@app.route("/api/generate-proposal", methods=["POST"])
def generate_proposal():
    data = request.get_json()

    project_title   = data.get("project_title", "")
    client_name     = data.get("client_name", "Client")
    project_desc    = data.get("project_desc", "")
    skills          = data.get("skills", "")
    budget          = data.get("budget", "")
    timeline        = data.get("timeline", "")
    tone            = data.get("tone", "professional")   # professional / friendly / bold

    if not project_title or not project_desc:
        return jsonify({"error": "Project title and description are required."}), 400

    prompt = f"""
Generate a {tone} freelance project proposal for the following:

Project Title: {project_title}
Client Name: {client_name}
Project Description: {project_desc}
My Skills / Experience: {skills}
Budget Range: {budget}
Timeline: {timeline}

Structure the proposal with these sections:
1. Opening / Hook (2-3 sentences, grab attention)
2. Understanding the Problem (show you understand the client's need)
3. My Proposed Approach (explain your solution / methodology)
4. Why Choose Me (highlight relevant skills and experience)
5. Deliverables & Timeline (clear bullet list)
6. Investment / Pricing (how you frame your cost)
7. Closing / Call to Action

Keep it under 400 words. Use plain readable paragraphs (no excessive markdown symbols).
"""

    try:
        response = model.generate_content(prompt)
        proposal_text = response.text

        # Save to session history
        session.setdefault("proposals", [])
        session["proposals"].append({
            "title": project_title,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "text": proposal_text,
        })
        session.modified = True

        return jsonify({"proposal": proposal_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Chatbot ─────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    # Build or restore chat session
    history = session.get("chat_history", [])

    # Convert stored history to Gemini format
    gemini_history = []
    for turn in history:
        gemini_history.append({"role": turn["role"], "parts": [turn["content"]]})

    chat_session = model.start_chat(history=gemini_history)

    try:
        response = chat_session.send_message(user_message)
        bot_reply = response.text

        # Update session history
        history.append({"role": "user",  "content": user_message})
        history.append({"role": "model", "content": bot_reply})
        # Keep last 20 turns to avoid token overflow
        session["chat_history"] = history[-20:]
        session.modified = True

        return jsonify({"reply": bot_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear-chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    session.modified = True
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
