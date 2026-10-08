from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, render_template, request

from agent_service import get_agent_service
from process import (
    clear_history,
    get_history_summary,
    get_questions_list,
    process_ask,
    process_help_chat,
    translate_text_to_language,
)
from rag import get_rag_store, rebuild_knowledge_base

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def no_cache(response):
    """Prevent browsers from serving stale HTML/JS after code changes."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/help/chat", methods=["POST"])
def help_chat():
  """Route to live agent when online, otherwise Help Assistant AI."""
  data = request.get_json(silent=True) or {}
  question = (data.get("question") or "").strip()
  session_id = data.get("session_id")

  svc = get_agent_service()
  if svc.has_online_agent():
    if not session_id:
      session = svc.create_session()
      session_id = session["id"]
    elif not svc.get_session(session_id):
      session = svc.create_session()
      session_id = session["id"]

    if question:
      result = svc.add_message(session_id, "visitor", question)
      if not result:
        return jsonify({"ok": False, "error": "Could not send message."}), 400

    session = svc.get_session(session_id)
    msg_count = len(session["messages"]) if session else 0
    return jsonify({
      "ok": True,
      "mode": "live",
      "session_id": session_id,
      "message_count": msg_count,
      "answer": None,
    })

  result = process_help_chat(question)
  result["mode"] = "ai"
  result["session_id"] = None
  return jsonify(result)


@app.route("/help/chat/poll")
def help_chat_poll():
    session_id = request.args.get("session_id", "")
    after = request.args.get("after", 0, type=int)
    msgs, total = get_agent_service().visitor_poll(session_id, after)
    return jsonify({"ok": True, "messages": msgs, "total": total})


@app.route("/agent")
def agent_desk():
  return render_template("agent.html")


@app.route("/agent/status")
def agent_status():
  return jsonify(get_agent_service().agent_status())


@app.route("/agent/online", methods=["POST"])
def agent_online():
  data = request.get_json(silent=True) or {}
  info = get_agent_service().register_agent(data.get("name", "Agent"))
  return jsonify({"ok": True, **info})


@app.route("/agent/offline", methods=["POST"])
def agent_offline():
  data = request.get_json(silent=True) or {}
  ok = get_agent_service().unregister(data.get("agent_id", ""))
  if not ok:
    return jsonify({"ok": False, "error": "Not found."}), 404
  return jsonify({"ok": True})


@app.route("/agent/heartbeat", methods=["POST"])
def agent_heartbeat():
  data = request.get_json(silent=True) or {}
  ok = get_agent_service().heartbeat(data.get("agent_id", ""))
  if not ok:
    return jsonify({"ok": False, "error": "Not found."}), 404
  return jsonify({"ok": True})


@app.route("/agent/queue")
def agent_queue():
  agent_id = request.args.get("agent_id", "")
  sessions = get_agent_service().list_sessions(agent_id)
  return jsonify({"ok": True, "sessions": sessions})


@app.route("/agent/sessions/<session_id>")
def agent_session(session_id):
  session = get_agent_service().get_session(session_id)
  if not session:
    return jsonify({"ok": False, "error": "Not found."}), 404
  return jsonify({"ok": True, "session": session})


@app.route("/agent/sessions/<session_id>/claim", methods=["POST"])
def agent_claim(session_id):
  data = request.get_json(silent=True) or {}
  session = get_agent_service().claim(session_id, data.get("agent_id", ""))
  if not session:
    return jsonify({"ok": False, "error": "Cannot claim."}), 400
  return jsonify({"ok": True, "session": session})


@app.route("/agent/sessions/<session_id>/read", methods=["POST"])
def agent_read(session_id):
  data = request.get_json(silent=True) or {}
  ok = get_agent_service().mark_read(session_id, data.get("agent_id", ""))
  if not ok:
    return jsonify({"ok": False, "error": "Cannot mark read."}), 400
  return jsonify({"ok": True})


@app.route("/agent/sessions/<session_id>/messages", methods=["POST"])
def agent_message(session_id):
  data = request.get_json(silent=True) or {}
  svc = get_agent_service()
  if data.get("role") == "agent":
    result = svc.add_message(session_id, "agent", data.get("text", ""), agent_id=data.get("agent_id"))
  else:
    result = svc.add_message(session_id, "visitor", data.get("text", ""))
  if not result:
    return jsonify({"ok": False, "error": "Send failed."}), 400
  return jsonify({"ok": True, **result})


@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        question = data.get("question", "")
    else:
        question = request.args.get("q", "")
    return jsonify(process_ask(question))


@app.route("/translate", methods=["POST"])
def translate_endpoint():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    target = data.get("target", "hi")
    if not text or not text.strip():
        return jsonify({"ok": False, "error": "No text provided."}), 400
    translated, ok = translate_text_to_language(text, target)
    if not ok:
        return jsonify({"ok": False, "translated": text, "error": "Translation unavailable."}), 503
    return jsonify({"ok": True, "translated": translated})


@app.route("/prompt")
def prompt_info():
    from prompt import get_system_prompt
    return jsonify({"system_prompt": get_system_prompt()})


@app.route("/history", methods=["GET", "DELETE"])
def history():
    if request.method == "DELETE":
        clear_history()
        return jsonify({"ok": True, "message": "History cleared."})
    limit = request.args.get("limit", type=int)
    summary = get_history_summary()
    if limit:
        summary["questions"] = summary["questions"][:limit]
        summary["history"] = summary["history"][:limit]
    return jsonify(summary)


@app.route("/history/questions", methods=["GET"])
def history_questions():
    limit = request.args.get("limit", type=int)
    return jsonify({"total": len(get_questions_list()), "questions": get_questions_list(limit=limit)})


@app.route("/rag/rebuild", methods=["POST"])
def rag_rebuild():
    return jsonify({"ok": True, "chunk_count": rebuild_knowledge_base()})


@app.route("/rag/search")
def rag_search():
    q = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 5))
    store = get_rag_store()
    return jsonify({
        "question": q,
        "prompt_context": store.get_prompt_context(q, top_k=top_k),
        "chunks": store.retrieve(q, top_k=top_k),
        "similar_questions": store.similar_questions(q, top_k=top_k),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
