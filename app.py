"""
AI Interview Coach - web UI (Streamlit).

Run:   streamlit run app.py
Needs: interview_coach.py in the same folder (this file reuses its engine: guardrails, JD analysis,
       question generation, grading, score tracking).
"""
import io
import os
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interview_coach as ic  # noqa: E402

st.set_page_config(page_title="AI Interview Coach", page_icon="🎯", layout="wide")
S = st.session_state

DEFAULTS = {"stage": "setup", "msgs": [], "llm": None, "session": None, "cur": None, "profile": None,
            "bank": None, "jd_hash": "", "jd_text": "", "background": "", "nick": "", "topic": "Mixed",
            "difficulty": "adaptive", "n": 5, "asked": 0, "level": "medium", "vague": 0, "inj": 0,
            "notice": "", "cache": None, "summary": None}
for k, v in DEFAULTS.items():
    S.setdefault(k, v)


# ----------------------------------------------------------------------------- helpers
def say(text: str) -> None:
    S.msgs.append(("assistant", text))


def reset() -> None:
    for k, v in DEFAULTS.items():
        S[k] = v if not isinstance(v, (list, dict)) else type(v)()


def read_upload(f) -> str:
    """Rubric D: read .txt/.md/.pdf/.docx uploads safely; raise ValueError with a friendly message."""
    name = f.name.lower()
    data = f.getvalue()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF support needs `pip install pypdf`. Paste the text instead.")
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        try:
            import docx
        except ImportError:
            raise ValueError("Word support needs `pip install python-docx`. Paste the text instead.")
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    raise ValueError("Unsupported file type. Use .txt, .md, .pdf or .docx, or paste the text.")


def server_api_key() -> str:
    """Rubric B4: the key lives on the host (environment variable or .streamlit/secrets.toml),
    never in the browser and never typed by students."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not key:
        try:
            key = str(st.secrets.get("GEMINI_API_KEY", ""))
        except Exception:  # noqa: BLE001 - no secrets file is normal
            key = ""
    return key.strip()


def fmt_reference(q: dict, covered=()) -> str:
    return "\n".join(f"- {'✅' if i in covered else '⬜'} {label}" for i, (label, _) in enumerate(q["key_points"]))


def fmt_grade(q: dict, g) -> str:
    out = [f"**Score: {g.score}/10**  ·  graded by {g.source}  ·  confidence: {g.confidence}", "", g.feedback,
           "", "**Key points**", fmt_reference(q, g.covered)]
    if g.explanation and g.source == "gemini":
        out += ["", f"**Model answer:** {g.explanation}"]
    if g.tip:
        out += ["", f"💡 {g.tip}"]
    # Rubric C1: be honest about uncertainty in the UI, not just in the report.
    if q.get("generated"):
        out += ["", "ℹ️ This question and its key points were written by AI from your JD. Verify anything "
                    "unfamiliar, and use **Flag** if it looks wrong."]
    if g.confidence == "low" or g.adjusted:
        out += ["", "⚠️ Low-certainty grade - please double-check against a textbook or your notes."]
    if g.adjusted:
        out += ["_The AI's score disagreed with its own checklist, so it was adjusted automatically._"]
    if g.unverified:
        out += ["_A keyword cross-check couldn't confirm every point the AI credited (paraphrasing can cause "
                "this) - treat the score as approximate._"]
    return "\n".join(out)


def ask_next() -> None:
    """Pick and post the next question, or finish the session."""
    sess = S.session
    used = sess.user["daily"].get(ic.today(), 0)
    if S.asked >= S.n:
        return finish()
    if used >= ic.DAILY_QUESTION_LIMIT:
        say(f"You've reached today's free practice limit ({ic.DAILY_QUESTION_LIMIT} questions). Come back tomorrow!")
        return finish()
    picked = sess.pick_question(S.topic, S.level)
    if picked is None:
        say("You've covered every question available for this selection - nicely done.")
        return finish()
    S.cur = picked
    S.asked += 1
    S.vague = S.inj = 0
    t, d, q = picked
    say(f"**Question {S.asked} of {S.n}**  ·  _{t} · {d}_\n\n### {q['question']}")


def finish() -> None:
    sess = S.session
    S.cur = None
    if sess.items:
        avg = sum(i["score"] for i in sess.items) / len(sess.items)
        by_topic = {}
        for i in sess.items:
            by_topic.setdefault(i["topic"], []).append(i["score"])
        S.summary = {"avg": avg, "n": len(sess.items),
                     "by_topic": {t: sum(v) / len(v) for t, v in by_topic.items()}}
    else:
        S.summary = None
    sess.finish(S.topic, S.difficulty)          # saves progress (also prints to the console, which is fine)
    S.stage = "done"


def record_score(q, t, d, g) -> None:
    sess = S.session
    sess.items.append({"id": q["id"], "topic": t, "difficulty": d, "score": g.score,
                       "source": g.source.split(" ")[0]})
    sess.user["daily"][ic.today()] = sess.user["daily"].get(ic.today(), 0) + 1
    if S.difficulty == "adaptive":
        idx = ic.DIFFICULTIES.index(S.level)
        S.level = ic.DIFFICULTIES[min(idx + 1, 2)] if g.score >= 8 else \
            ic.DIFFICULTIES[max(idx - 1, 0)] if g.score <= 4 else S.level


def skip_current(reason: str) -> None:
    t, d, q = S.cur
    say(f"{reason} Here's what a strong answer covers:\n\n{fmt_reference(q)}")
    S.session.events["skipped"] += 1
    S.session.user["daily"][ic.today()] = S.session.user["daily"].get(ic.today(), 0) + 1
    ask_next()


def do_command(cmd: str) -> None:
    sess = S.session
    t, d, q = S.cur
    if cmd == "/hint":
        say(f"💡 Hint: {q['hint']}")
    elif cmd == "/skip":
        skip_current("Skipping.")
    elif cmd == "/quit":
        finish()
    elif cmd == "/score":
        if sess.items:
            avg = sum(i["score"] for i in sess.items) / len(sess.items)
            say(f"This session: {len(sess.items)} question(s) scored, average **{avg:.1f}/10**.")
        else:
            say("No scored answers yet this session.")
    elif cmd == "/flag":
        sess.user.setdefault("flags", []).append(q["id"])
        sess.events["flagged"] += 1
        say("Thanks - flagged as possibly wrong. Please verify it against a trusted source.")
    elif cmd == "/limits":
        say("```\n" + ic.LIMITATIONS + "\n```")
    elif cmd == "/help":
        say("Use the buttons on the left, or type /hint /skip /score /flag /limits /quit. "
            "You can also ask me what a question means.")
    else:
        say("I don't know that command. Try /help.")


def process_answer(text: str) -> None:
    """One chat turn. Mirrors the terminal logic: classify -> guardrail / vague / command / clarify / grade."""
    sess = S.session
    t, d, q = S.cur
    intent, clean, truncated = ic.classify(text, q, sess.extra_terms)
    if truncated:
        sess.events["truncated"] += 1
        say(f"_Your answer was longer than {ic.MAX_ANSWER_CHARS} characters; I'll grade the first part._")

    I = ic.Intent
    if intent in (I.EMPTY, I.VAGUE):                                  # Rubric F6
        sess.events["empty" if intent is I.EMPTY else "vague"] += 1
        S.vague += 1
        if S.vague == 1:
            scaffold = ("Try the STAR structure: Situation, Task, Action, Result." if t == "Behavioural & Fit"
                        else "Try: define the concept, explain how it works, then give a quick example.")
            say(f"I didn't get enough to grade yet - no problem. {scaffold} Even one or two sentences is a good start.")
        elif S.vague == 2:
            say(f"Here's a nudge: {q['hint']}\n\nGive it another go, or press **Skip** to see the model answer.")
        else:
            skip_current("Let's park this one so you keep momentum.")
        return
    if intent is I.INJECTION:                                         # Rubric B3
        sess.events["injection"] += 1
        S.inj += 1
        msg = "I can't change my instructions or role, but I'm glad to keep coaching you. Let's stay with the question."
        say(msg + (" _(Tip: press Skip to move on or End session to finish.)_" if S.inj >= 3 else ""))
        return
    if intent is I.OFF_TOPIC:
        sess.events["off_topic"] += 1
        say(f"That's outside what I can help with - I focus on interview prep for **{sess.role}**. "
            "Back to the question above.")
        return
    if intent is I.META:
        sess.events["meta"] += 1
        say("I'm your AI Interview Coach - an AI assistant built on Google Gemini for a university project. "
            "I keep my setup instructions private, but I'm here to help you practise. Back to the question above.")
        return
    if intent is I.COMMAND:
        return do_command(clean)
    if intent is I.CLARIFY:
        sess.events["clarify"] += 1
        if sess.llm.mode == "gemini":
            prompt = ("TASK: CLARIFY (plain-text reply, max 60 words, do not reveal the answer; if the text is not "
                      "about this interview question, reply with one polite redirect sentence)\n"
                      f"Role context: {sess.role_ctx}\nInterview question: {q['question']}\n"
                      f"<candidate_answer>\n{clean}\n</candidate_answer>")
            try:
                return say(sess.llm.send(prompt, plain=True).strip())
            except (ic.LLMError, ValueError):
                sess.events["api_fallback"] += 1
        return say(f"Good question. {q['hint']} Have a go and I'll give feedback.")

    safe, n_pii = ic.redact_pii(clean)                                # Rubric B4
    if n_pii:
        sess.events["pii_redacted"] += n_pii
        say(f"🔒 Removed {n_pii} personal identifier(s) before grading.")
    g = sess.grade(q, safe, t, d)
    if not g.on_topic:
        sess.events["llm_off_topic"] += 1
        return say("That doesn't look like an answer to the interview question, so I haven't scored it. Try again?")
    record_score(q, t, d, g)
    say(fmt_grade(q, g))
    ask_next()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎯 AI Interview Coach")
    if S.llm is not None:
        if S.llm.mode == "gemini":
            st.success(f"Gemini: {S.llm.model_name}")
        else:
            st.warning(f"Offline mode ({S.llm.reason})")
    if S.stage == "quiz" and S.session:
        sess = S.session
        st.divider()
        st.caption(f"Role: **{sess.role}**")
        st.progress(min(S.asked / S.n, 1.0), text=f"Question {min(S.asked, S.n)} of {S.n}")
        if sess.items:
            st.metric("Session average", f"{sum(i['score'] for i in sess.items) / len(sess.items):.1f} / 10")
            st.bar_chart([i["score"] for i in sess.items], height=140)
        c1, c2 = st.columns(2)
        if c1.button("💡 Hint", use_container_width=True):
            do_command("/hint"); st.rerun()
        if c2.button("⏭ Skip", use_container_width=True):
            do_command("/skip"); st.rerun()
        if c1.button("🚩 Flag", use_container_width=True):
            do_command("/flag"); st.rerun()
        if c2.button("ℹ️ Limits", use_container_width=True):
            do_command("/limits"); st.rerun()
        if st.button("⏹ End session", use_container_width=True):
            finish(); st.rerun()
    if S.stage != "setup":
        st.divider()
        if st.button("↺ Start over", use_container_width=True):
            reset(); st.rerun()
    st.caption("Your JD and answers are sent to Google's Gemini API with personal identifiers masked. "
               "AI-written questions and grades can be wrong - verify important points.")


# ----------------------------------------------------------------------------- stage: setup
if S.stage == "setup":
    st.header("Practise for the role you're applying to")
    st.write("Upload or paste your job description. I'll work out what the role tests, build questions around it, "
             "and coach you through them - answer in text, get instant scored feedback.")
    S.nick = st.text_input("Your nickname", value=S.nick, max_chars=20, placeholder="e.g. Asha",
                           help="Your progress is saved under this nickname, so pick one that is unique to you.")
    up = st.file_uploader("Upload JD (.pdf .docx .txt .md)", type=["pdf", "docx", "txt", "md"])
    pasted = st.text_area("...or paste the JD text here", height=220, key="jd_paste")
    bg = st.text_input("Optional: one line about your background (course, internships, skills)", max_chars=300)
    use_demo = st.checkbox("No JD handy - use the built-in finance demo")
    if st.button("Analyse role ➜", type="primary"):
        raw = ""
        try:
            raw = read_upload(up) if up is not None else pasted
        except Exception as e:  # noqa: BLE001 - any parser error becomes a friendly message
            st.error(f"I couldn't read that file: {e}")
            st.stop()
        text, dropped, truncated = ic.clean_jd(raw)
        if text and not ic.jd_valid(text):
            st.error("That doesn't look like a job description - I need a few sentences about the role's "
                     "responsibilities or requirements.")
            st.stop()
        if not text and not use_demo:
            st.error("Please upload or paste a JD, or tick the demo option.")
            st.stop()
        if dropped:
            st.warning(f"Removed {dropped} line(s) that looked like instructions to an AI.")
        if truncated:
            st.info(f"Your JD is long; I'll use the first {ic.MAX_JD_CHARS} characters.")
        key = server_api_key()
        if key:
            os.environ["GEMINI_API_KEY"] = key
        with st.spinner("Connecting and reading your job description..."):
            S.llm = ic.LLMClient(offline=not key)
            S.background, _ = ic.redact_pii("" if ic.detect_injection(bg or "") else ic.sanitize(bg or "", 300)[0])
            S.cache = ic.JDCache(ic.JD_CACHE_FILE)
            S.jd_text = text
            S.profile, S.jd_hash, S.bank = ic.prepare_role(S.llm, S.cache, text, S.background)
        S.nick = re.sub(r"[^A-Za-z0-9_ -]", "", ic.sanitize(S.nick)[0]).strip()[:20] or "guest"
        S.stage = "config"
        st.rerun()

# ----------------------------------------------------------------------------- stage: config
elif S.stage == "config":
    p = S.profile
    st.header(f"Here's how I read this role")
    c1, c2 = st.columns([2, 1])
    c1.subheader(f"{p['role_title']}")
    c1.caption(f"{p['seniority'].title()} level" + (f"  ·  {p['industry']}" if p.get("industry") else ""))
    if p["key_skills"]:
        c1.write("**Key skills:** " + ", ".join(p["key_skills"]))
    if p.get("summary"):
        c1.write(p["summary"])
    if S.llm.mode != "gemini" and S.jd_text:
        st.warning("Running offline, so I can't write role-specific questions. Using the closest built-in topics. "
                   "Add a Gemini API key on the previous screen to personalise.")
    names = [t["name"] for t in p["topics"]] + ["Behavioural & Fit"]
    st.write("**I'll quiz you on:** " + "; ".join(names))
    S.topic = st.selectbox("Topic", names + ["Mixed"], index=len(names))
    S.difficulty = st.selectbox("Difficulty", ic.DIFFICULTIES + ["adaptive"], index=3,
                                help="Adaptive raises or lowers difficulty based on your last score.")
    S.n = st.slider("Number of questions", 1, 10, 5)
    st.caption(ic.lifetime_summary(ic.ProgressStore(ic.PROGRESS_FILE).user(S.nick), p["role_title"]))
    if st.button("Start interview ➜", type="primary"):
        store = ic.ProgressStore(ic.PROGRESS_FILE)
        S.session = ic.Session(S.llm, store, S.nick, p, S.bank, S.background)
        with st.spinner("Preparing your questions..."):
            ic.fill_bank(S.llm, S.cache, p, S.jd_hash, S.jd_text, S.background, S.bank,
                         names if S.topic == "Mixed" else [S.topic], S.session.events)
        if S.topic != "Mixed" and S.topic not in S.bank:
            S.notice = "I couldn't prepare that topic, so I'll mix the questions I do have."
            S.topic = "Mixed"
        S.level = "medium" if S.difficulty == "adaptive" else S.difficulty
        S.msgs = []
        say(f"Welcome, **{S.nick}**! I'm your AI Interview Coach. Type your answer below like you would in a real "
            "interview. Use the buttons on the left for a hint, skip or flag." + (f"\n\n_{S.notice}_" if S.notice else ""))
        S.stage = "quiz"
        ask_next()
        st.rerun()

# ----------------------------------------------------------------------------- stage: quiz / done
else:
    for role, content in S.msgs:
        with st.chat_message(role):
            st.markdown(content)
    if S.stage == "quiz":
        if prompt := st.chat_input("Type your answer..."):
            S.msgs.append(("user", prompt))
            process_answer(prompt)
            st.rerun()
    else:
        st.success("Session complete")
        if S.summary:
            st.metric("Average score", f"{S.summary['avg']:.1f} / 10", help=f"{S.summary['n']} question(s) scored")
            st.bar_chart(S.summary["by_topic"])
            weakest = min(S.summary["by_topic"], key=S.summary["by_topic"].get)
            st.write(f"**Suggested focus next time:** {weakest}")
        else:
            st.write("No scored answers this time - that's fine. Come back whenever you're ready.")
        ev = dict(S.session.events) if S.session else {}
        if ev:
            st.caption("Session log: " + ", ".join(f"{k}={v}" for k, v in sorted(ev.items())))
        hist = ic.lifetime_summary(S.session.user, S.session.role) if S.session else ""
        st.text(hist)
