import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import api, state, ui  # noqa: E402
from components.constants import SAMPLE_QUESTIONS  # noqa: E402

st.title("AI Agent")
st.caption("Ask about your route plan. The agent re-runs the ML model for your plan (and for any what-if you "
           "mention) and answers from those predictions.")

plan = state.require_plan()
req, res = plan["request"], plan["result"]
ctx = res["context"]
rec = next(r for r in res["routes"] if r["recommended"])
st.markdown(f"**Plan:** {ctx['origin']} → {ctx['destination']} · {ctx['mode']} · {ctx['traveler']} "
            f"· {ctx['time']} · recommended: {rec['name']} ({rec['safety_score']:.0f}/100)")

chat = st.session_state.setdefault("chat", [])


def _queue(q: str) -> None:
    st.session_state["pending_q"] = q


st.markdown("**Try asking**")
cols = st.columns(3)
for i, q in enumerate(SAMPLE_QUESTIONS):
    cols[i % 3].button(q, key=f"sq_{i}", on_click=_queue, args=(q,))


def _show_tools(tools: list) -> None:
    with st.expander("How the agent answered (tools it used)"):
        for t in tools:
            st.markdown(f"**{t['tool']}**  \n{t['summary']}")
            st.json(t["args"], expanded=False)


for msg in chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            chip = (f"LLM · {msg['meta']['model']}" if msg["meta"]["mode"] == "llm" else "Offline agent")
            st.caption(f"{chip} · intent: {msg['meta']['intent']}")
            _show_tools(msg["meta"]["tools_used"])

question = st.chat_input("Ask about routes, risk, trade-offs or what-if conditions...")
question = question or st.session_state.pop("pending_q", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in chat[-6:]]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Re-running the model and thinking..."):
                out = api.chat(req, question, history)
            st.markdown(out["answer"])
            meta = {"mode": out["mode"], "model": out.get("model"), "intent": out["intent"],
                    "tools_used": out["tools_used"]}
            chip = f"LLM · {meta['model']}" if meta["mode"] == "llm" else "Offline agent"
            st.caption(f"{chip} · intent: {meta['intent']}")
            _show_tools(meta["tools_used"])
            chat.append({"role": "user", "content": question})
            chat.append({"role": "assistant", "content": out["answer"], "meta": meta})
        except api.BackendError as exc:
            st.error(str(exc))

if chat and st.button("Clear conversation"):
    st.session_state["chat"] = []
    st.rerun()
st.caption("Answers are predictions from synthetic demo data. " + "The agent does not know real-world conditions "
           "such as live traffic, incidents or road closures.")
ui.disclaimer()
