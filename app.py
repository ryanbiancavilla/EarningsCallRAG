import time
import datetime
import streamlit as st
from main import search_docs_with_scores, ask_rag, rag_agent

st.set_page_config(page_title="NVIDIA Conference Explorer", layout="wide")

# -----------------------------------------------------------------------------
# App State & Constants
# -----------------------------------------------------------------------------
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)

SUGGESTIONS = {
    "Summarize all conferences": "Summarize each conference separately.",
    "Metrics in transcripts": "What metrics are present in the transcripts? Give the exact numbers and context.",
    "Trends Across Conferences": "What trends are there across each conference? Be specific.",
    "Future of AI": "Given the 5 conferences, what seems to be the future of AI?",
}

# -----------------------------------------------------------------------------
# Header UI
# -----------------------------------------------------------------------------

title_row = st.container()

with title_row:
    col_title, col_mode = st.columns([3, 2])
    with col_title:
        st.title("NVIDIA Conference Transcript Research Tool", anchor=False)
    with col_mode:
        mode = st.selectbox(
            "Execution Engine",
            ["Agentic RAG", "Standard RAG", "Vector Search"],
            index=0,
            key="rag_mode"
        )

# -----------------------------------------------------------------------------
# Landing Page State vs. Active Chat State
# -----------------------------------------------------------------------------
user_just_asked_initial_question = (
        "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
        "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = (
        user_just_asked_initial_question or user_just_clicked_suggestion
)

has_message_history = (
        "messages" in st.session_state and len(st.session_state.messages) > 0
)

# Landing View (No messages yet)
if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Ask a question about NVIDIA transcripts...", key="initial_question")

        st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=list(SUGGESTIONS.keys()),
            key="selected_suggestion",
        )

    st.stop()

# -----------------------------------------------------------------------------
# Action Bar (Clear/Restart Button)
# -----------------------------------------------------------------------------
user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

with title_row:
    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None


    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

# -----------------------------------------------------------------------------
# Display Chat History
# -----------------------------------------------------------------------------
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()

        st.markdown(message["content"])

# -----------------------------------------------------------------------------
# Process New User Input
# -----------------------------------------------------------------------------
if user_message:
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        question_timestamp = datetime.datetime.now()
        time_diff = question_timestamp - st.session_state.prev_question_timestamp
        st.session_state.prev_question_timestamp = question_timestamp

        if time_diff < MIN_TIME_BETWEEN_REQUESTS:
            time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

        user_message_clean = user_message.replace("'", "")

        # Execute selected pipeline
        with st.spinner("Analyzing transcripts..."):
            selected_engine = st.session_state.get("rag_mode", "Agentic RAG")

            if selected_engine == "Vector Search":
                hits = search_docs_with_scores(user_message_clean, n_results=5)
                response = "### Vector Similarity Search Results\n\n"
                for idx, hit in enumerate(hits, 1):
                    response += (
                        f"**Result {idx}** | *Source: {hit['source']}* | **Cosine Similarity: {hit['score']}**\n\n"
                        f"{hit['text']}\n\n---\n\n"
                    )
            elif selected_engine == "Standard RAG":
                response = ask_rag(user_message_clean, n_results=15)
            else:
                response = rag_agent(user_message_clean)

        with st.container():
            st.markdown(response)

            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})