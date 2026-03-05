import streamlit as st
from agent import run_research_agent
# Import our agent function from agent.py
# This is why we kept them separate — clean import!

from dotenv import load_dotenv
load_dotenv()

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="centered"
)

# ================================================================
# SESSION STATE
# ================================================================

if "report" not in st.session_state:
    st.session_state.report = None
    # Store the generated report
    # None means no report generated yet

if "topic" not in st.session_state:
    st.session_state.topic = ""
    # Store the last researched topic

# ================================================================
# UI: HEADER
# ================================================================

st.title("🔍 AI Research Agent")
st.write("Give me any topic and I'll research it autonomously and write a report!")

with st.expander("ℹ️ How this works — Click to expand"):
    st.write("""
    **This is an Agentic AI system built with LangGraph:**

    1. You type a research topic
    2. Agent plans what to search for
    3. Agent searches the internet using Tavily
    4. Agent evaluates if it found enough information
    5. If not enough → searches again with different keywords
    6. Once satisfied → writes a comprehensive report
    7. You get a structured research report!

    **Tech Stack:**
    - 🧠 LangGraph — Agentic workflow framework
    - ⚡ Groq + Llama 3.3 — Free ultra-fast LLM
    - 🔍 Tavily — AI-optimized web search
    - 🎨 Streamlit — Web UI
    """)

st.divider()

# ================================================================
# UI: INPUT
# ================================================================

topic = st.text_input(
    "Enter your research topic:",
    placeholder="Example: Impact of AI on jobs, Climate change solutions, Future of electric vehicles...",
    help="Be specific for better results!"
)

col1, col2 = st.columns([1, 4])
# Create two columns — button on left, space on right
# [1, 4] means first column is 1 part wide, second is 4 parts wide

with col1:
    research_button = st.button(
        "🔍 Research",
        type="primary",
        use_container_width=True
    )

# ================================================================
# UI: RUN AGENT
# ================================================================

if research_button and topic.strip():
    # User clicked Research AND typed something

    st.session_state.report = None
    # Clear any previous report

    st.session_state.topic = topic

    # Show live progress to user
    with st.status("Agent is working...", expanded=True) as status:
        # st.status creates a nice expandable progress box

        st.write("🧠 Planning search queries...")
        st.write("🔍 Searching the internet...")
        st.write("📊 Evaluating results...")
        st.write("✍️ Writing your report...")

        try:
            report = run_research_agent(topic)
            # Run the entire LangGraph agent!
            # This might take 30-60 seconds

            st.session_state.report = report
            status.update(label="Research complete!", state="complete")
            # Update the status box to show completion

        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"Error: {e}")

elif research_button and not topic.strip():
    st.warning("Please enter a research topic first!")

# ================================================================
# UI: DISPLAY REPORT
# ================================================================

if st.session_state.report:
    st.divider()

    st.subheader(f"📋 Research Report: {st.session_state.topic}")

    st.markdown(st.session_state.report)
    # st.markdown renders the report with proper formatting
    # Headers, bold text, bullet points all render nicely

    st.divider()

    # Download button so user can save the report
    st.download_button(
        label="📥 Download Report",
        data=st.session_state.report,
        file_name=f"research_{st.session_state.topic[:30].replace(' ', '_')}.txt",
        # Create filename from topic — replace spaces with underscores
        mime="text/plain"
    )


