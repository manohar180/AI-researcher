AI Research Agent
I built this project to learn LangGraph and understand how agentic AI works. Unlike my previous project (a PDF chatbot using LangChain), this one actually goes out and finds information on its own — I don't have to give it any documents. I just type a topic and it figures out the rest.
🔗 Live Demo: https://ai-researcher-b4bd5mcpweu8f2scfmvn6w.streamlit.app/

What it does
You give it a research topic and it autonomously:

Decides what to search for
Searches the internet multiple times
Checks if it found enough information
If not, searches again with different keywords
Writes a structured report once it's satisfied

The whole thing runs without any input from me after I type the topic. That's what makes it "agentic" — it makes its own decisions.

Why I built this
After building a RAG-based PDF chatbot, I wanted to understand what happens when you give an AI the ability to make decisions instead of just following fixed steps. LangGraph was the natural next step because it lets you build workflows that can loop, branch, and decide — unlike LangChain which just goes in a straight line.
I wanted to see the difference between:

An AI that answers from documents you give it (my PDF bot)
An AI that goes and finds the information itself (this project)


How it works
The agent is built as a graph with 4 nodes:
plan_searches — the LLM reads the topic and generates 3 specific search queries
execute_search — uses Tavily API to actually search the internet
evaluate_results — checks if the results gathered so far are enough to write a good report
write_report — takes everything found and writes a structured research report
The key part is the loop between execute_search and evaluate_results. If the agent decides it doesn't have enough information, it goes back and searches again. This loop is what makes it different from a simple chain.
plan_searches
      ↓
execute_search
      ↓
evaluate_results
      ↓
enough info? → YES → write_report → done
      ↓
      NO → back to execute_search

Tech stack

LangGraph — for building the agentic workflow with nodes and conditional edges
Groq + Llama 3.3 — free and very fast LLM API for the actual thinking and writing
Tavily — search API built specifically for AI agents, returns clean readable results
Streamlit — for the frontend, same as my previous project
Python — everything is in Python


What I learned building this
The biggest thing I learned is the difference between a chain and a graph. In my PDF chatbot everything went in one direction — load, chunk, embed, retrieve, answer. There was no decision making. Here the agent can go backwards, retry steps, and decide its own path based on what it finds.
I also learned about state management in LangGraph. Every node reads from and writes to a shared state dictionary. This is how information travels through the whole workflow — like a shared notebook every step can read and update.
Another thing was understanding why tools matter in agentic AI. Giving the LLM access to Tavily search is what makes it actually useful — without tools an agent can only use knowledge from its training data which has a cutoff date.

Running locally
Clone the repo:
bashgit clone https://github.com/manohar180/research-agent.git
cd research-agent
Install dependencies:
bashpip install -r requirements.txt
Create a .env file:
GROQ_API_KEY=your-groq-key-here
TAVILY_API_KEY=your-tavily-key-here
Get free API keys from:

Groq: console.groq.com
Tavily: tavily.com

Run the app:
bashstreamlit run app.py

Project structure
research-agent/
  ├── app.py            — Streamlit frontend
  ├── agent.py          — LangGraph agent logic
  ├── requirements.txt  — dependencies
  ├── .gitignore        — keeps .env out of GitHub
  └── README.md         — this file
