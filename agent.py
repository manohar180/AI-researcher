# ================================================================
# IMPORTS
# ================================================================

from langgraph.graph import StateGraph, END
# StateGraph = the main LangGraph class
# We use this to build our workflow as a graph
# END = a special constant that means "stop the workflow here"

from langchain_groq import ChatGroq
# Same as your PDF bot — connects to Groq's free API
# This is the LLM that will think, decide, and write the report

from langchain_tavily import TavilySearch
# Tavily is a search engine built specifically for AI agents
# It returns clean, readable results — better than Google scraping
# Your agent will use this to search the internet

from typing import TypedDict, List
# TypedDict lets us define exactly what our state dictionary looks like
# List is used to define list type hints

from dotenv import load_dotenv
import os

load_dotenv()
# Load API keys from .env file

# ================================================================
# STEP 1: DEFINE THE STATE
#
# State is the most important concept in LangGraph
# Think of it as a shared notebook that every node can read and write
# As the agent moves through steps, it keeps updating this notebook
# Every node receives the current state and returns updated state
#
# Example journey of state:
# Start:    {topic: "AI and jobs", searches: [], report: ""}
# After search: {topic: "AI and jobs", searches: ["result1", "result2"], report: ""}
# After write:  {topic: "AI and jobs", searches: [...], report: "AI is changing..."}
# ================================================================

class ResearchState(TypedDict):
    topic: str
    # The research topic the user typed
    # Example: "impact of AI on jobs"
    # Never changes throughout the workflow

    search_queries: List[str]
    # List of search queries the agent decides to use
    # Agent might decide to search multiple different queries
    # Example: ["AI impact on jobs", "automation replacing workers", "future of work AI"]

    search_results: List[str]
    # Raw results returned from Tavily searches
    # Each search adds more results to this list
    # Agent reads these to decide if it has enough info

    report: str
    # The final research report
    # Empty at start, filled in at the last step
    # This is what gets shown to the user

    search_count: int
    # Tracks how many times we've searched
    # Important for preventing infinite loops!
    # If agent searches more than 3 times, we force it to stop

# ================================================================
# STEP 2: SET UP TOOLS AND LLM
# ================================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
# Same Groq setup as your PDF bot
# This LLM will be the brain of our agent
# It will plan searches, evaluate results, and write the report

search_tool = TavilySearch(
    max_results=3,
    # Return top 3 results per search
    # More results = more tokens = slower and more expensive
    # 3 is a good balance for a free project

    api_key=os.getenv("TAVILY_API_KEY")
)
# This gives our agent the ability to search the internet
# Think of it like giving the agent a Google search button
# The agent can call this whenever it decides to search

# ================================================================
# STEP 3: DEFINE THE NODES
#
# Each node is a Python function that:
# - Receives the current state as input
# - Does one specific job
# - Returns updated state as output
#
# Our agent has 4 nodes:
# 1. plan_searches   → decide what to search for
# 2. execute_search  → actually search the internet
# 3. evaluate_results → decide if we have enough info
# 4. write_report    → write the final report
# ================================================================

def plan_searches(state: ResearchState) -> ResearchState:
    # This node is called FIRST
    # Job: decide what search queries to use for this topic
    # The LLM thinks about the topic and generates 3 good search queries

    print("Planning searches...")

    topic = state["topic"]
    # Get the research topic from state

    prompt = f"""
You are a research planner. Given a research topic, generate 3 specific search queries
that would help gather comprehensive information about the topic.

Topic: {topic}

Return ONLY a Python list of 3 search queries, nothing else.
Example format: ["query 1", "query 2", "query 3"]
"""
    # We ask the LLM to generate search queries
    # "Return ONLY a Python list" makes it easier to parse the response

    response = llm.invoke(prompt)
    # Send prompt to Groq/Llama3.3 and get response

    response_text = response.content
    # .content extracts just the text from the response object

    # Parse the list from the response
    # The LLM returns something like: ["query1", "query2", "query3"]
    try:
        import ast
        queries = ast.literal_eval(response_text.strip())
        # ast.literal_eval safely converts a string representation
        # of a Python list into an actual Python list
        # Example: '["a", "b"]' → ["a", "b"]
    except:
        # If parsing fails for any reason, fall back to simple queries
        queries = [topic, f"{topic} latest research", f"{topic} impact analysis"]

    print(f"Planned queries: {queries}")

    return {
        **state,
        # **state means "keep everything in current state"
        # Then we add/update specific fields below
        "search_queries": queries,
        "search_count": 0
        # Start search count at 0
    }


def execute_search(state: ResearchState) -> ResearchState:
    # This node searches the internet
    # Job: take the next unsearched query and search for it
    # Each time this node runs, it searches ONE query

    search_count = state["search_count"]
    queries = state["search_queries"]
    existing_results = state["search_results"]

    if search_count >= len(queries):
        # All queries have been searched already
        # Nothing new to search — return state unchanged
        return state

    current_query = queries[search_count]
    # Get the query for this search round
    # search_count=0 → first query
    # search_count=1 → second query etc

    print(f"Searching: {current_query}")

    try:
        results = search_tool.invoke(current_query)
        # Actually search the internet using Tavily
        # Returns a list of results with title, content, url

        # Convert results to readable text
        results_text = ""
        for r in results:
            if isinstance(r, dict):
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")
                results_text += f"Title: {title}\nContent: {content}\nSource: {url}\n\n"
            else:
                results_text += str(r) + "\n\n"
        # Format each result nicely so the LLM can read it clearly

    except Exception as e:
        results_text = f"Search failed: {e}"
        print(f"Search error: {e}")

    updated_results = existing_results + [f"Search '{current_query}':\n{results_text}"]
    # Add new results to existing results list
    # + combines two lists together

    return {
        **state,
        "search_results": updated_results,
        "search_count": search_count + 1
        # Increment search count so next time we use next query
    }


def evaluate_results(state: ResearchState) -> ResearchState:
    # This node reads all search results so far
    # Job: decide if we have enough information to write the report
    # This is what makes the agent intelligent — it judges its own progress

    print("Evaluating results...")

    all_results = state["search_results"]
    topic = state["topic"]

    combined = "\n".join(all_results)
    # Join all search results into one big text

    prompt = f"""
You are evaluating research quality. Based on the search results below,
do we have enough information to write a comprehensive report about: {topic}?

Search Results:
{combined[:3000]}
(showing first 3000 characters)

Answer with ONLY one word: YES or NO
"""
    # We limit to 3000 characters to avoid using too many tokens
    # Ask LLM to judge if we have enough info

    response = llm.invoke(prompt)
    evaluation = response.content.strip().upper()
    # .strip() removes extra spaces
    # .upper() converts to uppercase so YES/NO comparison works reliably

    print(f"Evaluation: {evaluation}")

    return {
        **state,
        "report": evaluation
        # Temporarily store evaluation in report field
        # We'll use this in the conditional edge to decide next step
    }


def write_report(state: ResearchState) -> ResearchState:
    # This is the FINAL node
    # Job: take all search results and write a proper research report

    print("Writing report...")

    all_results = state["search_results"]
    topic = state["topic"]

    combined = "\n".join(all_results)
    # Combine all search results

    prompt = f"""
You are an expert research writer. Using the search results below,
write a comprehensive, well-structured research report about: {topic}

Search Results:
{combined[:5000]}

Write the report with these sections:
1. Executive Summary (2-3 sentences overview)
2. Key Findings (main points discovered)
3. Detailed Analysis (in-depth discussion)
4. Implications (what this means)
5. Conclusion (final thoughts)

Make the report informative, clear, and professional.
"""

    response = llm.invoke(prompt)
    report = response.content
    # Get the complete report text

    print("Report written!")

    return {
        **state,
        "report": report
        # Store final report in state
        # This is what app.py will display to the user
    }

# ================================================================
# STEP 4: DEFINE THE DECISION FUNCTION
#
# This function is used by a conditional edge
# It looks at the current state and decides which node to go to next
# This is the heart of agentic behavior — the AI decides its own path
# ================================================================

def should_continue_searching(state: ResearchState) -> str:
    # This function is called after evaluate_results node
    # It reads the evaluation and search count
    # Returns the NAME of the next node to go to

    evaluation = state["report"]
    search_count = state["search_count"]
    max_searches = len(state["search_queries"])

    if search_count >= max_searches:
        # We've searched all planned queries
        # Must write report now regardless of evaluation
        print("All queries searched — writing report")
        return "write_report"

    if "YES" in evaluation:
        # LLM says we have enough info
        # Skip more searching, go straight to writing
        print("Enough info found — writing report")
        return "write_report"

    else:
        # LLM says we need more info
        # Go back and search again with next query
        print("Need more info — searching again")
        return "execute_search"

    # Notice how this creates a LOOP:
    # execute_search → evaluate_results → should_continue_searching
    #                        ↑                         |
    #                        |___ execute_search ←_____|  (if need more info)
    #                                                  |
    #                                         write_report (if enough info)

# ================================================================
# STEP 5: BUILD THE GRAPH
#
# Now we connect all nodes and edges together
# This creates the actual workflow structure
# ================================================================

def build_research_agent():

    # Create a new graph using our state definition
    graph = StateGraph(ResearchState)
    # StateGraph knows our state structure from ResearchState
    # Every node must accept and return ResearchState

    # ADD NODES
    # Tell the graph about each node function
    graph.add_node("plan_searches", plan_searches)
    graph.add_node("execute_search", execute_search)
    graph.add_node("evaluate_results", evaluate_results)
    graph.add_node("write_report", write_report)
    # First argument = name we give this node (used in edges)
    # Second argument = the actual function to run

    # SET ENTRY POINT
    graph.set_entry_point("plan_searches")
    # Tell LangGraph which node to start with
    # When we run the agent, it always starts here first

    # ADD REGULAR EDGES (always go from A to B)
    graph.add_edge("plan_searches", "execute_search")
    # After planning → always search
    # No decision needed here

    graph.add_edge("execute_search", "evaluate_results")
    # After searching → always evaluate
    # No decision needed here

    graph.add_edge("write_report", END)
    # After writing report → always end
    # END is imported from langgraph.graph

    # ADD CONDITIONAL EDGE (decision point!)
    graph.add_conditional_edges(
        "evaluate_results",
        # After this node runs...

        should_continue_searching,
        # ...call this function to decide where to go

        {
            "execute_search": "execute_search",
            # If function returns "execute_search" → go to execute_search node
            "write_report": "write_report"
            # If function returns "write_report" → go to write_report node
        }
    )
    # This is the magic line that creates the decision loop!

    # COMPILE THE GRAPH
    compiled = graph.compile()
    # Compiles all nodes and edges into a runnable workflow
    # Like pressing "build" on the assembled Lego structure

    return compiled


# ================================================================
# STEP 6: FUNCTION TO RUN THE AGENT
# Called by app.py when user submits a topic
# ================================================================

def run_research_agent(topic: str) -> str:
    # topic = what user typed in the search box
    # Returns the final research report as a string

    agent = build_research_agent()
    # Build the graph fresh each time

    initial_state = {
        "topic": topic,
        "search_queries": [],
        "search_results": [],
        "report": "",
        "search_count": 0
    }
    # This is the starting state — like an empty backpack
    # The agent will fill it up as it works through each node

    result = agent.invoke(initial_state)
    # Run the entire agent workflow!
    # LangGraph handles all the node execution and routing automatically
    # Returns the final state after all nodes have run

    return result["report"]
    # Extract just the report text from final state
    # This is what gets displayed to the user