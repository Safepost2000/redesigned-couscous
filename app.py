import streamlit as st
import os
from dotenv import load_dotenv # To load API keys from .env file for local development

# Langchain Core/Community Imports
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper # Default Search Tool
# from langchain_community.utilities import SerpAPIWrapper # Alternative Search Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler # To display agent thoughts in UI

# LLM Integrations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
# from langchain_openai import ChatOpenAI # Uncomment if adding OpenAI option

# --- Load Environment Variables ---
# Load keys from .env file if it exists (useful for local development)
# In production, prefer setting environment variables directly.
load_dotenv()

# --- Constants ---
DEFAULT_SEARCH_TOOL_DESC = (
    "Use this tool to search Google. Input should be a standard search query "
    "OR a specialized Google Dork query (e.g., 'site:example.com filetype:pdf', "
    "'intitle:\"admin login\"', 'inurl:config.json'). Useful for finding specific "
    "file types, pages on specific sites, specific titles, or URLs containing specific strings."
)

# --- Helper Functions ---

def get_llm(provider, model_name, temperature=0.1):
    """Initializes the selected LLM provider based on environment variables."""
    # API keys MUST be set as environment variables (or loaded via .env)
    if provider == "Gemini":
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            st.error("❌ Google AI (Gemini) API Key not found in environment variables (GOOGLE_AI_API_KEY).")
            return None
        try:
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key, convert_system_message_to_human=True)
        except Exception as e:
            st.error(f"Error initializing Gemini ({model_name}): {e}")
            return None
    elif provider == "Groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("❌ Groq API Key not found in environment variables (GROQ_API_KEY).")
            return None
        try:
            return ChatGroq(model_name=model_name, temperature=temperature, groq_api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing Groq ({model_name}): {e}")
            return None
    elif provider == "Mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            st.error("❌ Mistral API Key not found in environment variables (MISTRAL_API_KEY).")
            return None
        try:
            return ChatMistralAI(model=model_name, temperature=temperature, api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing Mistral ({model_name}): {e}")
            return None
    # Add other LLM providers like OpenAI here if needed
    # elif provider == "OpenAI":
    #     api_key = os.getenv("OPENAI_API_KEY")
    #     if not api_key:
    #         st.error("❌ OpenAI API Key not found in environment variables (OPENAI_API_KEY).")
    #         return None
    #     try:
    #         return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)
    #     except Exception as e:
    #         st.error(f"Error initializing OpenAI: {e}")
    #         return None
    else:
        st.error(f"Unsupported LLM provider selected: {provider}")
        return None

def run_search_sanitized(query: str, search_wrapper: GoogleSearchAPIWrapper) -> str:
    """Strips whitespace/newlines from query before running the search."""
    sanitized_query = query.strip() # KEY CHANGE: Remove leading/trailing whitespace & newlines
    if not sanitized_query:
        return "Error: Search query cannot be empty after sanitization."
    # Optional: Log the sanitized query for debugging
    # print(f"[Agent Tool] Running sanitized search: '{sanitized_query}'")
    try:
        return search_wrapper.run(sanitized_query)
    except Exception as e:
        return f"Error during search: {e}" # Return error message to the agent

# --- Streamlit App UI ---

st.set_page_config(page_title="Langchain Agent with Google Dorks", layout="wide")
st.title("🕵️‍♂️ Langchain Agent using Google Dorks")
st.write("Enter a query, and the agent will decide if a Google Dork is needed to find the answer.")
st.caption("Ensure required API keys are set as environment variables.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Agent Settings")

    # LLM Provider and Model Selection
    llm_provider = st.selectbox(
        "Choose LLM Provider",
        ("Gemini", "Groq", "Mistral"), # Add "OpenAI" etc. if implemented in get_llm
        index=0,
        key="llm_provider_select",
        help="Ensure the corresponding API key environment variable is set.",
    )

    model_name = "Default" # Placeholder
    if llm_provider == "Gemini":
        model_name = st.selectbox("Choose Gemini Model", ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro"), key="gemini_model_select")
    elif llm_provider == "Groq":
        model_name = st.selectbox("Choose Groq Model", ("mixtral-8x7b-32768", "llama3-70b-8192", "gemma-7b-it"), key="groq_model_select")
    elif llm_provider == "Mistral":
        model_name = st.selectbox("Choose Mistral Model", ("mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"), key="mistral_model_select")
    # elif llm_provider == "OpenAI":
    #     model_name = st.selectbox("Choose OpenAI Model", ("gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"), key="openai_model_select")

    # Agent Iteration Limit
    max_iterations = st.slider("Max Agent Iterations", 1, 10, 5, key="max_iter_slider")

    # Display required environment variables (Informational)
    st.divider()
    st.subheader("Required Environment Variables")
    st.markdown(f"""
    - `GOOGLE_API_KEY` (For Google Search)
    - `GOOGLE_CSE_ID` (For Google Search)
    - `{llm_provider.upper()}_API_KEY` (Selected LLM)
    *(Optionally use a `.env` file for local development)*
    """)
    # Check if search keys are present for immediate feedback
    if not (os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")):
         st.warning("⚠️ Google Search API Key or CSE ID missing!")
    # Add similar check for SerpAPI if using it


# --- Main UI Area ---
query = st.text_area("Enter your query:", key="query_input", height=100,
                     placeholder="e.g., Find PDF financial reports on example.com\nSearch for 'admin login' titles on internal.example.com\nFind exposed .env files related to 'database credentials'")

run_button = st.button("Run Agent")

st.divider()
# Create containers to clearly separate agent thinking from the final result
thinking_container = st.container(border=True)
result_container = st.container()


# --- Agent Execution Logic ---
if run_button and query:
    # 1. Validate API Keys (Essential Check - Primarily for Search Tool)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    search_key_present = bool(google_api_key and google_cse_id)
    # serpapi_key = os.getenv("SERPAPI_API_KEY") # Uncomment if using SerpAPI
    # search_key_present = bool(serpapi_key) # Check for SerpAPI if using

    # LLM Key presence is checked within get_llm function

    if not search_key_present:
        st.error("❌ Google Search API Key (`GOOGLE_API_KEY`) and/or CSE ID (`GOOGLE_CSE_ID`) missing from environment variables.")
        # st.error("❌ Search API Key (GOOGLE_API_KEY/CSE_ID or SERPAPI_API_KEY) missing...") # Adapt if using SerpAPI
        st.stop() # Stop execution if search keys are missing

    # 2. Initialize LLM
    llm = get_llm(llm_provider, model_name)
    if not llm: # get_llm handles displaying error message in the UI
        st.stop()

    # 3. Initialize Tools and Agent
    try:
        # Setup Search Tool (Pass keys explicitly or rely on env vars being picked up by wrapper)
        # Note: GoogleSearchAPIWrapper picks up GOOGLE_API_KEY and GOOGLE_CSE_ID automatically if set.
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=google_api_key, # Optional: explicitly pass if needed
            google_cse_id=google_cse_id    # Optional: explicitly pass if needed
        )
        # search_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_key) # Uncomment if using SerpAPI

        # Define the tool using the sanitizing function
        google_search_tool = Tool(
            name="Google Search with Dorks",
            description=DEFAULT_SEARCH_TOOL_DESC,
            # Use a lambda to pass the wrapper instance to the sanitizer
            func=lambda q: run_search_sanitized(q, search_wrapper),
        )
        tools = [google_search_tool]

        # Pull the ReAct agent prompt
        prompt = hub.pull("hwchase17/react")

        # Create the agent
        agent = create_react_agent(llm, tools, prompt)

        # Setup callback handler for Streamlit UI
        st_callback = StreamlitCallbackHandler(thinking_container,
                                             expand_new_thoughts=True,
                                             collapse_completed_thoughts=False)

        # Create the Agent Executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True, # Required for callbacks to receive data
            handle_parsing_errors=True, # Attempt to recover from LLM format errors
            max_iterations=max_iterations,
            callbacks=[st_callback], # Register the Streamlit callback
        )

        # 4. Run the Agent and Display Results
        with thinking_container:
             st.markdown(f" M Running agent with **{llm_provider}** (`{model_name}`)...")

        response = None
        error_message = None
        try:
            # Invoke the agent executor with the input query
            response = agent_executor.invoke(
                {"input": query},
                # Pass callbacks config for potential streaming benefits
                {"callbacks": [st_callback]}
            )
        except Exception as e:
            error_message = f"An error occurred during agent execution: {e}"
            st.error(error_message) # Display runtime errors in the main UI

        # Display final answer or status in the result container
        with result_container:
            st.subheader("✅ Result:")
            if response and "output" in response:
                st.success(f"{response['output']}") # Display final answer clearly
            elif not error_message:
                st.warning("Agent finished processing, but no final answer was explicitly provided.")

    except Exception as e:
        # Catch errors during agent/tool setup more broadly
        st.error(f"❌ Failed to initialize or run agent components: {e}")

elif run_button and not query:
    st.warning("⚠️ Please enter a query.")
