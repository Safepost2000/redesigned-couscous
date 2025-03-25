import streamlit as st
import os
from dotenv import load_dotenv # To load API keys from .env file for local development

# --- Apply Nest Asyncio Patch ---
# This MUST be applied very early, before libraries using asyncio are imported or used.
import nest_asyncio
nest_asyncio.apply()
# --- End Patch ---

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
load_dotenv()

# --- Constants ---
DEFAULT_SEARCH_TOOL_DESC = (
    "Use this tool to search Google. Input should be a standard search query "
    "OR a specialized Google Dork query (e.g., 'site:example.com filetype:pdf', "
    "'intitle:\"admin login\"', 'inurl:config.json'). Useful for finding specific "
    "file types, pages on specific sites, specific titles, or URLs containing specific strings."
)
ITERATION_LIMIT_MESSAGE = "Agent stopped due to iteration limit or time limit." # Used for checking output

# --- Helper Functions ---

def get_llm(provider, model_name, temperature=0.1):
    """Initializes the selected LLM provider based on environment variables."""
    if provider == "Gemini":
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            st.error("❌ Google AI (Gemini) API Key not found in environment variables (GOOGLE_AI_API_KEY).")
            return None
        try:
            # nest_asyncio applied above should prevent the event loop error here
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
    else:
        st.error(f"Unsupported LLM provider selected: {provider}")
        return None

def run_search_sanitized(query: str, search_wrapper: GoogleSearchAPIWrapper) -> str:
    """
    Strips leading/trailing whitespace AND removes double quotes (")
    before running the Google search to prevent API errors.
    Returns search results or an error message string.
    """
    sanitized_query = query.strip()
    sanitized_query = sanitized_query.replace('"', '')

    if not sanitized_query:
        return "Error: Search query cannot be empty after sanitization."

    try:
        results = search_wrapper.run(sanitized_query)
        return results if results else "No results found for the query."
    except Exception as e:
        error_detail = str(e)
        status_code = "N/A"
        if hasattr(e, 'resp') and hasattr(e.resp, 'status'):
             status_code = e.resp.status
        return f"Error during Google Search API call: Status {status_code} - {error_detail}"


# --- Streamlit App UI ---

st.set_page_config(page_title="Langchain Agent with Google Dorks", layout="wide")
st.title("🕵️‍♂️ Langchain Agent using Google Dorks")
st.write("Enter a query, and the agent will decide if a Google Dork is needed to find the answer.")
st.caption("Ensure required API keys are set as environment variables.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Agent Settings")

    llm_provider = st.selectbox(
        "Choose LLM Provider",
        ("Gemini", "Groq", "Mistral"), # Add "OpenAI" etc.
        index=0,
        key="llm_provider_select",
        help="Ensure the corresponding API key environment variable is set.",
    )

    model_options = {
        "Gemini": (
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.0-pro",
            "gemini-2.5-pro-exp-03-25",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
                  ),
        "Groq": ("llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"),
        "Mistral": ("mistral-small-latest", "mistral-medium-latest", "mistral-large-latest", "codestral-latest"),
        # "OpenAI": ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"),
    }

    model_name = st.selectbox(
        f"Choose {llm_provider} Model",
        options=model_options.get(llm_provider, ("default-model",)),
        key=f"{llm_provider.lower()}_model_select"
    )

    max_iterations = st.slider(
        "Max Agent Iterations", min_value=3, max_value=15, value=7, key="max_iter_slider",
        help="Maximum steps (LLM calls + Tool calls) the agent can take."
        )

    st.divider()
    st.subheader("🔑 Required Environment Variables")
    st.markdown(f"""
    - `GOOGLE_API_KEY` (For Google Search)
    - `GOOGLE_CSE_ID` (For Google Search)
    - **`{llm_provider.upper()}_API_KEY`** (For selected LLM)
    *(Optionally use a `.env` file for local development)*
    """)
    if not (os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")):
         st.warning("⚠️ Google Search API Key or CSE ID missing!")

# --- Main UI Area ---
query = st.text_area("Enter your query:", key="query_input", height=100,
                     placeholder="e.g., Find PDF financial reports on example.com...")

run_button = st.button("Run Agent")

st.divider()
st.subheader("🤔 Agent Thinking Process")
thinking_container = st.container(border=True)
st.subheader("🏁 Final Result")
result_container = st.container()

# --- Agent Execution Logic ---
if run_button and query:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    search_key_present = bool(google_api_key and google_cse_id)

    if not search_key_present:
        st.error("❌ Google Search API Key/CSE ID missing from environment variables.")
        st.stop()

    llm = get_llm(llm_provider, model_name)
    if not llm:
        st.stop()

    try:
        search_wrapper = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=google_cse_id)

        google_search_tool = Tool(
            name="Google Search with Dorks",
            description=DEFAULT_SEARCH_TOOL_DESC,
            func=lambda q: run_search_sanitized(q, search_wrapper),
        )
        tools = [google_search_tool]

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        st_callback = StreamlitCallbackHandler(
            thinking_container, expand_new_thoughts=True, collapse_completed_thoughts=False
            )
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,
            max_iterations=max_iterations, callbacks=[st_callback],
        )

        with thinking_container:
             thinking_container.empty()
             st.info(f"⏳ Running agent with **{llm_provider}** (`{model_name}`), Max Iterations: {max_iterations}...")
        with result_container:
            result_container.empty()

        response = None
        agent_error_occurred = False
        try:
            response = agent_executor.invoke(
                {"input": query}, {"callbacks": [st_callback]}
            )
        except Exception as e:
            agent_error_occurred = True
            st.error(f"❌ An error occurred during agent execution: {e}")

        with result_container:
            if agent_error_occurred:
                pass # Error already displayed
            elif response and "output" in response:
                agent_output = response['output']
                if ITERATION_LIMIT_MESSAGE in agent_output:
                     st.warning(f"⚠️ {ITERATION_LIMIT_MESSAGE}")
                     st.info("Increase 'Max Iterations', simplify query, or try a different LLM model.")
                else:
                     st.success(agent_output)
            else:
                st.warning("Agent finished processing, but no definitive final answer was provided.")
                st.info("This might happen if the answer wasn't found or the iteration limit was hit implicitly.")

    except Exception as e:
        st.error(f"❌ Failed to initialize agent components: {e}")

elif run_button and not query:
    st.warning("⚠️ Please enter a query.")
