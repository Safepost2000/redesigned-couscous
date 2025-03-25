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
# In production/deployment, prefer setting environment variables directly.
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
    # API keys MUST be set as environment variables (or loaded via .env)
    if provider == "Gemini":
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            st.error("❌ Google AI (Gemini) API Key not found in environment variables (GOOGLE_AI_API_KEY).")
            return None
        try:
            # Use convert_system_message_to_human=True for better compatibility with ReAct prompts
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
    # elif provider == "OpenAI": ...
    else:
        st.error(f"Unsupported LLM provider selected: {provider}")
        return None

def run_search_sanitized(query: str, search_wrapper: GoogleSearchAPIWrapper) -> str:
    """
    Strips leading/trailing whitespace AND removes double quotes (")
    before running the Google search to prevent API errors.
    Returns search results or an error message string.
    """
    # Step 1: Remove leading/trailing whitespace (like newlines)
    sanitized_query = query.strip()

    # Step 2: Remove ALL double quote characters (").
    # Fixes errors from unpaired/misplaced quotes causing 400 Bad Request.
    # NOTE: This disables intentional phrase searches using quotes.
    sanitized_query = sanitized_query.replace('"', '')

    if not sanitized_query:
        return "Error: Search query cannot be empty after sanitization."

    # Optional: Log the final sanitized query for debugging
    # print(f"[Agent Tool] Running final sanitized search: '{sanitized_query}'")

    try:
        # Execute the search using the underlying wrapper
        results = search_wrapper.run(sanitized_query)
        return results if results else "No results found for the query."
    except Exception as e:
        # Attempt to return a more informative error message to the agent
        error_detail = str(e)
        status_code = "N/A"
        # Extract status code if available in the exception (common for HttpError)
        if hasattr(e, 'resp') and hasattr(e.resp, 'status'):
             status_code = e.resp.status
        # Log the detailed error server-side for debugging if needed
        # print(f"ERROR during Google Search API call: Status {status_code} - {error_detail}")
        # Return a formatted error message that the agent can potentially understand
        return f"Error during Google Search API call: Status {status_code} - {error_detail}"


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

    # Define model options for each provider (Expanded)
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
        "Groq": (
            "llama3-8b-8192", # Fast general purpose
            "llama3-70b-8192", # Powerful general purpose
            "mixtral-8x7b-32768", # Large context window
            "gemma-7b-it",
            "gemma2-9b-it",
        ),
        "Mistral": (
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
            "codestral-latest", # Code focused
        ),
        # "OpenAI": ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"), # Uncomment if adding OpenAI
    }

    # Display the selectbox with the appropriate models
    model_name = st.selectbox(
        f"Choose {llm_provider} Model",
        options=model_options.get(llm_provider, ("default-model",)), # Get models or provide default
        key=f"{llm_provider.lower()}_model_select" # Unique key per provider ensures state retention
    )

    # Agent Iteration Limit Slider
    # Set a reasonable default, max value, and current value
    max_iterations = st.slider(
        "Max Agent Iterations",
        min_value=3,
        max_value=15,
        value=7, # Default increased slightly
        key="max_iter_slider",
        help="Maximum number of steps (LLM calls + Tool calls) the agent can take before stopping."
        )

    # Display required environment variables (Informational)
    st.divider()
    st.subheader("🔑 Required Environment Variables")
    st.markdown(f"""
    - `GOOGLE_API_KEY` (For Google Search)
    - `GOOGLE_CSE_ID` (For Google Search)
    - **`{llm_provider.upper()}_API_KEY`** (For selected LLM)
    *(Optionally use a `.env` file for local development)*
    """)
    # Check if search keys are present for immediate feedback
    if not (os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")):
         st.warning("⚠️ Google Search API Key or CSE ID missing!")
    # Add similar check for SerpAPI if using it as an alternative


# --- Main UI Area ---
query = st.text_area("Enter your query:", key="query_input", height=100,
                     placeholder="e.g., Find PDF financial reports on example.com\nSearch for 'admin login' titles on internal.example.com\nFind exposed .env files related to 'database credentials'")

run_button = st.button("Run Agent")

st.divider()
# Create containers to clearly separate agent thinking from the final result
st.subheader("🤔 Agent Thinking Process")
thinking_container = st.container(border=True) # Added border for visual separation
st.subheader("🏁 Final Result") # Changed header for clarity
result_container = st.container()


# --- Agent Execution Logic ---
if run_button and query:
    # 1. Validate Required Search API Keys (Essential upfront check)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    search_key_present = bool(google_api_key and google_cse_id)
    # Adapt this check if using SerpAPI as an alternative or fallback

    if not search_key_present:
        st.error("❌ Google Search API Key (`GOOGLE_API_KEY`) and/or CSE ID (`GOOGLE_CSE_ID`) missing from environment variables. Cannot proceed.")
        st.stop() # Stop execution if essential search keys are missing

    # 2. Initialize LLM (using selected provider and model)
    llm = get_llm(llm_provider, model_name)
    if not llm: # get_llm handles displaying error message in the UI if key is missing or init fails
        st.stop()

    # 3. Initialize Tools and Agent within a try-except block for robustness
    try:
        # Setup Search Tool Wrapper
        search_wrapper = GoogleSearchAPIWrapper(
             google_api_key=google_api_key, # Explicitly pass keys (optional, wrapper often picks them up)
             google_cse_id=google_cse_id
        )
        # search_wrapper = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY")) # Example if using SerpAPI

        # Define the Google Search tool using the sanitizing function
        google_search_tool = Tool(
            name="Google Search with Dorks",
            description=DEFAULT_SEARCH_TOOL_DESC,
            # Use a lambda to ensure the current search_wrapper instance is passed correctly
            func=lambda q: run_search_sanitized(q, search_wrapper),
            # Configure how tool errors are handled (optional, defaults are often sufficient)
            # return_direct=False, # Agent processes output (default)
            # handle_tool_error=True, # Agent is informed about tool errors (default)
        )
        tools = [google_search_tool] # List of tools agent can use

        # Pull the ReAct agent prompt template from Langchain Hub
        prompt = hub.pull("hwchase17/react")

        # Create the agent (the core runnable logic)
        agent = create_react_agent(llm, tools, prompt)

        # Setup callback handler for displaying agent steps in Streamlit UI
        st_callback = StreamlitCallbackHandler(
            thinking_container, # Target container for thoughts
            expand_new_thoughts=True, # Auto-expand new thoughts for visibility
            collapse_completed_thoughts=False # Keep previous steps visible
            )

        # Create the Agent Executor (manages the agent's execution loop)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True, # Must be True for callbacks to receive thought process data
            handle_parsing_errors=True, # Attempt to recover from LLM output formatting errors
            max_iterations=max_iterations, # Use the value from the slider
            callbacks=[st_callback], # Register the Streamlit callback handler
            # Consider adding early stopping methods if needed for more complex scenarios
            # early_stopping_method="generate", # LLM decides when to stop
        )

        # 4. Run the Agent and Display Results
        with thinking_container:
             # Clear previous thinking process before a new run
             thinking_container.empty()
             st.info(f"⏳ Running agent with **{llm_provider}** (`{model_name}`), Max Iterations: {max_iterations}...")

        # Clear previous result before a new run
        with result_container:
            result_container.empty()

        response = None # Initialize response variable
        agent_error_occurred = False # Flag to track if agent execution itself threw an error

        try:
            # Invoke the agent executor with the user's query
            # This starts the Thought -> Action -> Observation -> Thought loop
            response = agent_executor.invoke(
                {"input": query},
                # Pass callbacks config again (recommended practice)
                {"callbacks": [st_callback]}
            )
        except Exception as e:
            # Catch potential errors during the agent's execution loop
            agent_error_occurred = True
            st.error(f"❌ An error occurred during agent execution: {e}")

        # 5. Process and Display the Final Outcome in the Result Container
        with result_container:
            if agent_error_occurred:
                # If invoke failed, we don't need to process 'response'
                pass # Error already displayed by the except block
            elif response and "output" in response:
                agent_output = response['output']
                # Check specifically for the iteration limit message in the output
                if ITERATION_LIMIT_MESSAGE in agent_output:
                     st.warning(f"⚠️ {ITERATION_LIMIT_MESSAGE}")
                     st.info("The agent could not reach a final answer within the allowed steps. Consider increasing the 'Max Agent Iterations' slider in the sidebar, simplifying your query, or trying a different LLM model.")
                else:
                     # Success case: Agent provided a final answer
                     st.success(agent_output)
            else:
                # Agent finished, but response structure is unexpected or 'output' is missing
                # This *could* also happen if it hit the limit but didn't output the specific message.
                st.warning("Agent finished processing, but no definitive final answer was provided.")
                st.info("This might happen if the agent couldn't find the answer or hit the iteration limit without stating it explicitly. Try adjusting settings or the query.")

    except Exception as e:
        # Catch errors during the setup of tools/agent/executor components
        st.error(f"❌ Failed to initialize agent components: {e}")

elif run_button and not query:
    # Handle case where run is clicked with no query input
    st.warning("⚠️ Please enter a query.")
