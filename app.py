import streamlit as st
import os
import glob
import tempfile
from dotenv import load_dotenv
from io import BytesIO
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import StreamlitCallbackHandler
# Corrected import for create_retriever_tool
try:
    from langchain.tools.retriever import create_retriever_tool  # Newer versions
except ImportError:
    from langchain_core.tools import create_retriever_tool  # Older versions fallback

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RBI_DOCS_PATH = "rbi_docs"
MAX_UPLOAD_TEXT_CHARS = 150000

# --- Helper Functions ---
@st.cache_resource(show_spinner="Loading and Indexing RBI Documents...")
def load_and_index_docs(docs_path):
    """Loads PDFs, splits them, creates embeddings, and builds a FAISS vector store."""
    all_docs = []
    if not os.path.exists(docs_path):
        try:
            os.makedirs(docs_path)
            st.warning(f"Directory '{docs_path}' created. Please add RBI documents.", icon="⚠️")
        except Exception as e:
            st.error(f"Error creating '{docs_path}': {e}", icon="🚨")
            return None

    pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))
    if not pdf_files:
        st.warning(f"No PDFs found in '{docs_path}'. Agent will lack local RBI knowledge.", icon="⚠️")
        return None

    st.write(f"Found {len(pdf_files)} PDF(s) to load...")
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if docs:
                all_docs.extend(docs)
                st.write(f"- Loaded: {os.path.basename(pdf_path)}")
            else:
                st.warning(f"No content extracted from {os.path.basename(pdf_path)}")
        except Exception as e:
            st.warning(f"Skipping {os.path.basename(pdf_path)} due to error: {e}")

    if not all_docs:
        st.error("No valid documents loaded. Check PDF integrity.")
        return None

    st.write("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)

    st.write("Creating embeddings...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=GOOGLE_API_KEY
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        st.write("Vector store created successfully.")
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error creating embeddings or vector store: {e}")
        return None

@st.cache_resource(show_spinner="Initializing Compliance Agent...")
def initialize_agent(_retriever):
    """Initializes the LangChain Agent with Gemini and tools."""
    try:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY
        )

        tools = []
        if _retriever:
            retriever_tool = create_retriever_tool(
                _retriever,
                "rbi_guidelines_search",
                "MUST USE FIRST. Searches RBI regulations, circulars, and master directions.",
            )
            tools.append(retriever_tool)
        else:
            st.warning("Retriever tool unavailable.", icon="⚠️")

        search_tool = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="web_search",
                func=search_tool.run,
                description="Use for recent RBI announcements or general knowledge.",
            )
        )

        prompt_template = """
        You are an expert RBI Compliance Officer AI Assistant for a Regulated Entity (RE).
        Your goal is to help the RE adhere to RBI regulations, guidelines, circulars, and master directions.
        Prioritize 'rbi_guidelines_search' for regulatory questions, then 'web_search' for recent updates.

        Tools: {tools}

        Reasoning format:
        Question: {input}
        Thought: Break down the question and plan steps.
        Action: [{tool_names}]
        Action Input: Input for the action
        Observation: Result of the action
        ... (repeat as needed)
        Thought: I have enough info to answer.
        Final Answer: Comprehensive, actionable response.

        - Reference RBI guidelines from 'rbi_guidelines_search' when possible.
        - Specify if info is from 'web_search'.
        - Do not invent regulations if info is unavailable.
        - Compare provided policy (in 'Context') against RBI guidelines if requested.

        Previous conversation: {chat_history}
        Context: {user_context}
        New input: {input}
        {agent_scratchpad}
        """
        prompt = PromptTemplate.from_template(prompt_template)

        agent = create_react_agent(llm, tools, prompt)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors="I encountered an issue. Please rephrase.",
            max_iterations=7,
        )
        return agent_executor
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

def process_uploaded_pdf(uploaded_file):
    """Loads and extracts text from an uploaded PDF."""
    if uploaded_file is None:
        return None
    tmp_pdf_path = None
    try:
        bytes_data = uploaded_file.getvalue()
        file_like_object = BytesIO(bytes_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file_like_object.getbuffer())
            tmp_pdf_path = tmpfile.name

        loader = PyPDFLoader(tmp_pdf_path)
        docs = loader.load()
        if not docs:
            st.warning("No content extracted from uploaded PDF.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        full_text = " ".join([doc.page_content for doc in splits])
        if len(full_text) > MAX_UPLOAD_TEXT_CHARS:
            full_text = full_text[:MAX_UPLOAD_TEXT_CHARS]
            st.warning(f"Text truncated to {MAX_UPLOAD_TEXT_CHARS} characters.")
        return full_text
    except Exception as e:
        st.error(f"Error processing uploaded PDF: {e}")
        return None
    finally:
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)

# --- Streamlit App UI ---
st.set_page_config(page_title="Enhanced RBI Compliance Assistant", layout="wide")
st.title("🤖 Enhanced RBI Compliance & Monitoring Assistant")
st.caption("Powered by LangChain, Google Gemini, DuckDuckGo Search & Streamlit")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input(
        "Google API Key (Gemini)",
        value=GOOGLE_API_KEY if GOOGLE_API_KEY else "",
        type="password",
        help="Get yours from Google AI Studio",
    )

    st.header("Knowledge Base (RBI Docs)")
    st.write(f"Looking for PDFs in: `./{RBI_DOCS_PATH}/`")
    st.info("Upload RBI PDFs here. Reload required after changes.")
    if st.button("🔄 Reload Knowledge Base & Agent"):
        st.cache_resource.clear()
        st.session_state.messages = [
            AIMessage(content="Hello! I am your RBI Compliance Assistant.")
        ]
        st.session_state.uploaded_doc_text = None
        st.session_state.current_upload_filename = None
        st.success("Knowledge base and agent re-initialized. Chat cleared.")
        st.rerun()

    st.header("Analyze Your Document")
    uploaded_file = st.file_uploader("Upload your policy document (PDF)", type="pdf")
    if "uploaded_doc_text" not in st.session_state:
        st.session_state.uploaded_doc_text = None
    if "current_upload_filename" not in st.session_state:
        st.session_state.current_upload_filename = None

    if uploaded_file:
        if st.session_state.current_upload_filename != uploaded_file.name:
            with st.spinner("Processing uploaded document..."):
                doc_text = process_uploaded_pdf(uploaded_file)
                if doc_text:
                    st.session_state.uploaded_doc_text = doc_text
                    st.session_state.current_upload_filename = uploaded_file.name
                    st.success(f"✅ Ready to analyze '{uploaded_file.name}'.")
                else:
                    st.error("Could not extract text from the uploaded document.")
                    st.session_state.uploaded_doc_text = None
                    st.session_state.current_upload_filename = None
        elif st.session_state.uploaded_doc_text:
            st.success(f"✅ Document '{st.session_state.current_upload_filename}' loaded.")

    if st.session_state.get("uploaded_doc_text"):
        if st.button("Clear Uploaded Document Context"):
            st.session_state.uploaded_doc_text = None
            st.session_state.current_upload_filename = None
            st.info("Uploaded document context cleared.")
            st.rerun()

# --- Main Chat Interface ---
if not api_key_input:
    st.warning("Please enter your Google API Key in the sidebar.")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = api_key_input
    GOOGLE_API_KEY = api_key_input

retriever = load_and_index_docs(RBI_DOCS_PATH)
agent_executor = initialize_agent(retriever)

if not agent_executor:
    st.error("Agent initialization failed. Check configuration and API key.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I am your RBI Compliance Assistant.")
    ]

for message in st.session_state.messages:
    avatar = "🧑‍💻" if isinstance(message, HumanMessage) else "🤖"
    with st.chat_message(message.type, avatar=avatar):
        st.markdown(message.content)

if prompt := st.chat_input("Ask about RBI rules, analyze uploaded doc..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    user_context = st.session_state.get("uploaded_doc_text", "")
    context_info_for_prompt = (
        f"\n\n--- User Uploaded Document Context ---\n{user_context}\n--- End ---\n"
        if user_context
        else "No document context provided by user."
    )
    context_display_message = "ℹ️ Querying with context from uploaded document." if user_context else ""

    agent_input = {"input": prompt, "user_context": context_info_for_prompt}

    with st.chat_message("assistant", avatar="🤖"):
        if context_display_message:
            st.info(context_display_message)

        st_callback_container = st.container()
        st_callback = StreamlitCallbackHandler(
            st_callback_container,
            max_thought_containers=3,
            expand_new_thoughts=True,
            collapse_completed_thoughts=True,
        )
        message_placeholder = st.empty()
        full_response = ""

        try:
            response = agent_executor.invoke(agent_input, {"callbacks": [st_callback]})
            full_response = response.get("output", "Sorry, I couldn't generate a response.")
        except Exception as e:
            st.error(f"Error running agent: {e}")
            full_response = f"Sorry, I encountered an error: {e}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append(AIMessage(content=full_response))
