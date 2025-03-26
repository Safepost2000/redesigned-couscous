import streamlit as st
import os
import glob
import tempfile
from dotenv import load_dotenv
from io import BytesIO

# Langchain components - UPDATED IMPORTS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS # Corrected import
from langchain_community.document_loaders import PyPDFLoader # Corrected import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool # Import Tool directly for DuckDuckGo
from langchain_community.tools import DuckDuckGoSearchRun # Corrected import
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough

# Streamlit Langchain Callbacks
from langchain_community.callbacks import StreamlitCallbackHandler # Corrected import (already community)

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RBI_DOCS_PATH = "rbi_docs" # Folder containing RBI PDF documents
MAX_UPLOAD_TEXT_CHARS = 15000 # Limit context size from uploaded doc

# --- Helper Functions ---

@st.cache_resource(show_spinner="Loading and Indexing RBI Documents...")
def load_and_index_docs(docs_path):
    """Loads PDFs from path, splits them, creates embeddings, and builds a FAISS vector store."""
    all_docs = []
    if not os.path.exists(docs_path):
        st.warning(f"Directory '{docs_path}' not found. No local RBI documents loaded.", icon="⚠️")
        return None
    pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))
    if not pdf_files:
        st.warning(f"No PDF documents found in '{docs_path}'. The agent will lack specific RBI knowledge from local files.", icon="⚠️")
        return None

    st.write(f"Found {len(pdf_files)} PDF(s) to load...")
    loaded_count = 0
    for pdf_path in pdf_files:
        try:
            # Use the corrected PyPDFLoader import
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
            loaded_count += 1
            st.write(f"- Loaded: {os.path.basename(pdf_path)}")
        except Exception as e:
            st.error(f"Error loading {os.path.basename(pdf_path)}: {e}")

    if not all_docs:
         st.error("Failed to load any documents, though PDF files might exist. Check PDF integrity or loading errors above.")
         return None

    st.write("Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)

    st.write(f"Creating embeddings using Google Generative AI (this may take a moment)...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        # Use the corrected FAISS import
        vectorstore = FAISS.from_documents(splits, embeddings)
        st.write("Vector store created successfully.")
        return vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks
    except Exception as e:
        st.error(f"Error creating embeddings or vector store: {e}")
        return None

@st.cache_resource(show_spinner="Initializing Compliance Agent...")
def initialize_agent(_retriever):
    """Initializes the Langchain Agent with Gemini and tools."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                   temperature=0.3,
                                   google_api_key=GOOGLE_API_KEY)

        tools = []
        if _retriever:
            retriever_tool = create_retriever_tool(
                _retriever,
                "rbi_guidelines_search",
                "MUST USE FIRST. Searches and returns relevant excerpts from the internal knowledge base of RBI regulations, circulars, and master directions. Use this to answer specific questions about known RBI compliance requirements.",
            )
            tools.append(retriever_tool)
        else:
             st.warning("Retriever tool (RBI document search) is unavailable.", icon="⚠️")

        # Web Search Tool - using corrected DuckDuckGoSearchRun import
        search_tool = DuckDuckGoSearchRun()
        tools.append(
            Tool( # Use the generic Tool class to wrap the run method
                name="web_search",
                func=search_tool.run,
                description="Use this tool to find information about very recent RBI announcements, news, current events, or topics potentially not covered in the internal knowledge base.",
            )
        )

        # Enhanced Prompt Template (No changes needed here for the warnings)
        prompt_template = """
        You are an expert RBI Compliance Officer AI Assistant for a Regulated Entity (RE).
        Your primary goal is to help the RE adhere to all applicable Reserve Bank of India (RBI) regulations, guidelines, circulars, and master directions.
        You must prioritize information from the 'rbi_guidelines_search' tool when available, as it represents the curated knowledge base.
        Use the 'web_search' tool ONLY if the information needed is likely very recent news, announcements, or general knowledge not expected in the RBI regulatory documents.

        You have access to the following tools:
        {tools}

        Use the following format for your reasoning process:

        Question: the input question you must answer
        Thought: Break down the question. Plan your steps. ALWAYS prioritize checking the 'rbi_guidelines_search' tool first for regulatory questions. Explain your reasoning for choosing a tool.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now have enough information from my tools and analysis to provide a final answer.
        Final Answer: Provide a comprehensive, accurate, and actionable answer.
        - Reference specific RBI regulations or guidelines using information from the 'rbi_guidelines_search' tool whenever possible.
        - If information comes from 'web_search', state that clearly.
        - If the information isn't found in either tool, state that clearly. Do not invent regulations.
        - If asked to analyze a provided policy (present in the 'Context' below), compare it explicitly against the retrieved RBI guidelines.
        - If asked for a checklist, summary, or specific format, structure your Final Answer accordingly (e.g., use markdown lists, bullet points).

        Begin!

        Previous conversation history:
        {chat_history}

        Context (if any provided by user upload):
        {user_context}

        New input: {input}
        {agent_scratchpad}
        """
        prompt = PromptTemplate.from_template(prompt_template)

        # Create the ReAct Agent
        agent = create_react_agent(llm, tools, prompt)

        # Create the Agent Executor with Memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='input', output_key='output')

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors="I encountered an issue processing the response. Please try rephrasing.",
            max_iterations=7
        )
        return agent_executor

    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

def process_uploaded_pdf(uploaded_file):
    """Loads, splits, and extracts text from an uploaded PDF file."""
    if uploaded_file is None:
        return None
    try:
        bytes_data = uploaded_file.getvalue()
        file_like_object = BytesIO(bytes_data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file_like_object.getbuffer())
            tmp_pdf_path = tmpfile.name

        # Use the corrected PyPDFLoader import
        loader = PyPDFLoader(tmp_pdf_path)
        docs = loader.load()
        os.remove(tmp_pdf_path)

        if not docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        full_text = " ".join([doc.page_content for doc in splits])

        return full_text[:MAX_UPLOAD_TEXT_CHARS]

    except Exception as e:
        st.error(f"Error processing uploaded PDF: {e}")
        return None

# --- Streamlit App UI ---

st.set_page_config(page_title="Enhanced RBI Compliance Assistant", layout="wide")
st.title("🤖 Enhanced RBI Compliance & Monitoring Assistant")
st.caption("Powered by Langchain, Google Gemini, DuckDuckGo Search & Streamlit")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Google API Key (Gemini)", value=GOOGLE_API_KEY if GOOGLE_API_KEY else "", type="password", help="Get yours from Google AI Studio")

    st.header("Knowledge Base (RBI Docs)")
    st.write(f"Looking for PDFs in: `./{RBI_DOCS_PATH}/`")
    st.info("Upload relevant RBI PDFs (Master Directions, Circulars, etc.) here. Reload required after adding/changing files.")
    if st.button("🔄 Reload Knowledge Base & Agent"):
        st.cache_resource.clear()
        if "messages" in st.session_state:
            st.session_state.messages = []
        if "uploaded_doc_text" in st.session_state:
            st.session_state.uploaded_doc_text = None
        if "current_upload_filename" in st.session_state:
            st.session_state.current_upload_filename = None
        st.success("Knowledge base and agent will be re-initialized. Chat cleared.")
        st.rerun()

    st.header("Analyze Your Document")
    uploaded_file = st.file_uploader("Upload your policy document (PDF)", type="pdf")

    # Initialize session state keys if they don't exist
    if "uploaded_doc_text" not in st.session_state:
        st.session_state.uploaded_doc_text = None
    if "current_upload_filename" not in st.session_state:
        st.session_state.current_upload_filename = None


    if uploaded_file is not None:
        # Check if it's a new file or the first upload
        if st.session_state.current_upload_filename != uploaded_file.name:
            with st.spinner("Processing uploaded document..."):
                doc_text = process_uploaded_pdf(uploaded_file)
                if doc_text:
                    st.session_state.uploaded_doc_text = doc_text
                    st.session_state.current_upload_filename = uploaded_file.name
                    st.success(f"✅ Ready to analyze '{uploaded_file.name}'. Ask a question about it in the chat.")
                    st.info(f"Document content (limited to {MAX_UPLOAD_TEXT_CHARS} chars) will be added as context for your next query.")
                else:
                    st.error("Could not extract text from the uploaded document.")
                    # Reset state if processing failed
                    st.session_state.uploaded_doc_text = None
                    st.session_state.current_upload_filename = None
        # If it's the same file as already loaded, just confirm it's ready
        elif st.session_state.uploaded_doc_text:
             st.success(f"✅ Document '{st.session_state.current_upload_filename}' is loaded. Ask a question about it.")


    if st.session_state.get("uploaded_doc_text"):
        if st.button("Clear Uploaded Document Context"):
            st.session_state.uploaded_doc_text = None
            st.session_state.current_upload_filename = None
            st.info("Uploaded document context cleared.")
            st.rerun()


# --- Main Chat Interface ---

if not api_key_input:
    st.warning("Please enter your Google API Key in the sidebar to begin.")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = api_key_input
    GOOGLE_API_KEY = api_key_input

retriever = load_and_index_docs(RBI_DOCS_PATH)
agent_executor = initialize_agent(retriever)


if not agent_executor:
    st.error("Agent initialization failed. Please check configuration, API key, and logs.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I am your RBI Compliance Assistant. Upload internal documents via the sidebar or ask me about RBI regulations.")]


for message in st.session_state.messages:
    avatar = "🧑‍💻" if isinstance(message, HumanMessage) else "🤖"
    with st.chat_message(message.type, avatar=avatar):
        st.markdown(message.content)


if prompt := st.chat_input("Ask about RBI rules, analyze uploaded doc, check compliance..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    user_context = st.session_state.get("uploaded_doc_text", "")
    context_info_for_prompt = "" # Initialize as empty
    context_display_message = "" # Initialize as empty

    if user_context:
         context_info_for_prompt = "\n\n--- Start of User Uploaded Document Context ---\n" + user_context + "\n--- End of User Uploaded Document Context ---\n"
         context_display_message = "ℹ️ Querying with context from the uploaded document."
    else:
         context_info_for_prompt = "No document context provided by user." # Be explicit in prompt context


    agent_input = {
        "input": prompt,
        "user_context": context_info_for_prompt
    }

    with st.chat_message("assistant", avatar="🤖"):
        if context_display_message: # Display context usage message only if context is present
             st.info(context_display_message)

        st_callback_container = st.container() # Define container where callbacks will render
        st_callback = StreamlitCallbackHandler(
            st_callback_container,
            max_thought_containers=3,
            expand_new_thoughts=True,
            collapse_completed_thoughts=True
        )
        message_placeholder = st.empty()
        full_response = ""

        try:
            response = agent_executor.invoke(agent_input, {"callbacks": [st_callback]})
            full_response = response.get('output', "Sorry, I couldn't generate a response.")

        except Exception as e:
            st.error(f"An error occurred while running the agent: {e}")
            full_response = f"Sorry, I encountered an error: {e}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append(AIMessage(content=full_response))
    # Optionally clear context after use and rerun - decide based on desired UX
    # if user_context:
    #     st.session_state.uploaded_doc_text = None
    #     st.session_state.current_upload_filename = None
    #     st.rerun()
