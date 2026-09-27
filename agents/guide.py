"""Medical guideline Agent: retrieves from the guideline with RAG and gives preliminary advice."""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from agents.common import agent_update, ask_agent, create_model
from config import EMBEDDING_MODEL, GUIDELINE_PATH, INDEX_PATH
from state import State


def load_vector_store() -> FAISS:
    """Load the vector store if it exists; otherwise load document -> split by heading -> embed -> save."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if INDEX_PATH.exists():
        # The index was generated locally by us, so deserializing it is safe
        return FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )

    text = TextLoader(str(GUIDELINE_PATH), encoding="utf-8").load()[0].page_content

    # Split on "## <condition>" headings, one chunk per condition; strip_headers=False keeps the heading for retrieval
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "disease")],
        strip_headers=False,
    )
    # Drop the intro section at the top that has no condition heading
    chunks = [c for c in text_splitter.split_text(text) if "disease" in c.metadata]

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_PATH))
    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store


retriever = load_vector_store().as_retriever(search_kwargs={"k": 3})


@tool
def search_guideline(query: str) -> str:
    """Search the medical guideline for content related to a symptom or disease name."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant content found in the guideline"
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


GUIDE_FAIL_TEXT = "Sorry, medical advice is temporarily unavailable. If you feel unwell, please see a doctor promptly."

guide_agent = create_agent(
    model=create_model(),
    tools=[search_guideline],
    system_prompt="""
        You are the hospital's consultation assistant. Based on the symptoms the user describes, give preliminary reference advice.

        Follow these steps:
        1. Extract the main symptoms from the user's description and call search_guideline to search the guideline
        2. Answer only from the retrieved guideline content, covering: possible causes, initial care advice, recommended department, and when to seek medical care promptly
        3. If the guideline has no relevant content, say so honestly and advise the user to visit a hospital

        Do not make up diagnoses or medications that are not in the guideline.
        Always end your answer with: "This information is for reference only. Please follow your doctor's diagnosis."
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def guide_node(state: State):
    print("~~~~~~This is Guide Agent~~~~~~")
    answer = await ask_agent(guide_agent, state["messages"][0].content, GUIDE_FAIL_TEXT)
    return agent_update(answer)
