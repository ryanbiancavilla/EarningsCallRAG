import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=api_key)


def clean_transcript_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n")
    pattern = r"(?m)^(Presentation|Question-and-Answer Session|Company Participants|Conference Call Participants|[A-Z][a-z]+(?:\s[A-Z][a-z]+)+)$"
    text = re.sub(pattern, r"\n\n\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_document(text: str, source_name: str, max_chunk_size: int = 1200) -> list[dict]:
    cleaned_text = clean_transcript_text(text)
    paragraphs = cleaned_text.split("\n\n")

    ignore_patterns = [
        "company participants", "conference call participants",
        "extel vote", "wi-fi password", "research division",
        "thank you, everybody", "welcome to day"
    ]

    chunks, current_chunk, current_length = [], [], 0

    for para in paragraphs:
        para = para.strip()
        if not para or para.startswith("===="):
            continue

        lower_para = para.lower()
        if any(pattern in lower_para for pattern in ignore_patterns) and len(para) < 150:
            continue

        if current_length + len(para) > max_chunk_size and current_chunk:
            chunks.append({"text": "\n\n".join(current_chunk), "source": source_name})
            current_chunk = []
            current_length = 0

        current_chunk.append(para)
        current_length += len(para)

    if current_chunk:
        chunks.append({"text": "\n\n".join(current_chunk), "source": source_name})

    return chunks


def init_vector_store():
    file_paths = {
        "BofA Conference": "data/NVIDIA Corporation (NVDA) Presents at Bank of America 2026 Global Technology Conference Transcript.txt",
        "Morgan Stanley Conference": "data/NVIDIA Corporation (NVDA) Presents at Morgan Stanley Technology, Media & Telecom Conference 2026 Transcript.txt",
        "GTC AI Conference": "data/NVIDIA Corporation (NVDA) Presents at NVIDIA GTC AI Conference 2026 Prepared Remarks Transcript.txt",
        "AI Summit": "data/NVIDIA Corporation (NVDA) Presents at Second Annual AI Summit Transcript.txt",
        "TD Cowen Conference": "data/NVIDIA Corporation (NVDA) Presents at TD Cowen's 54th Annual Technology, Media & Telecom Conference Transcript.txt",
    }

    all_chunks = []
    for source_name, path in file_paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            all_chunks.extend(chunk_document(text, source_name))

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="nvidia_transcripts",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0 and all_chunks:
        documents = [c["text"] for c in all_chunks]
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        metadatas = [{"source": c["source"]} for c in all_chunks]
        collection.add(documents=documents, ids=ids, metadatas=metadatas)

    return collection


collection = init_vector_store()


# Vector Search with Similarity Scores
def search_docs_with_scores(query: str, n_results: int = 15):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity_score = round(1 - dist, 4)
        hits.append({
            "text": doc,
            "source": meta["source"],
            "score": similarity_score
        })
    return hits


# Tool wrapper for Agent (formats source names alongside text)
def search_docs(query: str) -> str:
    results = search_docs_with_scores(query, n_results=15)
    formatted_chunks = [
        f"--- SOURCE: {item['source']} ---\n{item['text']}"
        for item in results
    ]
    return "\n\n".join(formatted_formatted_chunks if 'formatted_formatted_chunks' in locals() else formatted_chunks)


# Standard RAG
def ask_rag(question: str, n_results: int = 15) -> str:
    hits = search_docs_with_scores(question, n_results=n_results)
    context = "\n\n".join([f"Source ({h['source']}):\n{h['text']}" for h in hits])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based ONLY on the provided context. "
                "Ensure you include relevant details from all conferences present in the context. "
                "If context is insufficient, state that you don't have enough context."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {question}"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


# Agentic RAG
rag_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search NVIDIA earnings call and investor conference transcripts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"]
            }
        }
    }
]

available_tools = {"search_docs": search_docs}


def rag_agent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant with access to 5 NVIDIA investor conference transcripts: "
                "1. BofA Conference\n"
                "2. Morgan Stanley Conference\n"
                "3. GTC AI Conference\n"
                "4. AI Summit\n"
                "5. TD Cowen Conference\n\n"
                "When asked to summarize, list, or compare across conferences, you MUST cover all 5. "
                "If an initial search does not return data for all 5 conferences, run focused search queries "
                "for the specific missing conference names until you have details for all 5."
            ),
        },
        {"role": "user", "content": question},
    ]

    for step in range(8):  # Increased step limit for multiple searches
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=rag_tools,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.message.tool_calls:
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = available_tools[func_name](**args)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return "Could not answer within step limit."