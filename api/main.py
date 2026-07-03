import time
import os
import uuid
import json
import glob
import datetime
import urllib.parse
import asyncio
import concurrent.futures
import requests

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ollama import Client
from pypdf import PdfReader
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import PatternMatchingEventHandler
from flashrank import Ranker, RerankRequest

app = FastAPI(title="Advanced Agentic RAG API", version="3.3")

# --- Clients & Config ---
qdrant = QdrantClient(host="vector-db", port=6333)
ollama_client = Client(host="http://llm-server:11434")
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/app/flashrank_cache")

COLLECTION_NAME = "knowledge_base"
CHAT_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
DOCS_DIR = "/app/docs"

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )

# --- Feature 1: Hot Folder Ingestion ---
def process_file(filepath: str):
    # Ignore hidden system files like .DS_Store, .swp, etc.
    if os.path.basename(filepath).startswith('.'):
        print(f"Watchdog: Ignoring hidden file {os.path.basename(filepath)}", flush=True)
        return

    try:
        print(f"Watchdog: Attempting to ingest {os.path.basename(filepath)}...", flush=True)
        
        # 1. Read content based on file type
        if filepath.lower().endswith(".pdf"):
            reader = PdfReader(filepath)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            # Try reading as UTF-8 with character replacement for corrupt bytes
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                # Fallback to latin-1 if utf-8 completely fails
                with open(filepath, "r", encoding="latin-1") as f:
                    text = f.read()

        if not text.strip():
            print(f"Watchdog Warning: No text content found in {os.path.basename(filepath)}", flush=True)
            return

        # 2. Chunk the text
        words = text.split()
        chunks = [" ".join(words[i:i + 100]) for i in range(0, len(words), 80)]
        
        print(f"DEBUG: Ingesting {len(chunks)} chunks for {os.path.basename(filepath)}. Chunk size: {len(chunks[0])} words.", flush=True)

        # 3. Vectorize and create points
        points = []
        for chunk in chunks:
            embed_res = ollama_client.embed(model=EMBEDDING_MODEL, input=chunk)
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=embed_res["embeddings"][0], 
                    payload={"text": chunk, "source": os.path.basename(filepath)}
                )
            )
        
        # 4. Upsert to Qdrant
        if points:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"Watchdog SUCCESS: Ingested {len(points)} chunks from {os.path.basename(filepath)} into Qdrant!", flush=True)
    except Exception as e:
        print(f"Watchdog FATAL ERROR processing {os.path.basename(filepath)}: {str(e)}", flush=True)

class DocHandler(PatternMatchingEventHandler):
    def on_created(self, event):
        # Wait until file size stops changing (e.g. large file copy)
        size = -1
        while True:
            try:
                current_size = os.path.getsize(event.src_path)
                if current_size == size:
                    break
                size = current_size
            except Exception:
                pass
            time.sleep(1)
        process_file(event.src_path)

@app.on_event("startup")
def start_watchdog():
    if not os.path.exists(DOCS_DIR): os.makedirs(DOCS_DIR)
    observer = Observer()
    observer.schedule(DocHandler(), path=DOCS_DIR, recursive=False)
    observer.start()



def summarize_tool_output(tool_name: str, raw_output: str) -> str:
    """Compresses large tool outputs into high-signal summaries for the agent's memory."""
    if len(raw_output) < 1000:  # If it's already short, don't waste an LLM call
        return raw_output
        
    prompt = f"Summarize the following output from tool '{tool_name}' for an AI agent. Focus on key facts, numbers, and answers. Keep it concise:\n\n{raw_output[:5000]}"
    
    try:
        response = ollama_client.chat(
            model=CHAT_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        summary = response['message']['content']
        return f"[SUMMARY OF {tool_name}]: {summary}"
    except Exception as e:
        return f"[ERROR SUMMARIZING {tool_name}]: {str(e)}"
    

def summarize_document(filename_hint: str) -> str:
    print(f"DEBUG: Executing Summarization Tool for hint: '{filename_hint}'", flush=True)
    list_of_files = glob.glob(f"{DOCS_DIR}/*")

    if not list_of_files:
        return "Error: There are no files in the docs directory to summarize."

    target_file = None
    if filename_hint:
        for file_path in list_of_files:
            basename = os.path.basename(file_path).lower()
            name_without_ext = os.path.splitext(basename)[0]
            if filename_hint.lower() in name_without_ext:
                target_file = file_path
                break
                
    if not target_file:
        target_file = max(list_of_files, key=os.path.getctime)

    filename = os.path.basename(target_file)
    print(f"DEBUG: Reading full text of {filename} for agent...", flush=True)

    full_text = ""
    try:
        if target_file.lower().endswith(".pdf"):
            reader = PdfReader(target_file)
            full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            with open(target_file, "r", encoding="utf-8", errors="replace") as f:
                full_text = f.read()
        
        # We return the raw text back to the loop. The Agent will read it and summarize it itself!
        return f"Contents of {filename}:\n\n{full_text}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# --- Features 2 & 3: Reranking & Tool ---
def search_knowledge_base(query: str) -> str:
    print(f"DEBUG: Searching Qdrant for: {query}", flush=True)
    
    # --- FIX 1: NOMIC QUERY PREFIX ---
    # We add "search_query: " strictly for the embedding math
    nomic_query = f"search_query: {query}"
    embed_res = ollama_client.embed(model=EMBEDDING_MODEL, input=nomic_query)
    
    # Fetch the top 15 broad matches
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_res["embeddings"][0],
        limit=15
    ).points

    if not search_results: 
        print("DEBUG: No results found in Qdrant!", flush=True)
        return "No internal data found."

    # --- FIX 2: CALIBRATED THRESHOLD ---
    top_score = search_results[0].score
    print(f"DEBUG: Qdrant Top Match Raw Vector Score: {top_score:.4f}", flush=True)
    
    # Lowered to 0.50 to account for Nomic's tight baseline clustering
    if top_score < 0.50:
        print(f"DEBUG: Top match score ({top_score:.4f}) below threshold (0.50). Triggering Tier 1 MISS.", flush=True)
        return "No internal data found."

    # Deduplicate chunks
    unique_passages = {}
    for hit in search_results:
        text = hit.payload.get("text")
        if text and text not in unique_passages:
            unique_passages[text] = hit

    passages = [
        {"id": str(i), "text": text, "meta": hit.payload}
        for i, (text, hit) in enumerate(unique_passages.items())
    ]

    # --- FLASHRANK PRECISION ---
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)

    if not reranked:
        return "No internal data found."

    # --- NEW: THE FLASHRANK GATEKEEPER ---
    # Flashrank scores are much stricter. A good keyword match usually scores 0.80 to 0.99.
    # If the absolute best chunk Flashrank could find scores less than 0.60, it's a false positive!
    best_rerank_score = reranked[0].get("score", 0)
    print(f"DEBUG: Flashrank Top Rerank Score: {best_rerank_score:.4f}", flush=True)

    if best_rerank_score < 0.60:
        print(f"DEBUG: Reranker rejected Qdrant's results (Score: {best_rerank_score:.4f}). Triggering Web Fallback.", flush=True)
        return "No internal data found."
    # ------------------------------------

    # If it survives both Qdrant AND Flashrank, it is a guaranteed high-quality match
    top_chunks = [item["text"] for item in reranked[:3]]
    
    print("DEBUG: Tier 1 HIT! Injecting high-confidence context chunks.", flush=True) 
    
    return "\n\n".join(top_chunks)

rag_tool = {
    'type': 'function',
    'function': {
        'name': 'search_knowledge_base',
        'description': 'Search internal knowledge base.',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}
    }
}


# --- Feature 4: Web Search Tool ---
def web_search(query: str) -> str:
    print(f"DEBUG: Executing Indestructible Web Scraper for: '{query}'", flush=True)
    try:
        url = "https://lite.duckduckgo.com/lite/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        data = {"q": query}
        
        response = requests.post(url, headers=headers, data=data, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        formatted_results = []
        
        # Strategy 1: Look for any class that contains the word "snippet"
        snippets = soup.find_all(class_=lambda x: x and 'snippet' in x.lower())
        
        if snippets:
            for node in snippets[:4]:
                formatted_results.append(node.get_text(strip=True))
        else:
            # Strategy 2: Nuclear Fallback. Rip out all raw text, split it up, and keep the data
            print("DEBUG: Standard classes failed, engaging fallback parser...", flush=True)
            
            # Kill script and style elements first so we don't scrape code
            for script in soup(["script", "style"]):
                script.extract()
                
            all_text = soup.get_text(separator=' | ', strip=True)
            
            # Split the text chunks and filter out the tiny header/footer links
            chunks = [chunk.strip() for chunk in all_text.split(' | ')]
            valid_chunks = [c for c in chunks if len(c) > 40 and "DuckDuckGo" not in c and "Settings" not in c]
            
            formatted_results = valid_chunks[:4]
            
        if not formatted_results:
            print("DEBUG: Scraper found absolutely no text.", flush=True)
            return "No web search results found."
            
        context = "\n\n---\n\n".join(formatted_results)
        print(f"DEBUG: Scraped {len(formatted_results)} results successfully.", flush=True)
        return context
        
    except Exception as e:
        print(f"DEBUG: Scraper FATAL ERROR: {str(e)}", flush=True)
        return f"Web search failed: {str(e)}"


class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False

@app.post("/v1/ingest")
def bulk_ingest_docs():
    print("DEBUG: Manual bulk ingestion triggered!", flush=True)
    if not os.path.exists(DOCS_DIR):
        return {"status": "error", "message": "Docs directory not found."}
        
    files_processed = 0
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        # Only process actual files, skip subdirectories
        if os.path.isfile(filepath):
            process_file(filepath)
            files_processed += 1
            
    print(f"DEBUG: Bulk ingestion complete. Processed {files_processed} files.", flush=True)
    return {"status": "success", "message": f"Processed {files_processed} files."}

@app.get("/v1/models")
def get_models():
    return {"object": "list", "data": [{"id": "agentic-rag", "object": "model", "created": int(time.time()), "owned_by": "custom"}]}


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the local vector database for private documents, contracts, policies, and specific facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The specific semantic query to look up."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the live internet for real-time data, current pricing, news, and external facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The exact search engine query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Reads a full local file from the system. Use this when the user explicitly asks to summarize a file by name (e.g., 'bio', 'decree').",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename_hint": {"type": "string", "description": "The name of the file to read. Leave empty to read the newest file."}
                },
                "required": ["filename_hint"]
            }
        }
    }
]


async def agent_stream_generator(request_messages, qwen_override):
    # 1. Fetch the exact current time on the server
    now = datetime.datetime.now()
    current_time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    # 2. Build the message array
    messages = [{'role': m.role, 'content': m.content or ""} for m in request_messages]
    
    # 3. Inject the time cleanly right into the system prompt override
    time_context = f"\n\n[SYSTEM CONTEXT: Today is {current_time_str}. Use this absolute temporal anchor to calculate ages, dates, and relative time frames natively.]"
    messages.insert(0, {'role': 'system', 'content': qwen_override + time_context})

    def prune_context(msg_list: list) -> list:
        if len(msg_list) <= 8: return msg_list
        sliced = msg_list[-7:]
        if sliced[0].get('role') == 'tool':
            sliced = msg_list[-8:]
        return [msg_list[0]] + sliced

    def execute_single_tool(tool_call):
        
        name = tool_call['function']['name']
        args = tool_call['function']['arguments']
        
        # --- CRITICAL SAFETY GATE ---
        if name in ["search_knowledge_base", "web_search"]:
            query = args.get('query', '').strip()
            if not query:
                print(f"DEBUG: BLOCKED EMPTY QUERY for {name}", flush=True)
                return "Error: You provided an empty search query. You must specify a keyword or question."
        
        
        
        try:
            if name == "search_knowledge_base":
                res = search_knowledge_base(args.get('query', ''))
            elif name == "web_search":
                res = web_search(args.get('query', ''))
            elif name == "summarize_document":
                res = summarize_document(args.get('filename_hint', ''))
            else:
                res = f"Error: Tool '{name}' not found."
        except Exception as e:
            res = f"Error executing {name}: {str(e)}"
        return {'role': 'tool', 'content': res, 'name': name}

    try:
        # Initial indicator
        init_payload = {'id': 'chatcmpl', 'choices': [{'delta': {'content': '🔍 Thinking...\n'}}]}
        yield f"data: {json.dumps(init_payload)}\n\n"

        max_loops = 5
        for loop_count in range(1, max_loops + 1):
            print(f"\n--- DEBUG: Agent Stream Loop {loop_count} Starting ---", flush=True)
            
            messages = prune_context(messages)
            
            # Execute blocking LLM call via threadpool
            response = await run_in_threadpool(
                ollama_client.chat,
                model=CHAT_MODEL, 
                messages=messages, 
                tools=AGENT_TOOLS,
                stream=False 
            )
            
            agent_message = response.get('message', {})
            messages.append(agent_message)

            if agent_message.get('tool_calls'):
                tool_calls = agent_message['tool_calls']
                
                # 1. Yield thoughts/status
                if agent_message.get('content'):
                    thought_payload = {'id': 'chatcmpl', 'choices': [{'delta': {'content': agent_message['content'] + '\n'}}]}
                    yield f"data: {json.dumps(thought_payload)}\n\n"
                
                for tc in tool_calls:
                    t_name = tc['function']['name']
                    status_payload = {'id': 'chatcmpl', 'choices': [{'delta': {'content': f'*(Executing {t_name}...)*\n'}}]}
                    yield f"data: {json.dumps(status_payload)}\n\n"
                    
                    # 2. Execute and Summarize
                    raw_result = await run_in_threadpool(execute_single_tool, tc)
                    
                    # --- NEW: COMPRESSION STEP ---
                    # We store the summary in the history, keeping the memory lean
                    summary_content = await run_in_threadpool(summarize_tool_output, t_name, raw_result['content'])
                    
                    messages.append({
                        'role': 'tool', 
                        'content': summary_content,
                        'name': t_name
                    })
                    print(f"DEBUG: Summarized and appended {t_name} output.", flush=True)
                
                # 3. Reminder
                messages.append({
                    'role': 'user', 
                    'content': "[SYSTEM REMINDER: Review the summarized tool data above. Fulfill all parts of the user's request.]"
                })
                continue

            else:
                # Final Answer Phase
                final_text = agent_message.get('content', '')
                if final_text:
                    words = final_text.split(" ")
                    for i in range(0, len(words), 3):
                        chunk = " ".join(words[i:i+3]) + " "
                        chunk_payload = {'id': 'chatcmpl', 'choices': [{'delta': {'content': chunk}}]}
                        yield f"data: {json.dumps(chunk_payload)}\n\n"
                        await asyncio.sleep(0.05) 
                
                yield "data: [DONE]\n\n"
                return 

        timeout_payload = {'id': 'chatcmpl', 'choices': [{'delta': {'content': '\n\nAgent loop timeout.'}}]}
        yield f"data: {json.dumps(timeout_payload)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"DEBUG: !!! FATAL STREAM CRASH !!! -> {str(e)}", flush=True)
        err_payload = {'error': str(e)}
        yield f"data: {json.dumps(err_payload)}\n\n"
        yield "data: [DONE]\n\n"
        

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    user_query = request.messages[-1].content
    
    # 1. Grab the messages safely
    messages = [{'role': m.role, 'content': m.content} for m in request.messages]
    
    # --- QWEN IDENTITY OVERRIDE ---
    qwen_override = (
            "You are a precise Agentic AI. "
            "CRITICAL KNOWLEDGE RETRIEVAL RULES:\n"
            "1. If you are looking for a specific fact about a known entity or file (e.g., 'Konrad's favorite color'), "
            "and your broad 'search_knowledge_base' keeps returning long bio documents, you MUST use the "
            "'summarize_document' tool to target specific files (like 'bio' or 'personal').\n"
            "2. DO NOT repeat the same query if it yields 4,000+ characters of irrelevant text. "
            "Change your strategy to 'summarize_document' immediately."
        )
    


    if messages and messages[0]['role'] == 'system':
        messages[0]['content'] = qwen_override + "\n\n" + messages[0]['content']
    else:
        messages.insert(0, {'role': 'system', 'content': qwen_override})


    # --- INTERCEPTOR 0: CHITCHAT & GREETINGS (FIXES THE "HI" BUG) ---
    chitchat_keywords = ["hi", "hello", "hey", "how are you", "who are you", "what's up", "good morning", "greetings"]
    if len(user_query.split()) <= 3 and any(word in user_query.lower() for word in chitchat_keywords):
        print("DEBUG: Conversational greeting detected. Bypassing database and web searches.", flush=True)
        return {
            "id": "chatcmpl",
            "choices": [{"message": {"role": "assistant", "content": "Hello! How can I help you today?"}}]
        }

    
    # --- THE AGENTIC LOOP (ReAct Architecture) ---
    
    # Check if the frontend requested a stream

    if request.stream:
        return StreamingResponse(
                    agent_stream_generator(request.messages, qwen_override), 
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )


    else:
    
        max_loops = 5
        loop_count = 0

        while loop_count < max_loops:
            loop_count += 1
            print(f"\n--- DEBUG: Agent Loop {loop_count} Starting ---", flush=True)

            # 1. Ask Qwen what it wants to do
            response = await run_in_threadpool(
                ollama_client.chat,
                model=CHAT_MODEL, 
                messages=messages, 
                tools=AGENT_TOOLS
            )
            
            agent_message = response.get('message', {})
            
            # 2. If no tool calls, the agent has found the final answer! Break the loop.
            if not agent_message.get('tool_calls'):
                print("DEBUG: Agent finished reasoning. Returning final answer to user.", flush=True)
                return {
                    "id": "chatcmpl", 
                    "choices": [{"message": {"role": "assistant", "content": agent_message.get('content', '')}}]
                }

            # 3. If it called tools, save its thought process to history
            messages.append(agent_message)

            # 4. Execute the requested tools
            for tool_call in agent_message['tool_calls']:
                tool_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                
                print(f"DEBUG: Agent elected to invoke -> {tool_name} with args: {arguments}", flush=True)

                # Route the execution natively
                tool_result = ""
                if tool_name == "search_knowledge_base":
                    tool_result = await run_in_threadpool(search_knowledge_base, arguments.get('query', ''))
                elif tool_name == "web_search":
                    tool_result = await run_in_threadpool(web_search, arguments.get('query', ''))
                elif tool_name == "summarize_document":
                    tool_result = await run_in_threadpool(summarize_document, arguments.get('filename_hint', ''))
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."

                # 5. Feed the result back into the agent's context history
                messages.append({
                    'role': 'tool',
                    'content': tool_result,
                    'name': tool_name
                })
                
                print(f"DEBUG: Fed {len(tool_result)} chars of observation data back to Agent.", flush=True)

        # Fallback if the agent gets stuck in an infinite loop
        return {
            "id": "chatcmpl", 
            "choices": [{"message": {"role": "assistant", "content": "I apologize, but I had to stop thinking because the task was taking too many steps."}}]
        }
