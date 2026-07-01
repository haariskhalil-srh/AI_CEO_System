import os
import time
import torch
import gc
import json
import yfinance as yf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
import warnings

warnings.filterwarnings("ignore")

print("Initializing Llama 3.1 8B Semantic Router Agent")

# 1. Initialize Base LLM Pipeline
local_model_path = "/home/jovyan/models/meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=1024, 
    return_full_text=False, 
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    device=0
)

# 2. Initialize RAG Subsystem
embedder = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
db = Chroma(persist_directory="./chroma_db", embedding_function=embedder)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})

# 3. Define Graph State
class AgentState(TypedDict):
    query: str
    chat_history: list
    intent: str
    tool_data: str
    final_response: str

# 4. Node 1: Intent Classification
def route_query(state: AgentState) -> AgentState:
    query = state["query"]
    history = state.get("chat_history", [])
    
    # Build a quick string of recent history for the router
    history_str = ""
    for msg in history[-3:]: 
        role = "CEO" if msg["role"] == "user" else "Agent"
        history_str += f"{role}: {msg['content']}\n"
        
    messages = [
        {"role": "system", "content": "You are a routing classification engine. You must output EXACTLY ONE WORD from this list: STOCK, DATABASE, CHAT. Do not output any other text, explanation, or punctuation."},
        {"role": "user", "content": f"Recent conversation:\n{history_str}\n\nClassify this latest query: {query}\nOptions:\n- STOCK (asks about stock, market cap, NVDA)\n- DATABASE (asks about NVIDIA strategy, news, chips, competitors)\n- CHAT (greetings, general chat)"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(prompt)[0]['generated_text'].strip().upper()
    
    if "STOCK" in output: intent = "STOCK"
    elif "DATABASE" in output: intent = "DATABASE"
    else: intent = "CHAT"
        
    print(f"[ROUTER DECISION (Raw Output: {output})]: {intent}")
    return {"intent": intent}

# 5. Node 2: Deterministic Tool Execution
def execute_tool(state: AgentState) -> AgentState:
    intent = state["intent"]
    query = state["query"]
    tool_data = ""

    if intent == "STOCK":
        try:
            stock = yf.Ticker("NVDA")
            data = stock.history(period="1d")
            price = data['Close'].iloc[0]
            vol = data['Volume'].iloc[0]
            tool_data = f"REAL-TIME DATA: NVIDIA (NVDA) Current Price: ${price:.2f}, Volume: {vol}"
        except Exception as e:
            tool_data = f"Failed to fetch stock: {e}"
            
    elif intent == "DATABASE":
        docs = retriever.invoke(query)
        tool_data = "\n\n".join([f"Date: {d.metadata.get('date', 'Unknown')}\nText: {d.page_content}" for d in docs])
        if not tool_data:
            tool_data = "No internal data found."
            
    else:
        tool_data = "No tools needed. Just chat."

    return {"tool_data": tool_data}

# 6. Node 3: Synthesis
def generate_response(state: AgentState) -> AgentState:
    query = state["query"]
    tool_data = state["tool_data"]
    history = state.get("chat_history", [])
    
    messages = [
        {"role": "system", "content": f"You are a strict NVIDIA executive advisor. You only discuss NVIDIA, AI, financial markets, and corporate strategy. If the user asks general knowledge, personal, or trivia questions (e.g., capitals, recipes, history), you MUST refuse to answer, state that the query is out of scope, and remind them of your role. \nContext Data: {tool_data}\nRule: Do NOT write notes, meta-commentary, or explain your tone. Just provide the answer."}
    ]
    
    # Inject conversational memory into the LLM context
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
        
    messages.append({"role": "user", "content": query})
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(prompt)[0]['generated_text'].strip()
    
    return {"final_response": output}

# 7. Compile the LangGraph State Machine
workflow = StateGraph(AgentState)
workflow.add_node("router", route_query)
workflow.add_node("executor", execute_tool)
workflow.add_node("synthesizer", generate_response)

workflow.set_entry_point("router")
workflow.add_edge("router", "executor")
workflow.add_edge("executor", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()
print("Semantic Router Agent active. Listening for queries...")

# 8. Inter-Process Communication Loop
while True:
    if os.path.exists("pending_query.json"):
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            with open("pending_query.json", "r", encoding="utf-8") as f:
                chat_history_raw = json.load(f)
                
            latest_query = chat_history_raw[-1]["content"]
            formatted_history = chat_history_raw[:-1] # Everything except the current query
            print(f"\n[RECEIVED QUERY]: {latest_query}")
            
            # Run the Graph with memory injected
            result = app.invoke({"query": latest_query, "chat_history": formatted_history})
            final_answer = result["final_response"]
            
            # === FIX: CRITICAL SUCCESS HANDLING CODE WAS MISSING ===
            # Write Response atomically so Streamlit can read it
            with open("temp_response.txt", "w", encoding="utf-8") as f:
                f.write(final_answer)
                
            os.replace("temp_response.txt", "agent_response.txt") 
            os.remove("pending_query.json") # Unlocks the loop
            print("[RESPONSE SENT]")
            
        except Exception as e:
            print(f"Worker Error: {e}")
            with open("temp_response.txt", "w", encoding="utf-8") as f:
                f.write(f"Execution error: {e}")
            os.replace("temp_response.txt", "agent_response.txt") 
            if os.path.exists("pending_query.json"): 
                os.remove("pending_query.json")
                
    time.sleep(1)