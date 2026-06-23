# NVIDIA Strategic Intelligence Agent

An automated, local-first executive advisory system designed to monitor, ingest, and analyze strategic intelligence regarding NVIDIA Corporation. The system leverages Retrieval-Augmented Generation (RAG) and zero-shot LLM classification to generate structured business intelligence dashboards — without relying on external cloud APIs.

---

## System Architecture

This architecture processes multi-source intelligence through a local vector pipeline, ensuring zero corporate data leakage.

```mermaid
flowchart TD
    classDef python fill:#306998,stroke:#FFE873,stroke-width:2px,color:#fff;
    classDef data fill:#1a1a1a,stroke:#76b900,stroke-width:2px,color:#fff;
    classDef model fill:#005522,stroke:#76b900,stroke-width:2px,color:#fff;
    classDef db fill:#4B0082,stroke:#9370DB,stroke-width:2px,color:#fff;

    subgraph Layer 1: Data Collection
        YF(Yahoo Finance) --> C[collector.ipynb]:::python
        HN(HackerNews) --> C
        RSS(NVIDIA RSS Feed) --> C
        C -->|Raw JSONL| RawData{nvidia_raw_intelligence.jsonl}:::data
    end

    subgraph Layer 2: Preprocessing & Ingestion
        RawData --> CL[cleaner.ipynb]:::python
        CL -->|Deduplication & Thresholds| CleanData{nvidia_clean_intelligence.jsonl}:::data
        CleanData --> P[processor.ipynb]:::python
        P -->|1000 char / 200 overlap| E[BAAI/bge-base-en-v1.5]:::model
        E -->|768-Dim Vectors| VDB[(ChromaDB)]:::db
    end

    subgraph Layer 3: Agentic Inference
        VDB -.->|Semantic Retrieval k=10| A[agent.ipynb]:::python
        LLM[Llama 3.1 8B Instruct]:::model -->|bfloat16| A
        A -->|Strict JSON Prompt| Report{ceo_intelligence_report.json}:::data
    end

    subgraph Layer 4: Presentation
        Report --> D[dashboard.py]:::python
        D -->|Streamlit App| UI([NVIDIA CEO Dashboard]):::data
    end
```

---

## Data Flow

The following sequence demonstrates the lifecycle of unstructured web data as it is transformed into a deterministic JSON artifact.

```mermaid
sequenceDiagram
    participant Web as Data Sources
    participant Scraper as Collection Layer
    participant ETL as Ingestion Layer
    participant VDB as ChromaDB
    participant Agent as LLM Agent
    participant UI as Streamlit UI

    Web->>Scraper: RSS, HTML, JSON
    Scraper->>ETL: Append nvidia_raw_intelligence.jsonl
    ETL->>ETL: Clean, Deduplicate, Chunk (1000 chars)
    ETL->>VDB: Embed and Store 768-Dim Vectors
    Agent->>VDB: Send Query Vector (Dense Retrieval)
    VDB->>Agent: Return Top-K (10) Context Chunks via Cosine Similarity
    Agent->>Agent: Llama 3.1 Zero-Shot Classification
    Agent->>UI: Export ceo_intelligence_report.json
    UI->>UI: Parse JSON, Render Plotly Gauges
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Large Language Model | Llama-3.1-8B-Instruct (Meta) |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Vector Database | ChromaDB |
| Data Processing | Pandas, LangChain Text Splitters |
| Data Ingestion | BeautifulSoup4, Feedparser, yfinance |
| Frontend | Streamlit, Plotly Graph Objects |

---

## Core Design Decisions

**Local-First Architecture**
Cloud-based APIs (OpenAI, Anthropic) were explicitly rejected to adhere to corporate data privacy constraints. All embedding and inference execute locally on an NVIDIA RTX A6000.

**Model Selection — Llama 3.1 8B Instruct**
The 8B parameter model was selected over the 70B variant to avoid destructive 4-bit quantization. Running 8B natively in bfloat16 retains mathematical fidelity for complex financial reasoning while easily fitting within the 48GB VRAM limit. The Instruct variant is strictly required for its tool-calling optimizations, forcing the model to output a parsable JSON schema rather than conversational text.

**Embedding Choice — BAAI/bge-base**
BAAI provides 768-dimensional dense vectors, offering significantly better semantic nuance for financial and semiconductor jargon compared to lighter 384-dimensional models (like MiniLM), without the heavy computational overhead of large-parameter encoders.

**Decoupled Frontend**
The Streamlit dashboard (`dashboard.py`) does not execute LLM inference or database queries. It strictly parses the final `ceo_intelligence_report.json` artifact, ensuring the executive UI remains highly performant and immune to underlying inference latency or VRAM exhaustion.

**Visualization Strategy**
Time-series line graphs were rejected for sentiment analysis because the LLM performs batch zero-shot analysis, yielding a current-state snapshot. Plotting a timeline would require hallucinating historical data. The UI instead uses Plotly gauge charts to accurately represent absolute bounded metrics (0 to 100).

---

## Pipeline Execution

The system operates in four distinct phases:

**Phase 1 — Collection** (`collector.ipynb`)
Ingests raw data from three tiers of intelligence:
- Ground Truth: NVIDIA Official RSS
- Market Reaction: Yahoo Finance
- Developer Sentiment: HackerNews

**Phase 2 — Preprocessing** (`cleaner.ipynb` & `processor.ipynb`)
Deduplicates records, drops low-quality strings, and uses LangChain's `RecursiveCharacterTextSplitter` to chunk documents into 1000-character segments with a 200-character overlap, preventing semantic severing.

**Phase 3 — Ingestion**
Text chunks are passed through the BAAI encoder and loaded into a persistent local ChromaDB instance.

**Phase 4 — Agentic Inference** (`agent.ipynb`)
A dense vector retrieval mechanism uses cosine similarity to extract the top 10 most relevant chunks. These are injected into the Llama 3.1 context window via a strict system prompt, forcing zero-shot sentiment classification and output into a deterministic JSON file.
