# Modern LangChain (v1.x) & LangGraph Masterclass

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain v1.x](https://img.shields.io/badge/LangChain-1.x-brightgreen.svg)](https://docs.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Engine-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A rigorous, hands-on production reference covering modern **LangChain (v1.x)** architecture, multi-provider model routing, typed tool calling, standard message ontologies, schema-enforced structured outputs, and stateful agent middleware with **Human-in-the-Loop (HITL)** controls.

---

## Table of Contents

- [Architectural Paradigm: Legacy vs. Modern LangChain (v1.x)](#architectural-paradigm-legacy-vs-modern-langchain-v1x)
- [Repository Structure](#repository-structure)
- [Module Deep Dives & Engineering Notes](#module-deep-dives--engineering-notes)
  - [1. Introduction to LangChain v1 & Agents (`1-langchainintro.ipynb`)](#1-introduction-to-langchain-v1--agents-1-langchainintroipynb)
  - [2. Multi-Provider Model Integrations & Inference (`2-modelintegration.ipynb`)](#2-multi-provider-model-integrations--inference-2-modelintegrationipynb)
  - [3. Tool Calling & The Agent Execution Loop (`3-tools.ipynb`)](#3-tool-calling--the-agent-execution-loop-3-toolsipynb)
  - [4. Cross-Provider Message Architecture (`4-messages.ipynb`)](#4-cross-provider-message-architecture-4-messagesipynb)
  - [5. Schema Enforcement & Structured Outputs (`5-structured output.ipynb`)](#5-schema-enforcement--structured-outputs-5-structured-outputipynb)
  - [6. Stateful Agent Middleware & HITL Control (`6-middleware.ipynb`)](#6-stateful-agent-middleware--hitl-control-6-middlewareipynb)
- [Getting Started & Installation](#getting-started--installation)
- [Environment Configuration](#environment-configuration)
- [Engineering Notes & Windows Troubleshooting](#engineering-notes--windows-troubleshooting)
- [License](#license)

---

## Architectural Paradigm: Legacy vs. Modern LangChain (v1.x)

In earlier iterations of LangChain (v0.1 - v0.2), workflows relied heavily on brittle pipeline abstractions such as `LLMChain`, `AgentExecutor`, and verbose string-based prompt templates. 

**Modern LangChain (v1.x)** fundamentally refactors agent design around **state machines backed by LangGraph**:

```
+-----------------------------------------------------------------------------------+
|                            Modern LangChain (v1.x) System                         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ User Input / Messages ]                                                        |
|             |                                                                     |
|             v                                                                     |
|  +-----------------------+       Checkpointing       +-------------------------+  |
|  |     create_agent      | <=======================> | InMemorySaver / Postgres|  |
|  | (CompiledStateGraph)  |    (Thread Isolation)     +-------------------------+  |
|  +-----------------------+                                                        |
|             |                                                                     |
|             +--- Middleware Layer (Pre/Post hooks, Summarization, HITL)           |
|             |                                                                     |
|             v                                                                     |
|  +-----------------------+       Tool Schemas        +-------------------------+  |
|  |    Chat Model Layer   | ------------------------> |   @tool Decorated       |  |
|  |  (init_chat_model)    | <------------------------ |   Python Functions      |  |
|  +-----------------------+        Tool Results       +-------------------------+  |
|             |                                                                     |
|             v                                                                     |
|  [ Structured / Streamed Response ]                                               |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### Key Differences at a Glance

| Feature | Legacy LangChain (v0.x) | Modern LangChain (v1.x) |
| :--- | :--- | :--- |
| **Agent Engine** | `AgentExecutor` (rigid loop, opaque state) | `create_agent` compiling to a `CompiledStateGraph` (LangGraph) |
| **Model Creation** | Explicit vendor classes (`ChatOpenAI`, `ChatAnthropic`) | Universal factory `init_chat_model("provider:model")` |
| **State Persistence** | Memory classes (`ConversationBufferMemory`) | Checkpointers (`InMemorySaver`, `PostgresSaver`) via `thread_id` |
| **Tool Definition** | StructuredTool subclasses or decorators | Lightweight `@tool` with native type annotations & Pydantic args |
| **Flow Control** | Hardcoded loop mechanisms | Modular **Middleware** (token summarization, interrupt approvals) |
| **Output Parsing** | Regex / OutputParsers | Native model-level `.with_structured_output()` |

---

## Repository Structure

```text
langchainupdated/
├── .env.example                     # Sanitized environment template for credentials
├── .gitignore                       # Git ignore preventing secret and cache leakage
├── pyproject.toml                   # Project metadata, dependencies, and uv config
├── requirements.txt                 # Legacy pip dependency manifest
├── README.md                        # Master documentation and engineering notes
└── updatedlangchain/                # Hands-on interactive engineering notebooks
    ├── 1-langchainintro.ipynb       # LangChain v1, create_agent, and agent invocation
    ├── 2-modelintegration.ipynb     # OpenAI, Gemini, Groq, Cloudflare, streaming & batching
    ├── 3-tools.ipynb                # @tool decorator, tool binding, and multi-step agent loops
    ├── 4-messages.ipynb             # System, Human, AI, Tool messages and multimodal payloads
    ├── 5-structured output.ipynb    # Pydantic, TypedDict, DataClasses schema enforcement
    └── 6-middleware.ipynb           # SummarizationMiddleware & HumanInTheLoopMiddleware (HITL)
```

---

## Module Deep Dives & Engineering Notes

### 1. Introduction to LangChain v1 & Agents (`1-langchainintro.ipynb`)

#### Concepts
- **`create_agent`**: The modern entry point for building reactive conversational agents. Behind the scenes, it constructs and compiles a LangGraph state graph.
- **Tools Binding**: Functions decorated with docstrings and type annotations are dynamically parsed into JSON Schema tool declarations that models consume.
- **Invocation Interface**: Unified schema accepting standard `{"messages": [...]}` payloads.

#### Core Pattern
```python
import os
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

def get_weather(city: str) -> str:
    """Get the current weather for a specified city."""
    return f"The weather in {city} is sunny and 22°C."

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant with real-time tool access."
)

# Invocation with user message
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in New York?"}]
})

# Access the final synthesized AI message
final_message = response["messages"][-1]
print(final_message.content)
```

> **Pro Tip**: The output returned by `agent.invoke()` is a state dictionary containing the complete conversational trace (`response["messages"]`). The final element is always the model's concluding `AIMessage`.

---

### 2. Multi-Provider Model Integrations & Inference (`2-modelintegration.ipynb`)

#### Concepts
- **`init_chat_model`**: A single unified abstraction allowing runtime switching between providers (`openai`, `google_genai`, `groq`, `anthropic`) without changing downstream code.
- **OpenAI-Compatible Custom Endpoints**: Utilizing `ChatOpenAI` or `init_chat_model` with custom `base_url` to target private gateways, local vLLM instances, or Cloudflare Workers AI.
- **Inference Styles**:
  - `.invoke()`: Synchronous, single-turn completion.
  - `.stream()`: Chunk-by-chunk generator yielding `AIMessageChunk` objects for low latency UI rendering.
  - `.batch()`: Parallel execution of multiple distinct inputs for high-throughput batch workloads.

#### Core Patterns

##### Universal Model Initialization & OpenAI-Compatible Gateways
```python
import os
from langchain.chat_models import init_chat_model

# 1. Google Gemini via init_chat_model
gemini_model = init_chat_model("google_genai:gemini-3.5-flash-lite")

# 2. Groq Cloud for ultra-low latency inference
groq_model = init_chat_model("groq:qwen/qwen3.8-27b")

# 3. Cloudflare Workers AI via OpenAI-compatible route
cloudflare_model = init_chat_model(
    "@cf/zai-org/glm-4.7-flash",
    model_provider="openai",
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CLOUDFLARE_ACCOUNT_ID')}/ai/v1",
    api_key=os.getenv("CLOUDFLARE_API_TOKEN")
)
```

##### Streaming vs. Batching
```python
# Real-time token streaming
for chunk in groq_model.stream("Explain quantum computing in three sentences:"):
    print(chunk.content, end="", flush=True)

# Parallel batch processing
batch_prompts = [
    "Why is the sky blue?",
    "Why do leaves change color in autumn?",
    "How does encryption work?"
]
responses = groq_model.batch(batch_prompts)
for res in responses:
    print(res.content[:80], "...")
```

---

### 3. Tool Calling & The Agent Execution Loop (`3-tools.ipynb`)

#### The Core Execution Mechanism
Language models cannot execute code directly. When a model uses a tool, it generates a **tool call request** containing the tool name and validated arguments. The runtime executes the Python function and injects the result back as a `ToolMessage`.

```
User Query ---> [ Model.invoke ] ---> Returns AIMessage(tool_calls=[...])
                                                |
                                                v
               [ Python Runtime ] <--- Extract tool name & arguments
                      |
                      v
             Executes Local Function
                      |
                      v
[ Model.invoke ] <--- Injects ToolMessage(content=result, tool_call_id=id)
       |
       v
Returns Final AIMessage with Synthesized Answer
```

#### Core Pattern
```python
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def calculate_compound_interest(principal: float, rate: float, years: int) -> float:
    """Calculate the compound interest given principal, annual rate (decimal), and years."""
    return round(principal * ((1 + rate) ** years), 2)

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
model_with_tools = model.bind_tools([calculate_compound_interest])

# Step 1: Model inspects prompt and issues tool call
messages = [HumanMessage(content="What is the compound interest on $10,000 at 5% for 3 years?")]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Runtime executes tool
for tool_call in ai_msg.tool_calls:
    if tool_call["name"] == "calculate_compound_interest":
        result = calculate_compound_interest.invoke(tool_call["args"])
        # Step 3: Append ToolMessage linked via tool_call_id
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

# Step 4: Final synthesis
final_response = model_with_tools.invoke(messages)
print(final_response.content)
```

---

### 4. Cross-Provider Message Architecture (`4-messages.ipynb`)

#### Standardized Message Hierarchy
Modern LangChain provides universal message representations ensuring code portability across providers:

- **`SystemMessage`**: Injected at the conversation root to enforce personas, security guardrails, and behavioral boundaries.
- **`HumanMessage`**: Represents user queries. Supports multimodal inputs (text, image URLs, base64 data).
- **`AIMessage`**: Responses emitted by the LLM. May include reasoning tokens (`additional_kwargs`), text, or `tool_calls`.
- **`ToolMessage`**: The execution output of an external tool, mapped to an `AIMessage` via `tool_call_id`.

#### Core Pattern

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Multi-turn conversation with explicit roles
conversation = [
    SystemMessage(content="You are a senior financial analyst. Answer with precise metrics."),
    HumanMessage(content="What are the key drivers of gross margin?"),
    AIMessage(content="Gross margin is primarily driven by COGS, volume discounts, and pricing power."),
    HumanMessage(content="Summarize in bullet points.")
]

response = model.invoke(conversation)
```

##### Multimodal Payloads (Vision)
```python
multimodal_message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the contents and structure of this diagram:"},
        {
            "type": "image_url",
            "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"}
        }
    ]
)
response = gemini_model.invoke([multimodal_message])
```

---

### 5. Schema Enforcement & Structured Outputs (`5-structured output.ipynb`)

#### Why Structured Outputs?
Unstructured text responses are prone to format drift, parsing failures, and hallucinated keys. LangChain's `.with_structured_output()` constrains the LLM at the decoding/tool-calling level to guarantee conforming data.

#### Three Implementation Paradigms

```
                      +---------------------------------------+
                      |       with_structured_output()        |
                      +---------------------------------------+
                                          |
         +--------------------------------+-------------------------------+
         |                                |                               |
         v                                v                               v
+------------------+             +------------------+            +------------------+
|     Pydantic     |             |    TypedDict     |            |    Dataclass     |
|   (BaseModel)    |             |   (PEP 589/655)  |            |  (@dataclass)    |
| - Runtime check  |             | - Zero overhead  |            | - Clean syntax   |
| - Complex nested |             | - Builtin dict   |            | - Python native  |
| - Type coercion  |             | - Annotated docs |            | - Static typing  |
+------------------+             +------------------+            +------------------+
```

#### Code Comparisons

##### 1. Pydantic (Recommended for Production & Validation)
```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str = Field(description="Actor's full name")
    role: str = Field(description="Character portrayed")

class MovieDetails(BaseModel):
    title: str = Field(description="Official movie title")
    release_year: int = Field(description="Year of theatrical release")
    rating: float = Field(ge=0, le=10, description="Rating from 0 to 10")
    cast: list[Actor] = Field(description="Key cast members")

structured_model = model.with_structured_output(MovieDetails)
movie = structured_model.invoke("Provide details about Inception directed by Christopher Nolan.")
print(movie.title, movie.release_year, movie.cast[0].name)
```

##### 2. TypedDict (Fast, Zero Validation Overhead)
```python
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    title: Annotated[str, ..., "The movie title"]
    release_year: Annotated[int, ..., "Release year"]
    director: Annotated[str, ..., "Director name"]

dict_model = model.with_structured_output(MovieDict)
result = dict_model.invoke("Provide details about The Matrix.")
print(type(result), result["title"])  # Returns standard Python dict
```

##### 3. DataClasses
```python
from dataclasses import dataclass

@dataclass
class ContactCard:
    """Contact details extracted from conversation."""
    name: str
    email: str
    phone: str

dc_model = model.with_structured_output(ContactCard)
contact = dc_model.invoke("Contact John Doe at john@example.com or +1-555-0199")
print(contact.name, contact.email)
```

---

### 6. Stateful Agent Middleware & HITL Control (`6-middleware.ipynb`)

#### What is Middleware in LangChain?
Middleware acts as an interceptor in the agent lifecycle, controlling context manipulation, conversation compression, and human safety checkpoints without modifying the core agent logic.

#### 1. Summarization Middleware (`SummarizationMiddleware`)
Prevents context window exhaustion in long conversations by periodically compressing historical messages into an executive summary while preserving recent context.

- **Trigger Options**:
  - `trigger=("tokens", 550)`: Summarizes when total token count reaches threshold.
  - `trigger=("messages", 10)`: Summarizes when message count reaches threshold.
  - `trigger=("fraction", 0.8)`: Summarizes when context reaches 80% of window limit.
- **Retention**:
  - `keep=("tokens", 200)` or `keep=("messages", 4)`: Preserves the most recent messages verbatim.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="groq:openai/gpt-oss-20b",   # Lightweight model for summarization
            trigger=("messages", 10),           # Compress after 10 messages
            keep=("messages", 4)                # Retain last 4 messages intact
        )
    ]
)

# Thread ID isolates memory sessions
config = {"configurable": {"thread_id": "session-101"}}
```

#### 2. Human-in-the-Loop Middleware (`HumanInTheLoopMiddleware`)
Designed for high-consequence operations (financial transactions, email dispatch, database writes). It halts execution when designated tools are called and yields an `__interrupt__` signal.

```
Agent requests sensitive tool execution
                |
                v
+-------------------------------+
|  HumanInTheLoopMiddleware     | ---> HALTS EXECUTION (__interrupt__)
+-------------------------------+
                |
                v
      [ Human reviews request ]
                |
    +-----------+-----------+
    |           |           |
    v           v           v
[ APPROVE ]  [ REJECT ]  [ EDIT ]
    |           |           |
    v           v           v
Execute tool   Abort tool  Mutate args & execute
```

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

agent = create_agent(
    model="groq:openai/gpt-oss-20b",
    tools=[send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approved", "edit", "reject"]
                }
            }
        )
    ]
)

config = {"configurable": {"thread_id": "approval-flow-1"}}

# Turn 1: Triggers interrupt
result = agent.invoke(
    {"messages": [HumanMessage(content="Send an email to ceo@company.com with subject 'Q3' and body 'Profits up 20%'.")]},
    config=config
)

# Verify interrupt state
if "__interrupt__" in result:
    print("Execution paused. Pending human decision...")

# Turn 2: Resume with decision
# Option A: Approve
resume_decision = Command(resume={"decisions": [{"type": "approve"}]})

# Option B: Reject
# resume_decision = Command(resume={"decisions": [{"type": "reject"}]})

# Option C: Edit arguments before execution
# resume_decision = Command(resume={
#     "decisions": [{
#         "type": "edit",
#         "edited_action": {
#             "name": "send_email_tool",
#             "args": {"recepient": "ceo@company.com", "subject": "Reviewed Q3", "body": "Profits verified."}
#         }
#     }]
# })

final_result = agent.invoke(resume_decision, config=config)
print(final_result["messages"][-1].content)
```

---

## Getting Started & Installation

### 1. Prerequisites
- **Python**: Version `3.13` or higher
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/langchainupdated.git
cd langchainupdated
```

### 3. Create Virtual Environment & Install Dependencies

#### Using `uv` (Fastest)
```bash
uv venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

uv pip install -e .
```

#### Using standard `pip`
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Environment Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Populate `.env` with your active provider credentials:
   ```env
   OPENAI_API_KEY="sk-..."
   GROQ_API_KEY="gsk_..."
   GOOGLE_API_KEY="AIza..."
   CLOUDFLARE_API_TOKEN="..."
   CLOUDFLARE_ACCOUNT_ID="..."
   ```

> [!CAUTION]
> **Never commit your `.env` file to Git.** Verify that `.gitignore` contains `.env` before pushing to any public remote.

---

## Engineering Notes & Windows Troubleshooting

### Windows Smart App Control (SAC) & `xxhash` / `.pyd` Extensions
On Windows 11 systems with **Smart App Control (SAC)** enabled (`VerifiedAndReputablePolicyState = 1`), Windows blocks unsigned or unverified binary DLL / `.pyd` files (such as `_xxhash.pyd` installed with `langsmith` / `xxhash`).

If you encounter:
```text
ImportError: DLL load failed while importing _xxhash: An Application Control policy has blocked this file.
```

**Resolution**:
A pure-Python fallback (leveraging standard library `hashlib`) has been introduced in the local `.venv\Lib\site-packages\xxhash\__init__.py` to gracefully degrade when `_xxhash.pyd` is intercepted by code integrity policies. Alternatively, whitelist the project root in **Windows Security > Virus & threat protection > Exclusions**.

---

## License

This project is licensed under the [MIT License](LICENSE).
