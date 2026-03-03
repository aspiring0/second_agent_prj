# 企业级 RAG 多智能体问答系统 - 完整技术文档

## 📋 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [目录结构详解](#3-目录结构详解)
4. [核心模块详解](#4-核心模块详解)
   - 4.1 [配置层 (config/)](#41-配置层-config)
   - 4.2 [RAG模块 (src/rag/)](#42-rag模块-srcrag)
   - 4.3 [Agent模块 (src/agent/)](#43-agent模块-srcagent)
   - 4.4 [Service层 (src/service/)](#44-service层-srcservice)
   - 4.5 [工具层 (src/utils/)](#45-工具层-srcutils)
5. [数据流与执行流程](#5-数据流与执行流程)
6. [核心技术知识点](#6-核心技术知识点)
7. [依赖与环境配置](#7-依赖与环境配置)
8. [使用指南](#8-使用指南)
9. [扩展与二次开发](#9-扩展与二次开发)

---

## 1. 项目概述

### 1.1 项目定位

本项目是一个**企业级 RAG（Retrieval-Augmented Generation，检索增强生成）多智能体问答系统**。它结合了：

- **RAG 技术**：通过向量检索从知识库中获取相关上下文，增强大模型的回答能力
- **多智能体架构**：基于 LangGraph 实现研究员（Researcher）和作家（Writer）两个角色的协作
- **知识库管理**：支持多知识库、多会话的隔离管理
- **Web UI**：基于 Streamlit 的交互式界面

### 1.2 核心功能

| 功能模块 | 描述 |
|---------|------|
| 文档入库 | 支持 TXT、PDF、DOCX、MD、代码文件等多种格式的文档处理与向量化入库 |
| 语义检索 | 基于向量相似度的智能检索，支持 Top-K 结果返回 |
| 多知识库隔离 | 通过 project_id 实现不同知识库的数据隔离 |
| 会话管理 | 支持多会话、会话历史持久化 |
| 多智能体协作 | 研究员负责检索信息，作家负责生成最终回答 |
| 工具扩展 | 支持通用问答、文本总结、翻译、代码分析、时间查询、数学计算等多种工具 |

### 1.3 技术栈

```
┌─────────────────────────────────────────────────────────────┐
│  前端层    │  Streamlit (web_app.py)                        │
├─────────────────────────────────────────────────────────────┤
│  服务层    │  ChatService / KnowledgeBaseService / DocumentService │
├─────────────────────────────────────────────────────────────┤
│  Agent层   │  LangGraph (StateGraph) + LangChain            │
├─────────────────────────────────────────────────────────────┤
│  RAG层     │  ETL + VectorStore + Retriever + Generator     │
├─────────────────────────────────────────────────────────────┤
│  存储层    │  ChromaDB (向量) + SQLite (会话/消息)           │
├─────────────────────────────────────────────────────────────┤
│  模型层    │  OpenAI API (GPT-3.5/4 + text-embedding-3-small) │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            用户界面 (Streamlit)                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   知识库管理页   │  │    聊天页面     │  │   文件上传组件   │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
└───────────┼────────────────────┼────────────────────┼────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          Service 层 (业务逻辑)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │KBService        │  │ChatService      │  │DocumentService  │          │
│  │ - 知识库CRUD    │  │ - 消息管理      │  │ - 文件处理      │          │
│  │ - 会话管理      │  │ - Agent调用     │  │ - 入库流程      │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
└───────────┼────────────────────┼────────────────────┼────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Agent 层 (LangGraph 状态机)                        │
│                                                                          │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐             │
│   │ Researcher  │ ───▶ │   Tools     │ ───▶ │ Researcher  │ (循环)      │
│   │ (研究员)    │      │ (工具执行)   │      │             │             │
│   └─────────────┘      └─────────────┘      └──────┬──────┘             │
│                                                    │                    │
│                              ┌─────────────────────┘                    │
│                              ▼                                          │
│                       ┌─────────────┐                                   │
│                       │   Writer    │ ───▶ END                         │
│                       │   (作家)    │                                   │
│                       └─────────────┘                                   │
└──────────────────────────────────────────────────────────────────────────┘
            │                                          │
            ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           RAG 核心层                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ ETL             │  │ VectorStore     │  │ Generator       │          │
│  │ - 文档加载      │  │ - ChromaDB      │  │ - Prompt模板    │          │
│  │ - 智能切分      │  │ - 向量持久化    │  │ - LLM调用       │          │
│  │ - 元数据管理    │  │ - 相似度检索    │  │ - 答案生成      │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└──────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           数据存储层                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────┐             │
│  │   ChromaDB (向量数据库)  │    │   SQLite (关系数据库)   │             │
│  │   - 文档向量存储         │    │   - 会话管理            │             │
│  │   - 语义检索             │    │   - 消息历史            │             │
│  │   - 元数据过滤           │    │   - 知识库元信息        │             │
│  └─────────────────────────┘    └─────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流向

```
用户提问 → ChatService → Agent Graph → Researcher节点
                                          ↓
                                    判断是否需要工具
                                          ↓
                              ┌───────────┴───────────┐
                              ↓                       ↓
                         需要工具                  不需要工具
                              ↓                       ↓
                     ToolNode执行工具           Writer节点
                     (ask_knowledge_base)            ↓
                              ↓                 生成最终回答
                     返回检索结果                     ↓
                              ↓               保存到SQLite
                     回到Researcher                    ↓
                              ↓                  返回用户
                     继续判断/转到Writer
```

---

## 3. 目录结构详解

```
Enterprise_RAG_Agent/
│
├── config/                          # 【配置层】全局配置管理
│   ├── __init__.py                  # 配置模块初始化
│   └── settings.py                  # 集中管理 API Key、路径、模型参数
│
├── data/                            # 【数据层】
│   ├── raw/                         # 存放原始文档 (PDF, MD, TXT)
│   └── vector_db/                   # ChromaDB 持久化存储目录（自动生成）
│
├── logs/                            # 【日志层】系统运行日志（自动生成）
│   └── app.log                      # 轮转日志文件
│
├── scripts/                         # 【脚本层】运维/初始化脚本
│   └── ingest_knowledge.py          # 知识入库入口脚本（命令行）
│
├── src/                             # 【核心代码层】
│   ├── __init__.py
│   │
│   ├── rag/                         # RAG 核心模块
│   │   ├── __init__.py
│   │   ├── etl.py                   # Extract, Transform (文档加载与切分)
│   │   ├── vectorstore.py           # Load (向量库管理)
│   │   ├── retriever.py             # Retrieve (语义检索)
│   │   └── generator.py             # Generate (答案生成)
│   │
│   ├── agent/                       # 多智能体模块
│   │   ├── __init__.py
│   │   ├── state.py                 # AgentState 状态定义
│   │   ├── tools.py                 # 工具函数定义
│   │   ├── nodes.py                 # Agent节点实现
│   │   └── graph.py                 # LangGraph 状态机编排
│   │
│   ├── service/                     # 服务层（业务逻辑）
│   │   ├── __init__.py              # 服务层导出
│   │   ├── kb_service.py            # 知识库管理服务
│   │   ├── chat_service.py          # 聊天服务
│   │   └── document_service.py      # 文档处理服务
│   │
│   └── utils/                       # 通用工具
│       ├── __init__.py
│       ├── db.py                    # SQLite 数据库操作
│       ├── logger.py                # 统一日志配置
│       └── model_manager.py         # 多模型管理器 (动态切换模型)
│
├── web_app.py                       # Streamlit Web应用入口
├── main.py                          # 命令行交互入口
├── .env                             # 环境变量（API密钥等）
├── .gitignore                       # Git忽略配置
├── chat_history.db                  # SQLite数据库文件（自动生成）
└── README.md                        # 项目说明
```

---

## 4. 核心模块详解

### 4.1 配置层 (config/)

#### `config/settings.py`

**职责**：集中管理所有配置项，实现配置与代码分离。

```python
class Settings:
    # 基础路径配置
    BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录
    DATA_DIR = BASE_DIR / "data" / "raw"               # 原始文档目录
    DB_DIR = BASE_DIR / "data" / "vector_db"           # 向量数据库目录
    LOG_DIR = BASE_DIR / "logs"                        # 日志目录

    # LLM & Embedding 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")       # API密钥
    OPENAI_BASE_URL = os.getenv("OPENAI_API_BASE")     # API地址（支持代理/兼容接口）
    
    # 模型选择
    EMBEDDING_MODEL = "text-embedding-3-small"         # 向量化模型
    CHAT_MODEL = "gpt-3.5-turbo"                       # 对话模型
    
    # RAG 业务参数
    CHUNK_SIZE = 800           # 切片大小（字符数）
    CHUNK_OVERLAP = 100        # 切片重叠（防止语义断裂）
```

**设计模式**：单例模式 + 集中配置

**知识点**：
- `pathlib.Path`：跨平台路径处理
- `python-dotenv`：从 `.env` 文件加载环境变量
- 配置分离原则：敏感信息（API Key）与代码分离

---

### 4.2 RAG模块 (src/rag/)

RAG 是本系统的核心，分为四个子模块：

#### 4.2.1 ETL模块 (`etl.py`)

**职责**：文档的加载（Extract）与切分（Transform）。

**核心类**：`ContentProcessor`

```python
class ContentProcessor:
    def load_uploaded_files(self, uploaded_files) -> List[Document]:
        """加载上传的文件，返回文档列表"""
        
    def split_documents(self, documents) -> List[Document]:
        """智能切分文档"""
```

**关键技术点**：

1. **多格式文档加载器**：
   ```python
   # 根据文件后缀选择加载器
   .txt  → TextLoader (支持UTF-8/GBK编码自动检测)
   .md   → UnstructuredMarkdownLoader
   .pdf  → PyPDFLoader
   .docx → Docx2txtLoader
   .py/.js/.java 等 → TextLoader
   ```

2. **临时文件技术**：
   - Streamlit上传的文件在内存中
   - LangChain Loader需要文件路径
   - 使用 `tempfile.NamedTemporaryFile` 桥接

3. **智能切分策略**：
   ```python
   # 中文文档分隔符（优先级从高到低）
   CHINESE_SEPARATORS = [
       "\n\n",    # 段落分隔
       "\n",      # 行分隔
       "。",      # 中文句号
       "！", "？", "；", "，", "：",
       ...
   ]
   
   # Markdown文档分隔符
   MARKDOWN_SEPARATORS = [
       "\n\n", "```", "## ", "### ", ...
   ]
   ```

4. **RecursiveCharacterTextSplitter**：
   - 递归地尝试不同的分隔符
   - 直到找到能够保持语义完整的切分点
   - `chunk_overlap` 保证块之间有重叠，防止关键信息被切断

**数据结构**：
```python
Document {
    page_content: str,      # 文档内容
    metadata: {
        source: str,        # 来源文件名
        file_type: str,     # 文件类型
        project_id: str     # 所属知识库ID
    }
}
```

---

#### 4.2.2 向量存储模块 (`vectorstore.py`)

**职责**：文档向量化与持久化存储。

**核心类**：`VectorDBManager`

```python
class VectorDBManager:
    def __init__(self):
        self.embedding_fn = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=...,
            openai_api_base=...
        )
    
    def create_vector_db(self, chunks, project_id: str):
        """将文档块存入向量数据库"""
```

**关键技术点**：

1. **向量嵌入（Embedding）**：
   - 将文本转换为高维向量（如1536维）
   - 语义相近的文本在向量空间中距离较近
   - 使用 OpenAI 的 `text-embedding-3-small` 模型

2. **ChromaDB**：
   - 开源的本地向量数据库
   - 支持持久化存储（`persist_directory`）
   - 支持元数据过滤（`filter={"project_id": "xxx"}`）

3. **知识库隔离**：
   - 每个文档块的 metadata 中添加 `project_id`
   - 检索时通过元数据过滤实现多租户隔离

---

#### 4.2.3 检索模块 (`retriever.py`)

**职责**：根据用户问题进行语义检索。

**核心类**：`VectorRetriever`

```python
class VectorRetriever:
    def query(self, question: str, project_id: str, top_k=3):
        """语义检索，返回最相关的K个文档"""
        results = self.vector_db.similarity_search_with_score(
            question,
            k=top_k,
            filter={"project_id": project_id}
        )
        return results  # [(Document, score), ...]
```

**关键技术点**：

1. **余弦相似度**：
   - 计算问题向量与文档向量的夹角余弦值
   - 值越大表示越相似

2. **Top-K 检索**：
   - 返回相似度最高的 K 个结果
   - 默认 K=3

3. **元数据过滤**：
   - 检索前先过滤出指定知识库的文档
   - 实现数据隔离

---

#### 4.2.4 生成模块 (`generator.py`)

**职责**：基于检索结果生成最终回答。

**核心类**：`RAGGenerator`

```python
class RAGGenerator:
    def __init__(self):
        self.retriever = VectorRetriever()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.prompt_template = ChatPromptTemplate.from_template(...)
    
    def get_answer(self, question: str, session_id, project_id) -> str:
        """生成回答"""
```

**关键技术点**：

1. **Prompt 模板**：
   ```python
   prompt_template = """
   你是一个知识库问答助手。
   
   【检索到的上下文】:
   {context}
   
   【用户问题】:
   {question}
   
   【回答规则】:
   1. 必须基于上下文回答
   2. 禁止说"未找到"
   ...
   """
   ```

2. **LCEL（LangChain Expression Language）**：
   ```python
   rag_chain = prompt_template | llm | StrOutputParser()
   answer = rag_chain.invoke({"context": context, "question": question})
   ```
   - `|` 操作符实现链式调用
   - 类似于 Unix 管道

3. **上下文格式化**：
   ```python
   def _format_docs(self, docs):
       """将文档列表格式化为字符串"""
       formatted = []
       for i, doc in enumerate(docs):
           source = doc.metadata.get('source', '未知来源')
           formatted.append(f"【文档{i+1} 来源: {source}】\n{doc.page_content}")
       return "\n\n".join(formatted)
   ```

---

### 4.3 Agent模块 (src/agent/)

#### 4.3.1 状态定义 (`state.py`)

**职责**：定义智能体之间传递的数据结构。

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # 消息列表（累加）
    next_step: str                                        # 下一步指示
```

**关键技术点**：

1. **TypedDict**：
   - Python 类型提示，定义字典的键值类型
   - 提高代码可读性和 IDE 支持

2. **Annotated + operator.add**：
   - `Annotated[List[BaseMessage], operator.add]` 表示当有新消息时，追加到列表而非覆盖
   - 这是 LangGraph 的状态聚合机制

3. **BaseMessage**：
   - LangChain 的消息基类
   - 子类包括：`HumanMessage`（用户消息）、`AIMessage`（AI消息）、`ToolMessage`（工具返回）

---

#### 4.3.2 工具定义 (`tools.py`)

**职责**：定义智能体可调用的工具函数。

**核心工具列表**：

| 工具名 | 功能 | 使用场景 |
|-------|------|---------|
| `ask_knowledge_base` | 知识库语义检索 | 用户有具体问题需要查询文档 |
| `list_knowledge_base_files` | 列出知识库文件 | 用户问"有哪些文件" |
| `search_by_filename` | 按文件名搜索 | 用户提到具体文件名或类型 |
| `general_qa` | 通用问答 | 编程问题、概念解释等 |
| `summarize_text` | 文本总结 | 用户要求总结内容 |
| `translate_text` | 翻译 | 用户要求翻译 |
| `analyze_code` | 代码分析 | 用户提交代码请求分析 |
| `get_current_time` | 获取时间 | 用户问时间 |
| `calculate_expression` | 数学计算 | 数学表达式求值 |

**关键技术点**：

1. **@tool 装饰器**：
   ```python
   from langchain_core.tools import tool
   
   @tool
   def ask_knowledge_base(query: str, config: RunnableConfig) -> str:
       """
       知识库语义搜索工具。
       参数:
           query: 用户问题
       """
       # 函数实现
   ```
   - 将普通 Python 函数转换为 LangChain Tool
   - 函数的 docstring 会作为工具的描述，帮助 LLM 理解何时使用

2. **RunnableConfig**：
   - LangChain 的配置传递机制
   - 通过 `config["configurable"]` 获取运行时参数
   - 如 `session_id`、`project_id`

3. **工具与状态机集成**：
   - 工具通过 `bind_tools()` 绑定到 LLM
   - LLM 根据用户输入决定是否调用工具
   - ToolNode 自动执行工具并返回结果

---

#### 4.3.3 节点实现 (`nodes.py`)

**职责**：实现智能体的具体行为逻辑。

**节点1：研究员（Researcher）**

```python
def researcher_node(state: AgentState) -> AgentState:
    """研究员节点：分析需求，调用工具获取信息"""
    
    # 1. 获取历史消息
    messages = state["messages"]
    
    # 2. 设置人设（System Prompt）
    system_prompt = SystemMessage(content="""
    你是一个全能智能助手，具备多种技能...
    
    【智能路由策略】：
    - 涉及文档 → 使用知识库工具
    - 编程问题 → 使用 general_qa
    - 文本处理 → 使用对应工具
    ...
    """)
    
    # 3. 调用模型（带工具绑定）
    response = llm_with_tools.invoke([system_prompt] + messages)
    
    # 4. 返回结果（追加到消息列表）
    return {"messages": [response]}
```

**节点2：作家（Writer）**

```python
def writer_node(state: AgentState):
    """作家节点：基于检索结果，撰写最终回答"""
    
    # 1. 将历史消息转换为字符串
    conversation_str = "\n".join([
        f"【{msg.type}】: {msg.content}" 
        for msg in state["messages"]
    ])
    
    # 2. 调用 LLM 生成回答
    prompt = ChatPromptTemplate.from_template("""
    你是资深技术作家。
    请阅读下面的【对话历史】，基于事实写一篇结构清晰的回答。
    
    【对话历史】:
    {history}
    """)
    
    chain = prompt | llm
    response = chain.invoke({"history": conversation_str})
    
    return {"messages": [response]}
```

**关键技术点**：

1. **SystemMessage vs HumanMessage**：
   - `SystemMessage`：设定 AI 的行为准则和人设
   - `HumanMessage`：用户的输入

2. **bind_tools**：
   ```python
   llm_with_tools = llm.bind_tools(all_tools)
   ```
   - 告诉 LLM 有哪些工具可用
   - LLM 可以在响应中返回 `tool_calls`（工具调用请求）

3. **Tool Calls 机制**：
   - 当 LLM 决定调用工具时，响应中包含 `tool_calls`
   - `tool_calls` 包含：工具名、参数
   - ToolNode 执行工具，返回 `ToolMessage`

---

#### 4.3.4 图编排 (`graph.py`)

**职责**：使用 LangGraph 编排智能体工作流。

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. 创建状态图
workflow = StateGraph(AgentState)

# 2. 添加节点
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("tools", ToolNode(all_tools))

# 3. 设置入口
workflow.set_entry_point("researcher")

# 4. 条件分支
def should_continue(state: AgentState):
    """判断下一步"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # 需要调用工具
    return "writer"     # 可以写回答了

workflow.add_conditional_edges(
    "researcher",
    should_continue,
    {"tools": "tools", "writer": "writer"}
)

# 5. 固定边
workflow.add_edge("tools", "researcher")  # 工具执行完回到研究员
workflow.add_edge("writer", END)          # 作家写完就结束

# 6. 编译
app = workflow.compile()
```

**流程图**：

```
          ┌─────────────────┐
          │   START         │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   Researcher    │◀──────────┐
          │   (研究员)      │           │
          └────────┬────────┘           │
                   │                    │
          ┌────────┴────────┐           │
          │ should_continue │           │
          │   (条件判断)     │           │
          └────────┬────────┘           │
                   │                    │
        ┌──────────┴──────────┐         │
        ▼                     ▼         │
  ┌───────────┐        ┌───────────┐    │
  │   tools   │        │   writer  │    │
  │  (工具)   │        │   (作家)  │    │
  └─────┬─────┘        └─────┬─────┘    │
        │                    │          │
        └────────────────────┘          │
                   │                    │
                   ▼                    │
              (回到研究员)──────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │      END        │
          └─────────────────┘
```

**关键技术点**：

1. **StateGraph**：
   - LangGraph 的核心类
   - 基于状态的图结构，节点间通过状态传递数据

2. **ToolNode**：
   - LangGraph 预置的工具执行节点
   - 自动识别 `tool_calls` 并执行对应工具

3. **条件边（Conditional Edges）**：
   - 根据运行时状态决定下一步
   - 实现动态流程控制

4. **编译（compile）**：
   - 将图定义转换为可执行的 `Runnable` 对象
   - 支持 `invoke()`、`stream()` 等方法

---

### 4.4 Service层 (src/service/)

Service层负责业务逻辑，与UI层解耦。

#### 4.4.1 知识库服务 (`kb_service.py`)

```python
class KnowledgeBaseService:
    def get_all_kbs(self) -> List[KnowledgeBase]
    def create_kb(self, kb_id: str, name: str) -> bool
    def delete_kb(self, kb_id: str) -> Tuple[bool, str]
    def get_kb_stats(self, kb_id: str) -> KnowledgeBaseStats
    def get_kb_files(self, kb_id: str) -> List[FileRecord]
    def get_or_create_session(self, kb_id: str, session_id) -> str
```

**功能**：
- 知识库的 CRUD 操作
- 会话管理
- 统计信息查询

#### 4.4.2 聊天服务 (`chat_service.py`)

```python
class ChatService:
    def get_history(self, session_id: str) -> List[ChatMessage]
    def save_user_message(self, session_id: str, content: str)
    def save_assistant_message(self, session_id: str, content: str)
    def stream_agent_response(self, prompt, session_id, project_id) -> Iterator
    def chat(self, prompt, session_id, project_id) -> Tuple[bool, str, List]
```

**功能**：
- 消息的存取
- Agent 的调用与流式响应
- 事件（Event）的抽象与传递

**事件类型**：
```python
class AgentNodeType(Enum):
    RESEARCHER = "researcher"  # 研究员节点
    TOOLS = "tools"            # 工具节点
    WRITER = "writer"          # 作家节点
```

#### 4.4.3 文档服务 (`document_service.py`)

```python
class DocumentService:
    def process_and_ingest(self, uploaded_files, project_id) -> IngestResult
    def get_supported_formats(self) -> List[str]
    def is_supported(self, filename: str) -> bool
    def filter_supported_files(self, files) -> Tuple[List, List]
```

**功能**：
- 文件上传处理
- 调用 ETL 和 VectorStore 完成入库
- 返回入库结果

**入库流程**：
```
上传文件 → ContentProcessor.load_uploaded_files()
         → ContentProcessor.split_documents()
         → VectorDBManager.create_vector_db()
         → add_project_file_record() (记录到SQLite)
```

---

### 4.5 工具层 (src/utils/)

#### 4.5.1 数据库模块 (`db.py`)

**职责**：SQLite 数据库操作，管理会话、消息、知识库元数据。

**数据表结构**：

```sql
-- 知识库表
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT
);

-- 会话表
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    name TEXT,
    project_id TEXT,
    created_at TEXT
);

-- 消息表
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,           -- 'user' 或 'assistant'
    content TEXT,
    created_at TEXT
);

-- 文件记录表
CREATE TABLE project_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT,
    source TEXT,         -- 文件名
    file_type TEXT,      -- 文件类型
    chunks_count INTEGER,
    created_at TEXT
);
```

**关键函数**：
```python
# 初始化数据库（自动建表）
init_db()

# 项目管理
create_project(project_id, name)
get_all_projects()
delete_project(project_id)

# 会话管理
create_session(session_id, name, project_id)
get_sessions_by_project(project_id)
get_latest_session_by_project(project_id)

# 消息管理
save_message(session_id, role, content)
get_messages(session_id)

# 文件记录
add_project_file_record(project_id, source, file_type, chunks_count)
list_project_files(project_id)

# 统计信息
get_project_stats(project_id)
```

#### 4.5.2 日志模块 (`logger.py`)

**职责**：统一的日志配置。

```python
def setup_logger(name: str) -> logging.Logger:
    """
    配置并返回一个 logger 对象
    
    特性：
    - 同时输出到控制台和文件
    - 日志轮转（5MB一个文件，保留5个备份）
    - UTF-8编码支持中文
    """
```

**日志格式**：
```
2024-01-15 10:30:45,123 - RAG_ETL - INFO - 📄 正在处理: test.pdf
```

#### 4.5.3 模型管理器 (`model_manager.py`)

**职责**：统一管理多个大语言模型和Embedding模型，支持动态切换。

**核心类**：`ModelManager`

```python
from src.utils.model_manager import model_manager

# 获取对话模型
llm = model_manager.get_chat_model()

# 获取指定模型
llm = model_manager.get_chat_model("deepseek-chat")

# 获取 Embedding 模型
embeddings = model_manager.get_embedding_model()

# 切换模型
model_manager.set_current_chat_model("gpt-4o")
model_manager.set_current_embedding_model("text-embedding-3-large")
```

**支持的模型提供商**：

| 提供商 | 对话模型 | Embedding模型 |
|--------|----------|---------------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | text-embedding-3-small, text-embedding-3-large |
| DeepSeek | deepseek-chat, deepseek-reasoner | - |
| 智谱AI | glm-4, glm-4-flash | embedding-2 |
| Moonshot | moonshot-v1-8k, moonshot-v1-32k | - |

**配置结构**：

```python
@dataclass
class ChatModelConfig:
    id: str                    # 模型ID（唯一标识）
    name: str                  # 显示名称
    model_name: str            # 实际调用的模型名
    provider: ModelProvider    # 提供商
    base_url: Optional[str]    # API基础URL
    api_key_env: str           # API Key环境变量名
    max_tokens: int            # 最大输出token
    temperature: float         # 默认温度
    description: str           # 模型描述
    supports_tools: bool       # 是否支持工具调用
    supports_vision: bool      # 是否支持视觉
```

**环境变量配置**：

```env
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1

# DeepSeek (国产替代)
DEEPSEEK_API_KEY=sk-xxxxxxxx

# 智谱AI
ZHIPU_API_KEY=xxxxxxxx

# Moonshot
MOONSHOT_API_KEY=sk-xxxxxxxx
```

**设计特点**：

1. **单例模式**：全局唯一的模型管理器实例
2. **延迟加载**：模型实例按需创建，减少资源占用
3. **缓存机制**：已创建的模型实例会被缓存，避免重复初始化
4. **API Key 检测**：切换模型前自动检测对应的 API Key 是否配置

**使用示例**：

```python
# 在代码中使用模型管理器
from src.utils.model_manager import model_manager

# 获取当前对话模型（用于Agent节点）
llm = model_manager.get_chat_model(temperature=0.1)

# 获取当前Embedding模型（用于向量化）
embeddings = model_manager.get_embedding_model()

# 查看模型状态
status = model_manager.get_model_status()
# {
#     "current_chat_model": {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai"},
#     "current_embedding_model": {"id": "text-embedding-3-small", "name": "OpenAI Embedding v3 Small", "dimension": 1536},
#     "available_chat_models": 10,
#     "available_embedding_models": 4
# }

# 检查模型是否可用
available, reason = model_manager.check_model_available("deepseek-chat")
# (True, "可用") 或 (False, "缺少API Key: DEEPSEEK_API_KEY")
```

---

## 5. 数据流与执行流程

### 5.1 完整问答流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户提问："Python是什么？"                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ web_app.py: chat_input 接收用户输入                                          │
│   ↓                                                                         │
│ ChatService.save_user_message() → 保存到 SQLite                             │
│   ↓                                                                         │
│ ChatService.stream_agent_response() → 调用 Agent Graph                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LangGraph: app.stream(inputs, config)                                       │
│   ↓                                                                         │
│ 进入 researcher 节点                                                         │
│   ↓                                                                         │
│ Researcher 分析：需要查知识库 → 返回 tool_calls: [ask_knowledge_base]        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 进入 tools 节点 (ToolNode)                                                   │
│   ↓                                                                         │
│ 执行 ask_knowledge_base("Python是什么")                                      │
│   ↓                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐     │
│ │ RAGGenerator.get_answer()                                            │     │
│ │   ↓                                                                  │     │
│ │ VectorRetriever.query() → ChromaDB.similarity_search_with_score()    │     │
│ │   ↓                                                                  │     │
│ │ 返回: [(Document("Python是..."), 0.85), ...]                         │     │
│ │   ↓                                                                  │     │
│ │ _format_docs() → 格式化上下文                                         │     │
│ │   ↓                                                                  │     │
│ │ prompt | llm | parser → 生成初步答案                                  │     │
│ └─────────────────────────────────────────────────────────────────────┘     │
│   ↓                                                                         │
│ 返回 ToolMessage                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 回到 researcher 节点                                                         │
│   ↓                                                                         │
│ Researcher 查看工具返回结果 → 判断信息足够 → 不再调用工具                      │
│   ↓                                                                         │
│ should_continue() 返回 "writer"                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 进入 writer 节点                                                             │
│   ↓                                                                         │
│ Writer 读取完整对话历史（包含工具返回的内容）                                   │
│   ↓                                                                         │
│ 基于事实撰写最终回答                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LangGraph: 流程结束，返回最终响应                                             │
│   ↓                                                                         │
│ ChatService 接收响应                                                         │
│   ↓                                                                         │
│ save_assistant_message() → 保存到 SQLite                                     │
│   ↓                                                                         │
│ web_app.py: st.markdown() 显示给用户                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 文档入库流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        用户上传文件: report.pdf                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ web_app.py: st.file_uploader() 接收文件                                      │
│   ↓                                                                         │
│ DocumentService.process_and_ingest()                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ContentProcessor.load_uploaded_files()                                       │
│   ↓                                                                         │
│ 1. 创建临时文件 (tempfile.NamedTemporaryFile)                                │
│ 2. 根据后缀选择 Loader (PyPDFLoader)                                         │
│ 3. loader.load() → 提取文本                                                  │
│ 4. 修复 metadata.source = 原始文件名                                         │
│ 5. 清理临时文件                                                              │
│   ↓                                                                         │
│ 返回: [Document(page_content="...", metadata={source: "report.pdf", ...})]  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ContentProcessor.split_documents()                                           │
│   ↓                                                                         │
│ 1. 根据文件类型选择分隔符策略                                                 │
│ 2. RecursiveCharacterTextSplitter                                           │
│    - chunk_size=800, chunk_overlap=100                                      │
│ 3. 递归切分，保持语义完整                                                     │
│   ↓                                                                         │
│ 返回: [Document(chunk1), Document(chunk2), ...]                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ VectorDBManager.create_vector_db()                                           │
│   ↓                                                                         │
│ 1. 为每个 chunk 添加 metadata.project_id                                     │
│ 2. Chroma.add_documents()                                                   │
│    - OpenAIEmbeddings 将文本转为向量                                         │
│    - 存储到 data/vector_db/                                                  │
│   ↓                                                                         │
│ 返回: Chroma 对象                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ add_project_file_record() → SQLite                                          │
│   ↓                                                                         │
│ 记录文件入库信息（用于UI展示）                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 核心技术知识点

### 6.1 RAG（检索增强生成）

**概念**：结合检索和生成的技术，通过外部知识库增强 LLM 的回答能力。

**核心流程**：
1. **索引阶段**：文档 → 切分 → 向量化 → 存入向量数据库
2. **检索阶段**：问题 → 向量化 → 相似度搜索 → 返回相关文档
3. **生成阶段**：Prompt = 上下文 + 问题 → LLM → 答案

**优势**：
- 解决 LLM 知识截止问题
- 减少幻觉（Hallucination）
- 支持私有/企业知识库

### 6.2 向量嵌入（Embedding）

**概念**：将文本转换为高维向量表示。

**特点**：
- 语义相近的文本在向量空间中距离较近
- 支持模糊/语义搜索，而非关键词匹配

**常用模型**：
- OpenAI: `text-embedding-3-small`, `text-embedding-3-large`
- 开源: `bge-large-zh`, `m3e-base`

### 6.3 LangChain 核心概念

| 概念 | 说明 |
|-----|------|
| **Message** | 消息对象（HumanMessage, AIMessage, SystemMessage, ToolMessage） |
| **Prompt Template** | 提示词模板，支持变量替换 |
| **LCEL** | LangChain Expression Language，链式调用语法 |
| **Tool** | 可被 LLM 调用的函数 |
| **Chain** | 多个组件的组合 |

### 6.4 LangGraph 核心概念

| 概念 | 说明 |
|-----|------|
| **StateGraph** | 基于状态的图结构 |
| **Node** | 图中的节点，处理状态并返回更新 |
| **Edge** | 节点之间的连接，定义流程方向 |
| **Conditional Edge** | 条件边，根据状态动态选择下一节点 |
| **State** | 在节点间传递的数据结构 |

### 6.5 多智能体协作模式

**本项目采用的模式**：流水线协作

```
Researcher (研究员)          Writer (作家)
     │                           │
     │ 1. 分析用户意图            │
     │ 2. 决定是否需要工具        │
     │ 3. 调用工具获取信息        │
     │                           │
     └─────── 传递信息 ──────────▶│
                                 │ 4. 整理信息
                                 │ 5. 生成最终回答
```

**设计原则**：
- 单一职责：每个 Agent 只做一件事
- 角色扮演：通过 System Prompt 定义 Agent 的"人设"
- 迭代协作：Researcher 可以多次调用工具，直到信息足够

### 6.6 知识库隔离技术

**实现方式**：元数据过滤（Metadata Filtering）

```python
# 入库时添加 project_id
for chunk in chunks:
    chunk.metadata["project_id"] = project_id

# 检索时过滤
results = vector_db.similarity_search(
    query,
    filter={"project_id": project_id}
)
```

**优势**：
- 单一向量数据库支持多租户
- 数据隔离，互不干扰
- 节省资源

---

## 7. 依赖与环境配置

### 7.1 核心依赖

```bash
# LangChain 生态
langchain                  # 基础框架
langchain-openai          # OpenAI 集成
langchain-community       # 社区组件（Loader等）
langchain-text-splitters  # 文本切分器
langchain-chroma          # ChromaDB 集成

# LangGraph
langgraph                 # 多智能体编排

# 向量数据库
chromadb                  # 本地向量数据库

# Web框架
streamlit                 # Web UI

# 工具库
python-dotenv             # 环境变量
```

### 7.2 安装命令

```bash
pip install langchain langchain-openai langchain-community langgraph chromadb python-dotenv streamlit
```

### 7.3 环境变量配置

创建 `.env` 文件：

```env
# OpenAI API 配置
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1  # 或代理地址
```

---

## 8. 使用指南

### 8.1 启动 Web 应用

```bash
streamlit run web_app.py
```

### 8.2 命令行交互

```bash
python main.py
```

### 8.3 知识入库脚本

```bash
python scripts/ingest_knowledge.py
```

### 8.4 使用流程

1. **创建知识库**：在知识库管理页面创建新的知识库（或使用默认的）
2. **上传文档**：在聊天页面的侧边栏上传文档
3. **开始对话**：在聊天框输入问题，系统会自动从知识库检索并生成回答

---

## 9. 扩展与二次开发

### 9.1 添加新的文档类型支持

在 `src/rag/etl.py` 的 `_select_loader` 方法中添加：

```python
elif suffix == ".xlsx":
    return YourExcelLoader(file_path)
```

### 9.2 添加新的工具

在 `src/agent/tools.py` 中添加：

```python
@tool
def my_new_tool(param: str, config: RunnableConfig) -> str:
    """
    工具描述（LLM会读取这个描述来决定是否调用）
    """
    # 实现逻辑
    return result
```

然后在 `all_tools` 列表中添加该工具。

### 9.3 添加新的 Agent 节点

1. 在 `src/agent/nodes.py` 中实现节点函数
2. 在 `src/agent/graph.py` 中注册节点
3. 定义节点间的流转逻辑

### 9.4 切换 Embedding 模型

在 `config/settings.py` 中修改：

```python
EMBEDDING_MODEL = "text-embedding-3-large"  # 或其他模型
```

### 9.5 切换 LLM 模型

```python
CHAT_MODEL = "gpt-4"  # 或其他模型
```

---

## 附录：常见问题

### Q1: 为什么检索不到内容？

**可能原因**：
1. 文档未正确入库 → 检查 `data/vector_db/` 目录
2. `project_id` 不匹配 → 确认当前选中的知识库
3. 向量数据库损坏 → 删除 `data/vector_db/` 重新入库

### Q2: 如何提高检索质量？

**优化方向**：
1. 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
2. 使用更好的 Embedding 模型
3. 增加重排序（Rerank）步骤
4. 优化切分策略（如按段落、按章节）

### Q3: 如何支持更多文件类型？

**扩展方法**：
1. 安装对应的 Loader 依赖
2. 在 `etl.py` 的 `_select_loader` 中添加分支

### Q4: 多智能体为什么会循环调用？

**排查方法**：
1. 检查 `should_continue` 的逻辑
2. 确认 Researcher 的 System Prompt 是否正确引导
3. 添加最大迭代次数限制

---

**文档版本**：1.0  
**最后更新**：2024年