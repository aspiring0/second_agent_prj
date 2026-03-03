# Enterprise RAG Agent

企业级 RAG 多智能体问答系统——把你的文档变成智能知识库。

## 项目简介

这个系统做了几件事：

- **文档入库**：把 PDF、Word、Markdown、代码文件等丢进去，自动切分、向量化、存入数据库
- **语义检索**：用户提问时，系统从知识库里挖出相关内容
- **智能问答**：基于检索结果，大模型生成有理有据的回答
- **多知识库隔离**：不同项目的数据互不干扰

技术亮点是采用了**多智能体协作**架构：研究员负责查资料，作家负责写答案，分工明确。

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Web 界面                                          │
├─────────────────────────────────────────────────────────────┤
│  Service 层：ChatService / KnowledgeBaseService              │
├─────────────────────────────────────────────────────────────┤
│  Agent 层：Researcher（研究员）→ Writer（作家）               │
├─────────────────────────────────────────────────────────────┤
│  RAG 层：ETL → VectorStore → Retriever → Generator          │
├─────────────────────────────────────────────────────────────┤
│  存储层：ChromaDB（向量）+ SQLite（会话/消息）                │
└─────────────────────────────────────────────────────────────┘
```

**多智能体工作流**：

```
用户提问 → Researcher 分析意图
              ↓
         需要查资料？──是──→ 调用知识库工具 → 获取检索结果
              ↓否                                    ↓
         Writer 生成回答 ←───────────────────────────┘
```

## 目录结构

```
Enterprise_RAG_Agent/
├── config/settings.py        # 配置管理（API Key、模型参数等）
├── data/
│   ├── raw/                  # 原始文档存放处
│   └── vector_db/            # 向量数据库（自动生成）
├── logs/                     # 运行日志
├── src/
│   ├── rag/                  # RAG 核心
│   │   ├── etl.py            # 文档加载与切分
│   │   ├── vectorstore.py    # 向量库管理
│   │   ├── retriever.py      # 语义检索
│   │   └── generator.py      # 答案生成
│   ├── agent/                # 多智能体
│   │   ├── state.py          # 状态定义
│   │   ├── tools.py          # 工具函数
│   │   ├── nodes.py          # 节点实现
│   │   └── graph.py          # 状态机编排
│   ├── service/              # 业务服务
│   └── utils/                # 工具函数
├── web_app.py                # Web 入口
├── main.py                   # 命令行入口
└── scripts/ingest_knowledge.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-openai langchain-community langgraph chromadb python-dotenv streamlit
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1
```

### 3. 启动应用

**Web 界面**：
```bash
streamlit run web_app.py
```

**命令行**：
```bash
python main.py
```

## 使用流程

1. 打开 Web 界面，左侧创建或选择知识库
2. 上传文档（支持 PDF、Word、Markdown、TXT、代码文件等）
3. 等待入库完成
4. 开始对话，系统会自动从知识库检索相关信息并生成回答

## 核心模块说明

### RAG 流程

**ETL（文档处理）**

文档进来后，先识别格式选择对应的加载器，然后按语义切分成 800 字左右的片段，相邻片段保留 100 字重叠防止语义断裂。

**向量检索**

问题和文档都转成向量（OpenAI `text-embedding-3-small`），通过余弦相似度找出最相关的内容。检索时按 `project_id` 过滤，实现多知识库隔离。

**答案生成**

把检索到的文档和用户问题拼成 Prompt，喂给 GPT 生成回答。Prompt 模板会强调"必须基于上下文回答"，减少幻觉。

### 多智能体协作

**研究员（Researcher）**

- 性格设定：严谨、只认数据
- 职责：分析用户意图，决定是否需要查知识库
- 装备：知识库检索、文件搜索、通用问答等多种工具

**作家（Writer）**

- 性格设定：文采好、善于总结
- 职责：基于研究员提供的资料，撰写最终回答
- 无需工具，纯靠大模型能力

### 知识库隔离

通过 ChromaDB 的元数据过滤实现。每个文档块入库时携带 `project_id`，检索时只查当前知识库的数据。

## 可用工具

| 工具 | 功能 |
|-----|------|
| `ask_knowledge_base` | 知识库语义检索 |
| `list_knowledge_base_files` | 列出知识库文件 |
| `search_by_filename` | 按文件名搜索 |
| `general_qa` | 通用问答 |
| `summarize_text` | 文本总结 |
| `translate_text` | 翻译 |
| `analyze_code` | 代码分析 |
| `get_current_time` | 获取时间 |
| `calculate_expression` | 数学计算 |

## 配置参数

在 `config/settings.py` 中可调整：

```python
CHUNK_SIZE = 800        # 切片大小
CHUNK_OVERLAP = 100     # 切片重叠
EMBEDDING_MODEL = "text-embedding-3-small"  # 向量化模型
CHAT_MODEL = "gpt-3.5-turbo"               # 对话模型
```

## 常见问题

**Q: 检索不到内容？**

检查文档是否正确入库（`data/vector_db/` 目录是否有数据），确认 `project_id` 匹配。

**Q: 如何提高检索质量？**

- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
- 换更好的 Embedding 模型
- 优化文档切分策略

**Q: 如何支持更多文件类型？**

在 `src/rag/etl.py` 的 `_select_loader` 方法中添加对应的 Loader。

## 技术栈

- **LangChain**：LLM 应用框架
- **LangGraph**：多智能体状态机编排
- **ChromaDB**：本地向量数据库
- **Streamlit**：Web 界面
- **SQLite**：会话和消息存储
- **OpenAI API**：GPT 模型 + Embedding

## 扩展开发

**添加新工具**：在 `src/agent/tools.py` 中用 `@tool` 装饰器定义函数，加入 `all_tools` 列表。

**添加新节点**：在 `src/agent/nodes.py` 实现节点函数，在 `graph.py` 中注册并定义流转逻辑。

**切换模型**：修改 `config/settings.py` 中的 `CHAT_MODEL` 或 `EMBEDDING_MODEL`。

---

详细技术文档见 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)