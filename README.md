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
# RAG 参数
CHUNK_SIZE = 800        # 切片大小
CHUNK_OVERLAP = 100     # 切片重叠
RETRIEVAL_TOP_K = 3     # 检索返回文档数

# 默认模型配置（可在Web界面动态切换）
EMBEDDING_MODEL = "text-embedding-3-small"  # 向量化模型
CHAT_MODEL = "gpt-3.5-turbo"               # 对话模型
AGENT_MODEL = "gpt-4o-mini"                # Agent模型

# 模型切换功能开关
ENABLE_MODEL_SWITCHING = True  # 是否启用Web界面模型切换
```

### 环境变量配置

在 `.env` 文件中配置各平台 API Key：

```env
# OpenAI (必需)
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1

# DeepSeek (可选 - 国产替代，性价比高)
DEEPSEEK_API_KEY=sk-xxxxxxxx

# 智谱AI (可选 - GLM系列)
ZHIPU_API_KEY=xxxxxxxx

# Moonshot (可选 - 月之暗面)
MOONSHOT_API_KEY=sk-xxxxxxxx

# 模型配置（可选，覆盖默认值）
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

## 🤖 多模型支持

系统支持多个大语言模型提供商，可在 Web 界面动态切换：

### 支持的对话模型

| 提供商 | 模型 | 特点 |
|--------|------|------|
| **OpenAI** | gpt-4o | 最新旗舰，性能强大 |
| **OpenAI** | gpt-4o-mini | 轻量版，性价比高 |
| **OpenAI** | gpt-4-turbo | 支持128K上下文 |
| **OpenAI** | gpt-3.5-turbo | 快速经济 |
| **DeepSeek** | deepseek-chat | 中文能力强，性价比高 |
| **DeepSeek** | deepseek-reasoner | 推理能力强 |
| **智谱AI** | glm-4 | 中文理解强 |
| **智谱AI** | glm-4-flash | 快速，免费额度大 |
| **Moonshot** | moonshot-v1-8k | 长文本能力突出 |
| **Moonshot** | moonshot-v1-32k | 超长文本支持 |

### 支持的 Embedding 模型

| 提供商 | 模型 | 维度 | 特点 |
|--------|------|------|------|
| **OpenAI** | text-embedding-3-small | 1536 | 性价比高 |
| **OpenAI** | text-embedding-3-large | 3072 | 效果最好 |
| **OpenAI** | text-embedding-ada-002 | 1536 | 经典稳定 |
| **智谱AI** | embedding-2 | 1024 | 中文优化 |

### 代码中使用

```python
from src.utils.model_manager import model_manager

# 获取当前对话模型
llm = model_manager.get_chat_model()

# 指定模型获取
llm = model_manager.get_chat_model("deepseek-chat")

# 获取 Embedding 模型
embeddings = model_manager.get_embedding_model()

# 切换模型
model_manager.set_current_chat_model("gpt-4o")
model_manager.set_current_embedding_model("text-embedding-3-large")

# 查看模型状态
status = model_manager.get_model_status()
print(status)
```

### 注意事项

⚠️ **重要**：切换 Embedding 模型后，**需要重新入库文档**，因为不同模型的向量维度和语义空间不同。

## 🐳 Docker 部署

### 快速部署到服务器

**1. 准备服务器**
- 一台有公网 IP 的云服务器（阿里云、腾讯云等）
- 安装 Docker 和 Docker Compose
- 开放 8501 端口

**2. 上传项目到服务器**
```bash
# 方式一：使用 scp
scp -r ./* user@your-server:/home/user/rag-agent/

# 方式二：使用 git clone
git clone https://github.com/aspiring0/second_agent_prj.git
cd second_agent_prj
```

**3. 配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
nano .env
```

**4. 启动服务**
```bash
# 构建并启动（后台运行）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

**5. 访问应用**
```
http://你的服务器公网IP:8501
```

### Docker 常用命令

```bash
# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose restart

# 查看容器状态
docker-compose ps

# 进入容器调试
docker exec -it rag-agent bash

# 查看实时日志
docker-compose logs -f rag-agent
```

### 数据持久化

以下目录会挂载到宿主机，数据不会丢失：
- `./data` - 向量数据库和上传的文件
- `./logs` - 运行日志
- `./metrics` - 指标数据

### 更新部署

```bash
# 拉取最新代码
git pull

# 重新构建并启动
docker-compose up -d --build
```

---

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

## 📊 指标与测试系统

本项目包含完整的指标收集和测试系统，支持性能监控、质量评估、A/B测试和高并发压力测试。

### 快速运行测试

```bash
# 运行综合指标测试
python tests/run_all_metrics.py

# 运行负载测试（需要配置查询函数）
python -c "from tests import run_quick_test; run_quick_test()"
```

### 指标类型

| 类型 | 指标 | 描述 |
|------|------|------|
| **性能** | latency, RPS, P50/P95/P99 | 响应延迟、吞吐量、百分位延迟 |
| **质量** | precision, recall, MRR, NDCG | 检索精确率、召回率、排名指标 |
| **生成** | faithfulness, relevance, hallucination | 忠实度、相关性、幻觉检测 |
| **负载** | concurrent_users, success_rate | 并发用户数、成功率 |

### 在代码中使用

```python
from src.metrics import metrics_collector, performance_tracker

# 追踪操作延迟
with performance_tracker.track_latency("my_operation"):
    # 执行操作
    result = do_something()

# 记录自定义指标
metrics_collector.record(
    metric_type="custom",
    metric_name="my_metric",
    value=123.45,
    unit="ms"
)
```

### 测试报告

测试报告自动保存到 `metrics/reports/` 目录，包括：
- `comprehensive_report_YYYYMMDD_HHMMSS.json` - 详细测试报告
- `latest_report.json` - 最新测试报告

📄 **详细指标文档**：[docs/METRICS_GUIDE.md](docs/METRICS_GUIDE.md)

---

详细技术文档见 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
