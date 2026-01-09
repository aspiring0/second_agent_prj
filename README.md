## 1第一阶段：基础设施搭建与技术选型 (环境配置、API 管理)
- 安装依赖：
```bash
pip install langchain langchain-openai langchain-community langgraph chromadb python-dotenv
```
- langchain: 基础框架。
- langchain-openai: 调用 OpenAI 接口（或兼容接口）。
- langgraph: 用于构建多智能体的状态机（State Graph）。
- chromadb: 本地向量数据库。
- python-dotenv: 读取 .env 文件中的密钥。



## 2第二阶段：RAG 核心构建 - 知识入库 (ETL) (文档加载、切片、向量化)
### 2.1
```
Enterprise_RAG_Agent/
├── config/                 # [配置层] 全局配置管理
│   ├── __init__.py
│   └── settings.py         # 集中管理 API Key、路径、模型参数
├── data/                   # [数据层] 
│   ├── raw/                # 存放原始文档 (PDF, MD, TXT)
│   └── vector_db/          # ChromaDB 持久化存储目录
├── logs/                   # [日志层] 存放系统运行日志
├── src/                    # [核心代码层]
│   ├── __init__.py
│   ├── rag/                # RAG 模块
│   │   ├── __init__.py
│   │   ├── etl.py          # Extract, Transform (加载与切分)
│   │   └── vectorstore.py  # Load & Retrieve (向量库管理)
│   └── utils/              # 通用工具
│       ├── __init__.py
│       └── logger.py       # 统一日志配置
├── scripts/                # [脚本层] 运维/初始化脚本
│   └── ingest_knowledge.py # 知识入库入口脚本
├── .env                    # 环境变量
├── .gitignore
└── main.py                 # 主程序入口

config:存放配置文件
data/raw 存放txt/pdf文件
    /vector_db 放生成的向量数据库文件
logs：存放运行的日志
src:存放源代码
    /utils:通用工具
    /rag:RAG业务逻辑
scripts：存放运维脚本
```
### 2.2 ETL Pipeline
Extract 提取 -> Transform 转换 -> Load 加载）
Extract (提取): 计算机不能直接读懂文件夹里的文件，需要专门的 Loader 去读取文本。

Transform (转换/切分): 大模型（如 GPT-3.5/4）是有“记忆长度限制”的（Context Window）。如果我们直接把一本 100 页的书扔进去，它会报错或者产生幻觉。所以，我们必须把长文章切成 300-500 字的小块，并且为了保证语义连贯，块与块之间要留一点“重叠区域”（Overlap）。



## 第三阶段：RAG 核心构建 - 检索与评估 (向量库查询、重排序、检索效果测试)
Retrieval (检索): 用户提问时，系统会先把问题转成向量，然后在向量数据库中找到最相似的文本块，作为“上下文”提供给大模型。


## 第四阶段：单智能体 (Single Agent) 原型 (让一个 Agent 能调用 RAG 工具)
```
现在我们有了“记忆”（Retriever），还需要一个“大脑”（LLM）来组织语言。

这一步的目标： 不再是返回冷冰冰的“文档片段”，而是直接返回答案。 比如：

输入：“Python 是什么？”

RAG 之前的回答：{'content': 'Python is the best language...', 'source': 'test.txt'}

RAG 之后的回答：“根据内部文档，Python 被描述为最好的语言。它在企业级开发中...”

核心概念：Prompt Engineering (提示工程)
我们需要写一个模板，把“用户的问题”和“检索到的文档”拼在一起，喂给大模型。

公式：Prompt = 系统指令 + 检索到的上下文 (Context) + 用户问题
```
### 4.1 生成器 (Generator)
```
生成器的任务： 接收用户问题和检索到的上下文，根据模板生成最终的答案。

核心组件：
Prompt Template: 定义了问题和上下文的格式。
LLM (如 OpenAI GPT-3.5/4): 负责根据模板生成文本。
```

## 第五阶段：多智能体 (Multi-Agent) 架构设计 (定义角色：研究员、写作者、审核员等)
核心设计思想：流水线模式
现在我们把任务拆给两个专家：

🔎 研究员 (Researcher)：

性格：严谨、死板、只认数据。

装备：RAG 检索工具 (ask_knowledge_base)。

任务：负责搞清楚事实，把资料挖出来。

✍️ 作家 (Writer)：

性格：文采好、善于总结、懂用户心理。

装备：无（只靠大脑）。

任务：拿着研究员给的资料，写出一篇漂亮的报告。


## 第六阶段：智能体协作编排 (Orchestration) (基于 LangGraph/AutoGen 实现状态流转)


## 第七阶段：服务化与接口封装 (FastAPI 封装与前端交互)