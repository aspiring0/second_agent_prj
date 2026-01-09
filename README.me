##1
第一阶段：基础设施搭建与技术选型 (环境配置、API 管理)
pip install langchain langchain-openai langchain-community langgraph chromadb python-dotenv
langchain: 基础框架。

langchain-openai: 调用 OpenAI 接口（或兼容接口）。

langgraph: 用于构建多智能体的状态机（State Graph）。

chromadb: 本地向量数据库。

python-dotenv: 读取 .env 文件中的密钥。

##2
第二阶段：RAG 核心构建 - 知识入库 (ETL) (文档加载、切片、向量化)
###2.1
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

###2.2ETL PipelineExtract 提取 -> Transform 转换 -> Load 加载）
Extract (提取): 计算机不能直接读懂文件夹里的文件，需要专门的 Loader 去读取文本。

Transform (转换/切分): 大模型（如 GPT-3.5/4）是有“记忆长度限制”的（Context Window）。如果我们直接把一本 100 页的书扔进去，它会报错或者产生幻觉。所以，我们必须把长文章切成 300-500 字的小块，并且为了保证语义连贯，块与块之间要留一点“重叠区域”（Overlap）。



第三阶段：RAG 核心构建 - 检索与评估 (向量库查询、重排序、检索效果测试)

第四阶段：单智能体 (Single Agent) 原型 (让一个 Agent 能调用 RAG 工具)

第五阶段：多智能体 (Multi-Agent) 架构设计 (定义角色：研究员、写作者、审核员等)

第六阶段：智能体协作编排 (Orchestration) (基于 LangGraph/AutoGen 实现状态流转)

第七阶段：服务化与接口封装 (FastAPI 封装与前端交互)