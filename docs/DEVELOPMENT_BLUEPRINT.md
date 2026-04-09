# RAG 2026 统一开发蓝图

> 整合演进路线图 + 智能体重编排 + Docker 基础设施 + Skill 系统，经架构审查修正后的执行方案

---

## 零、架构审查修正（必读）

> 以下问题由代码审查发现，必须在任何新功能开发之前修复

### 必修 Bug（Week 1）

| # | 问题 | 位置 | 修复方案 |
|---|------|------|---------|
| B1 | `llm_with_tools` 模块级实例化，模型切换后不生效 | [nodes.py:58](src/agent/nodes.py#L58) | 移入函数体内按需创建 |
| B2 | ChromaDB 导入不一致：`langchain_community.vectorstores` vs `langchain_chroma` | [vectorstore.py](src/rag/vectorstore.py), [retriever.py](src/rag/retriever.py), [tools.py](src/agent/tools.py) | 统一使用 `langchain_chroma` |
| B3 | `calculate_expression` 使用 `eval()` 存在安全漏洞 | [tools.py:179](src/agent/tools.py#L179) | 替换为 `ast.literal_eval` 或安全数学解析器 |
| B4 | `requirements.txt` 缺少 `langchain-chroma`、`langchain-text-splitters` | requirements.txt | 补充并锁定版本 |
| B5 | 默认模型 `gpt-3.5-turbo` 已过时 | [settings.py:47](config/settings.py#L47) | 更新为 `gpt-4o-mini` |

### 审查结论：方案优先级修正

| 原方案 | 修正后 | 原因 |
|--------|--------|------|
| Phase 1 先迁移 ChromaDB → Milvus | **暂不迁移**，先建抽象层 | 数据无便携格式，迁移成本高，当前未触达 ChromaDB 瓶颈 |
| 3→7 节点一步到位 | **逐步增加**，先加 Query Rewriter | 每个新 LLM 节点增加 1-3s 延迟，简单问题会变慢 |
| 完整 Skill YAML 插件系统 | **先用简单工具目录**，有外部贡献者再升级 | 9 个工具不需要 500 行插件基础设施 |
| SQLite→PostgreSQL 优先 | **延后**，除非有并发需求 | 当前 chat_history.db 仅 48KB |

---

## 一、项目现状

### 1.1 当前架构

```
Streamlit UI → Service 层 → LangGraph Agent (Researcher + Writer) → RAG 管线 → ChromaDB + SQLite
```

### 1.2 能力清单

| 模块 | 状态 | 成熟度 |
|------|------|--------|
| 文档加载 | PDF/Word/MD/TXT/代码 6+ 格式 | ★★★★ |
| 文档切分 | RecursiveCharacterTextSplitter，语言感知 | ★★★ |
| 向量存储 | ChromaDB 本地单实例 | ★★★ |
| 语义检索 | 余弦相似度 Top-K，元数据过滤 | ★★★ |
| 多智能体 | Researcher + Writer 双角色 | ★★★★ |
| 多模型 | OpenAI/DeepSeek/智谱/Moonshot | ★★★★ |
| 相关性检查 | 可配置阈值 + 拒答机制 | ★★★ |

### 1.3 核心短板

| 短板 | 影响 | 优先级 |
|------|------|--------|
| 无 Reranker | 检索精度天花板低 | P0 |
| 纯向量检索，无混合检索 | 关键词精确匹配弱 | P0 |
| 无查询改写 | 原始 query 直接检索，召回率低 | P0 |
| Researcher 职责过重 | 意图分析+工具选择+判断全靠一个节点 | P1 |
| 无检索质量评估 | 拿到什么用什么 | P1 |
| 无答案验证 | Writer 输出可能有幻觉 | P1 |
| ChromaDB 单机 | 扩展性受限 | P2 |
| SQLite 单文件 | 并发受限 | P2 |

---

## 二、2026 RAG 技术全景

```
┌─────────────────────────────────────────────────────────┐
│                  2026 RAG 技术方向                        │
├─────────────────────────────────────────────────────────┤
│  Agentic RAG    GraphRAG      Multimodal RAG            │
│  自主推理+检索   知识图谱增强   图/表/音/视频             │
│                                                          │
│  Hybrid Search  Reranker      Adaptive Chunking         │
│  混合检索       重排序         语义感知动态切分           │
│                                                          │
│  Self-RAG       CRAG          Long-Context RAG          │
│  自反思检索     纠正性检索     长上下文融合               │
└─────────────────────────────────────────────────────────┘
```

---

## 三、修正后的演进路线（4 个 Sprint）

### Sprint 0：Bug 修复 + 基础加固（1 周）

> **Worktree 分支：`fix/sprint0-bugs`**
> **目标：修复所有已知 Bug，为新功能开发扫清障碍**

| 任务 | 文件 | 产出 |
|------|------|------|
| B1: 修复 LLM 模块级实例化 | `src/agent/nodes.py` | `get_llm_with_tools()` 函数内创建 |
| B2: 统一 ChromaDB 导入 | `src/rag/vectorstore.py`, `src/agent/tools.py` | 全部使用 `langchain_chroma` |
| B3: 替换 eval() | `src/agent/tools.py` | 安全数学解析器 |
| B4: 修复 requirements.txt | `requirements.txt` | 锁定版本、补充缺失依赖 |
| B5: 更新默认模型 | `config/settings.py` | `gpt-4o-mini` |

---

### Sprint 1：检索质量飞跃（2 周）

> **Worktree 分支：`feat/sprint1-retrieval`**
> **目标：混合检索 + Reranker + 查询改写 — 单一最高价值改动**

#### 1.1 混合检索（Hybrid Search）

```
用户查询
    ├──→ 向量检索（语义相似）  → Top-10
    ├──→ BM25 检索（关键词匹配） → Top-10
    └──→ RRF (Reciprocal Rank Fusion) 融合 → Top-5
```

新建 `src/rag/hybrid_retriever.py`：

```python
class HybridRetriever:
    def __init__(self, vector_store, documents, project_id: str):
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=10)

    def retrieve(self, query: str, top_k=5) -> List[Document]:
        v_docs = self.vector_retriever.invoke(query)
        b_docs = self.bm25_retriever.invoke(query)
        return self._rrf_fuse(v_docs, b_docs, top_k)
```

修改：`src/rag/retriever.py`（混合模式开关）、`config/settings.py`（`RETRIEVAL_MODE`）

#### 1.2 Reranker 重排序

```
初始检索 Top-20 → Reranker 精排 → Top-3 → Generator
```

新建 `src/rag/reranker.py`（先 Cohere API 零部署，后可切 BGE 本地模型）

#### 1.3 Query Rewriter（唯一的 LangGraph 新节点）

新建 `src/agent/nodes_query.py`：

```python
def query_rewriter_node(state: AgentState) -> AgentState:
    """查询改写节点 — HyDE / Multi-Query"""
```

修改 `src/agent/graph.py`：

```
START → researcher → [tools] → researcher → query_rewriter → researcher → writer → END
                                                         ↑            ↓
                                                    检索评估(规则) ──┘
```

**不增加独立 LLM 节点**，query_rewriter 内嵌在 researcher 流程中，通过 `retrieval_mode` 状态字段控制。

#### 涉及文件

| 操作 | 文件 |
|------|------|
| 新建 | `src/rag/hybrid_retriever.py`, `src/rag/reranker.py`, `src/agent/nodes_query.py` |
| 修改 | `src/rag/retriever.py`, `src/rag/generator.py`, `src/agent/tools.py`, `src/agent/graph.py`, `config/settings.py` |

---

### Sprint 2：Docker 化 + 存储抽象（2 周）

> **Worktree 分支：`feat/sprint2-docker`**
> **目标：数据库解耦为 Docker 容器，应用轻量化**

#### 2.1 Docker Compose（轻量版）

```yaml
# docker-compose.yml
services:
  rag-agent:
    build: .
    ports: ["8501:8501"]
    environment:
      - VECTOR_STORE_BACKEND=${VECTOR_STORE_BACKEND:-qdrant}
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    depends_on: [qdrant, postgres, redis]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]

  postgres:
    image: postgres:16-alpine
    ports: ["5432:5432"]
    environment:
      POSTGRES_DB: rag_agent
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag_secret

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

#### 2.2 向量存储抽象

```
src/rag/stores/
├── __init__.py          # VectorStoreBase 抽象基类 + 工厂函数
├── chroma_store.py      # 现有 ChromaDB（保持兼容）
└── qdrant_store.py      # Qdrant 实现（新增）
```

**关键原则**：先在 ChromaDB 上验证抽象层，测试通过后再加 Qdrant 后端。不要同时改两个东西。

#### 2.3 SQLite → PostgreSQL

修改 `src/utils/db.py`：`sqlite3` → `psycopg2` + 连接池

#### 2.4 Redis 缓存

替代内存 LRU，实现跨重启的 Embedding 缓存。

#### 涉及文件

| 操作 | 文件 |
|------|------|
| 新建 | `docker-compose.yml`, `src/rag/stores/__init__.py`, `src/rag/stores/chroma_store.py`, `src/rag/stores/qdrant_store.py`, `scripts/init_db.sql` |
| 修改 | `src/rag/vectorstore.py`, `src/rag/retriever.py`, `src/utils/db.py`, `config/settings.py` |

---

### Sprint 3：智能体增强 + 工具目录（2 周）

> **Worktree 分支：`feat/sprint3-agents`**
> **目标：扩展 Agent 状态、增加评估节点、工具目录化**

#### 3.1 AgentState 扩展

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # 查询分析
    query_type: str
    rewritten_queries: List[str]
    # 检索状态
    retrieval_quality: str
    retrieval_attempts: int
    # 生成状态
    confidence_score: float
```

#### 3.2 新增评估节点（逐步添加）

```python
# src/agent/nodes_eval.py
def retrieval_evaluator_node(state):  # 评估检索质量
def answer_verifier_node(state):      # 验证答案忠实度
```

**Graph 最终形态**（Sprint 3 完成后）：

```
START → researcher → [tools] → retrieval_evaluator
              ↑                       ↓
              └── insufficient ←───┘  sufficient
                                      ↓
                              writer → answer_verifier → END
```

#### 3.3 工具目录化（简化版 Skill）

```
src/agent/tools/
├── __init__.py           # 自动扫描、汇总 get_tools()
├── knowledge_base.py     # ask_knowledge_base, search_by_filename, list_files
├── general.py            # general_qa, calculate, time
├── text_processing.py    # summarize, translate, analyze_code
└── hybrid_search.py      # 新增：混合检索工具
```

每个文件导出 `get_tools() -> List[BaseTool]`，`__init__.py` 自动扫描加载。无需 YAML 元数据。

#### 涉及文件

| 操作 | 文件 |
|------|------|
| 新建 | `src/agent/nodes_eval.py`, `src/agent/tools/` 目录 |
| 修改 | `src/agent/state.py`, `src/agent/graph.py`, `src/agent/nodes.py` |
| 重构 | `src/agent/tools.py` → 拆分为 `src/agent/tools/` 目录 |

---

## 四、架构演进总览

```
v1.0（当前）
  │  双智能体 + ChromaDB + SQLite + 纯向量检索
  │
  ▼ Sprint 0
  │  修复 Bug、统一依赖
  │
  ▼ Sprint 1
  │  + 混合检索 (BM25 + Vector + RRF)
  │  + Reranker 重排序
  │  + 查询改写 (HyDE/Multi-Query)
  │  = 检索质量提升 15-25%
  │
  ▼ Sprint 2
  │  + Docker 化 (Qdrant + PostgreSQL + Redis)
  │  + 向量存储抽象层
  │  + 连接池 + Redis 缓存
  │
  ▼ Sprint 3
  │  + AgentState 扩展
  │  + 检索评估器 + 答案验证器
  │  + 工具目录化（简化版 Skill）
     + 5 节点工作流
```

---

## 五、评估指标

### 检索质量

| 指标 | 当前基线 | Sprint 1 目标 | Sprint 3 目标 |
|------|---------|-------------|-------------|
| Precision@3 | ~0.70 | 0.82 | 0.90 |
| Recall@10 | ~0.65 | 0.78 | 0.88 |
| MRR | ~0.72 | 0.85 | 0.92 |

### 生成质量

| 指标 | 当前 | Sprint 3 目标 |
|------|------|-------------|
| Faithfulness | ~0.75 | 0.88 |
| Hallucination Rate | ~15% | 8% |

### 性能

| 指标 | 当前 | Sprint 2 目标 |
|------|------|-------------|
| 端到端 P50 | ~3s | ~2.5s |
| 检索延迟 | ~500ms | ~300ms |

---

## 六、并行开发地图

### 依赖关系与合并顺序

```
时间 ──────────────────────────────────────────────────────────────────→

Phase A（并行窗口）           Phase B（串行）              Phase C（串行）

┌──────────────────┐
│ Sprint 0 (bugs)  │
│ fix/sprint0-bugs │──────┐
└──────────────────┘      │
                          ├── merge 0+1 → main
┌──────────────────┐      │         │
│ Sprint 1 (retriev)│──────┘         │
│ feat/sprint1-     │                ▼
│ retrieval         │      ┌──────────────────┐
└──────────────────┘      │ Sprint 2 (docker) │
                          │ feat/sprint2-docker│── merge → main
                          └──────────────────┘         │
                                                       ▼
                                               ┌──────────────────┐
                                               │ Sprint 3 (agents)│
                                               │ feat/sprint3-    │── merge → main
                                               │ agents           │
                                               └──────────────────┘
```

**为什么 Sprint 0+1 可以并行**：它们修改的文件几乎不重叠。
- Sprint 0 改：`nodes.py`（B1）、`tools.py`（B3）、`vectorstore.py`/`retriever.py`（B2）、`requirements.txt`（B4）、`settings.py`（B5）
- Sprint 1 改：新建文件（`hybrid_retriever.py`、`reranker.py`、`nodes_query.py`）、`generator.py`、`graph.py`
- **交集**：`tools.py`、`retriever.py`、`settings.py`、`requirements.txt` — 但改动不冲突（Sprint 0 修 bug，Sprint 1 加新功能）

### 文件变更矩阵

| 文件 | Sprint 0 | Sprint 1 | Sprint 2 | Sprint 3 |
|------|:--------:|:--------:|:--------:|:--------:|
| `src/agent/nodes.py` | 修B1 | - | - | 重构 |
| `src/agent/tools.py` | 修B2,B3 | 加R1工具 | - | 拆目录 |
| `src/agent/state.py` | - | 加字段 | - | 扩展 |
| `src/agent/graph.py` | - | 改流程 | - | 重构 |
| `src/rag/vectorstore.py` | 修B2 | - | 重构 | - |
| `src/rag/retriever.py` | 修B2 | 改检索 | 重构 | - |
| `src/rag/generator.py` | - | 加rerank | - | - |
| `config/settings.py` | 改B5 | 加配置 | 加配置 | - |
| `requirements.txt` | 修B4 | 加依赖 | 加依赖 | - |
| `src/utils/db.py` | - | - | 重写PG | - |
| `docker-compose.yml` | - | - | 重写 | - |

**冲突预防规则**：
- `settings.py` 和 `requirements.txt`：只追加，不删除其他 Sprint 的内容
- `tools.py`：Sprint 0 修 B2/B3 时不动函数签名；Sprint 1 只在文件末尾加新函数
- `retriever.py`：Sprint 0 只改 import；Sprint 1 加 `mode` 参数，不改原有逻辑

---

## 七、合并与集成流程

### 合并前检查清单（每个 Sprint 必须通过）

```bash
# 1. 导入检查
bash scripts/test_runner.sh quick

# 2. 语法检查（针对修改过的文件）
python -m py_compile src/agent/nodes.py
python -m py_compile src/agent/graph.py

# 3. 确认无未提交修改
git status

# 4. 确认 commit 格式
git log --oneline -5
```

### 具体合并步骤

#### 合并 Sprint 0（第一个合并）

```bash
# 在 worktree 中提交
cd .claude/worktrees/sprint0-bugs
git add src/agent/nodes.py src/agent/tools.py src/rag/vectorstore.py \
        src/rag/retriever.py requirements.txt config/settings.py
git commit -m "fix: Sprint 0 - 修复 B1-B5 (LLM实例化/ChromaDB导入/eval安全/依赖/默认模型)"

# 回到主仓库合并
cd d:/Code/Agent/second_genimi_agent
git checkout main
git merge worktree-sprint0-bugs --no-ff -m "merge: Sprint 0 - Bug 修复"

# 合并后验证
bash scripts/test_runner.sh quick
```

#### 合并 Sprint 1（需先 rebase Sprint 0 的更改）

```bash
# 在 worktree 中提交
cd .claude/worktrees/sprint1-retrieval
git add src/rag/hybrid_retriever.py src/rag/reranker.py src/agent/nodes_query.py \
        src/rag/retriever.py src/rag/generator.py src/agent/graph.py \
        src/agent/tools.py config/settings.py requirements.txt
git commit -m "feat: Sprint 1 - 混合检索 + Reranker + 查询改写"

# Rebase 到最新的 main（已有 Sprint 0 的更改）
git rebase main
# 如有冲突（tools.py, retriever.py, settings.py），解决后：
# git add <冲突文件> && git rebase --continue

# 回到主仓库合并
cd d:/Code/Agent/second_genimi_agent
git checkout main
git merge worktree-sprint1-retrieval --no-ff -m "merge: Sprint 1 - 检索增强"

# 合并后验证
bash scripts/test_runner.sh quick
python -c "from src.rag.hybrid_retriever import HybridRetriever; print('Sprint 1 OK')"
```

#### 合并 Sprint 2/3（同上模式）

```bash
# Sprint 2
cd .claude/worktrees/sprint2-docker
git rebase main  # 已有 Sprint 0+1 的更改
git add -A && git commit -m "feat: Sprint 2 - Docker化 + 存储抽象 + PG + Redis"
cd d:/Code/Agent/second_genimi_agent
git merge worktree-sprint2-docker --no-ff -m "merge: Sprint 2 - Docker 化"
bash scripts/test_runner.sh quick

# Sprint 3
cd .claude/worktrees/sprint3-agents
git rebase main  # 已有 Sprint 0+1+2 的更改
git add -A && git commit -m "feat: Sprint 3 - Agent增强 + 评估节点 + 工具目录化"
cd d:/Code/Agent/second_genimi_agent
git merge worktree-sprint3-agents --no-ff -m "merge: Sprint 3 - 智能体增强"
bash scripts/test_runner.sh quick
```

### 全部合并后的集成测试

```bash
# 完整验证
bash scripts/test_runner.sh quick     # 导入检查
bash scripts/test_runner.sh unit      # 单元测试

# 端到端验证（需要 API Key）
streamlit run web_app.py &
sleep 5
curl -f http://localhost:8501/_stcore/health
kill %1

# Docker 验证
docker-compose up -d
sleep 15
curl -f http://localhost:8501/_stcore/health
docker-compose down
```

---

## 八、技术选型

| 组件 | Sprint 1 | Sprint 2 | 远期 |
|------|----------|----------|------|
| 向量存储 | ChromaDB（不变） | +Qdrant | Milvus（大规模） |
| 关系存储 | SQLite（不变） | PostgreSQL | PostgreSQL |
| 缓存 | 内存 LRU | Redis | Redis |
| Reranker | Cohere API | +BGE 本地 | BGE |
| Embedding | OpenAI API | +bge-m3 本地 | 本地为主 |
| 容器 | 单 Docker | docker-compose | K8s |
