# 测试指南 — RAG 多智能体问答系统

> 从环境搭建到全流程验证，分层级测试方案。

## 测试层级总览

```
Level 0  环境检查        无需 API Key    <30s
Level 1  数据库初始化    无需 API Key    <10s
Level 2  模型连接        需要 API Key    <30s
Level 3  组件单元测试    Mock API        <60s   ← pytest
Level 4  Agent 工作流    Mock LLM        <30s   ← pytest
Level 5  E2E 集成测试    需要 API Key    ~2min
Level 6  Docker 集成     需要 Docker     ~3min
```

---

## Level 0: 环境检查

> 不需要 API Key，验证依赖和配置完整。

```bash
# 1. Python 版本（需要 3.10+）
python --version

# 2. 核心依赖完整性
python -c "import langchain, langgraph, chromadb, streamlit; print('核心依赖 OK')"

# 3. .env 配置检查
python -c "
from config.settings import settings
assert settings.OPENAI_API_KEY, '请在 .env 中配置 OPENAI_API_KEY'
print(f'API Key: ...{settings.OPENAI_API_KEY[-8:]}')
print(f'Base URL: {settings.OPENAI_BASE_URL}')
print(f'Chat Model: {settings.CHAT_MODEL}')
print('环境配置 OK')
"

# 4. 全模块导入检查
bash scripts/test_runner.sh quick
```

**预期结果**：20/20 PASS，0 FAIL

---

## Level 1: 数据库初始化

> 不需要 API Key，验证 SQLite/PostgreSQL 正常工作。

```bash
python -c "
from src.utils.db import init_db, get_all_projects
init_db()
print('数据库初始化 OK')

projects = get_all_projects()
print(f'知识库列表: {projects}')
assert any(p[0] == 'default' for p in projects), '默认知识库不存在'
print('默认知识库 OK')
"
```

---

## Level 2: 模型连接测试

> 需要 API Key，验证 LLM 可达。

```bash
python -c "
from src.utils.model_manager import model_manager
llm = model_manager.get_chat_model(temperature=0)
resp = llm.invoke('请回答：1+1=?，只输出数字')
print(f'LLM 响应: {resp.content}')
print('模型连接 OK')
"
```

**常见问题**：
- `AuthenticationError` → API Key 无效或过期
- `ConnectionError` → Base URL 不对或网络问题
- `RateLimitError` → 请求太频繁

---

## Level 3: 组件单元测试

> Mock 外部 API，不需要真实 Key。

```bash
python -m pytest tests/test_components.py -v
```

测试覆盖：
- **数据库 CRUD**: init_db, create_project, save_message, get_messages
- **安全数学解析器**: _safe_math_eval 各种表达式
- **检索评估节点**: retrieval_evaluator_node（sufficient/insufficient/irrelevant）
- **答案验证节点**: answer_verifier_node（pass/fail）
- **缓存**: EmbeddingCache memory 模式读写
- **向量存储工厂**: get_vector_store() 返回正确类型

---

## Level 4: Agent 工作流测试

> Mock LLM 调用，验证图的节点路由逻辑。

```bash
python -m pytest tests/test_agent.py -v
```

测试覆盖：
- **简单查询** → researcher → writer → answer_verifier → END
- **检索评估 insufficient** → 回到 researcher 重试
- **检索评估 sufficient** → 进入 writer
- **答案验证 fail** → 重写 writer
- **答案验证 pass** → END
- **超过 max_retrieval_attempts** → 强制进入 writer

---

## Level 5: E2E 集成测试

> 需要 API Key + 文档已入库，完整用户流程。

### 5.1 文档入库

```bash
python scripts/ingest_knoledge.py
# 或通过 Streamlit UI 上传文件
```

### 5.2 Streamlit 启动与健康检查

```bash
streamlit run web_app.py &
sleep 5
curl -f http://localhost:8501/_stcore/health && echo "Web OK"
kill %1
```

### 5.3 CLI 模式测试

```bash
python main.py
# 输入：这个项目是什么？
# 预期：得到基于文档的回答
```

### 5.4 完整对话流程

在 Streamlit 中测试以下场景：
1. **新建知识库** → 创建项目 → 上传文件 → 等待入库完成
2. **知识库问答** → 切换到该项目 → 提问 → 验证回答引用了文档内容
3. **通用问答** → 问"Python如何读文件" → 不应调用知识库工具
4. **多轮对话** → 连续追问 → 验证上下文保持
5. **模型切换** → 侧边栏切换模型 → 验证新模型生效

---

## Level 6: Docker 集成测试

> 需要 Docker 环境。

```bash
# 1. 构建并启动
docker-compose up -d --build

# 2. 等待服务就绪
sleep 20

# 3. 健康检查
curl -f http://localhost:8501/_stcore/health && echo "RAG Agent OK"
curl -f http://localhost:6333/healthz && echo "Qdrant OK"
docker exec rag-postgres pg_isready -U rag && echo "PostgreSQL OK"
docker exec rag-redis redis-cli ping && echo "Redis OK"

# 4. 功能测试（同 Level 5 的 Streamlit 场景）

# 5. 清理
docker-compose down
```

---

## 快速命令速查

```bash
# 快速验证（Level 0-1，不需要 Key）
bash scripts/test_runner.sh quick

# 语法检查
bash scripts/test_runner.sh unit

# 单元测试（Level 3-4）
python -m pytest tests/test_components.py tests/test_agent.py -v

# 集成测试（Level 0-4）
bash scripts/test_runner.sh integration

# 完整测试套件（Level 0-5，需要 Key）
bash scripts/test_runner.sh full
```
