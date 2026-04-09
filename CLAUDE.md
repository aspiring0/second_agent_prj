# CLAUDE.md — AI 进程入口文件

> 本文件是所有 AI 编程助手（Claude Code / Gemini CLI / Copilot）的项目入口。
> 每个新进程启动时必须先完整阅读本文件，理解开发工作流后再执行任何操作。

## 项目简介

企业级 RAG 多智能体问答系统。LangChain + LangGraph + ChromaDB + Streamlit。
详细技术文档：`docs/DEVELOPMENT_BLUEPRINT.md`、`PROJECT_DOCUMENTATION.md`

---

## 开发工作流

### 第一步：阅读蓝图

打开 `docs/DEVELOPMENT_BLUEPRINT.md`，理解：
- 当前处于哪个 Sprint
- 你的任务是哪个
- 哪些文件可以改、哪些不能碰

### 第二步：选择并进入 Worktree

本项目使用 git worktree 隔离各 Sprint 的开发：

```bash
# 查看所有 worktree
git worktree list

# 进入对应 Sprint 的 worktree
cd .claude/worktrees/sprint0-bugs        # Bug 修复
cd .claude/worktrees/sprint1-retrieval    # 检索增强
cd .claude/worktrees/sprint2-docker       # Docker 化
cd .claude/worktrees/sprint3-agents       # 智能体增强
```

进入 worktree 后，阅读该目录下的 `SPRINT_PROMPT.md`，它是你的完整任务说明。

### 第三步：开发

1. 按 `SPRINT_PROMPT.md` 中的任务清单逐项执行
2. 每完成一个 Task，立即运行对应测试验证
3. 遵守文件变更约束，不要修改其他 Sprint 负责的文件

### 第四步：测试

开发完成后，运行测试：

```bash
# 快速验证（必须通过，<10s）
bash scripts/test_runner.sh quick

# 如果涉及新功能，运行单元测试
bash scripts/test_runner.sh unit
```

### 第五步：提交与合并

参见下方「合并流程」章节。

---

## 并行开发规则

### 依赖关系

```
Sprint 0 (fix/sprint0-bugs)     ──────┐
                                      ├──→ Sprint 2 (docker) ──→ Sprint 3 (agents)
Sprint 1 (feat/sprint1-retrieval) ────┘
```

- **可并行**：Sprint 0 和 Sprint 1（改不同文件，互不影响）
- **串行依赖**：Sprint 2 需等 0+1 合并后开始；Sprint 3 需等 2 合并后开始

### 合并顺序（严格遵守）

1. Sprint 0 合并到 main
2. Sprint 1 rebase main，然后合并到 main
3. Sprint 2 rebase main，然后合并到 main
4. Sprint 3 rebase main，然后合并到 main

### 文件所有权（冲突预防）

| 文件 | Sprint 0 | Sprint 1 | Sprint 2 | Sprint 3 |
|------|----------|----------|----------|----------|
| `src/agent/nodes.py` | 修改 B1 | - | - | 重构 |
| `src/agent/tools.py` | 修改 B2,B3 | 修改 R1 | - | 拆分为目录 |
| `src/agent/state.py` | - | 新增字段 | - | 扩展 |
| `src/agent/graph.py` | - | 修改 | - | 重构 |
| `src/rag/vectorstore.py` | 修改 B2 | - | 重构 | - |
| `src/rag/retriever.py` | 修改 B2 | 修改 | 重构 | - |
| `src/rag/generator.py` | - | 修改 | - | - |
| `config/settings.py` | 修改 B5 | 新增配置 | 新增配置 | - |
| `requirements.txt` | 修改 B4 | 新增依赖 | 新增依赖 | - |
| `src/utils/db.py` | - | - | 重写 | - |
| `docker-compose.yml` | - | - | 重写 | - |

**规则**：
- 标记为「修改」的文件，只做最小改动，不要重构
- 标记为「重构/重写」的文件，可以大改
- 标记为「-」的文件，**绝对不要碰**
- `config/settings.py` 和 `requirements.txt` 多个 Sprint 都会改，只在文件末尾追加，不要删除其他 Sprint 的配置

---

## 测试要求

### 每个 Task 完成后

```bash
python -c "from src.agent.graph import app; print('OK')"
```

### 每个 Sprint 完成后

```bash
# 快速验证（必须通过）
bash scripts/test_runner.sh quick
```

### 合并前检查清单

- [ ] 所有 Task 已完成
- [ ] `scripts/test_runner.sh quick` 通过
- [ ] 无语法错误（`python -m py_compile src/agent/graph.py` 等）
- [ ] 无未提交的修改（`git status` 干净）
- [ ] commit message 符合格式（`fix: ...` 或 `feat: ...`）

---

## 合并流程

### 从 Worktree 合并到 Main

```bash
# 1. 在 worktree 中提交所有更改
cd .claude/worktrees/sprint0-bugs
git add -A
git commit -m "fix: Sprint 0 - 修复 B1-B5 全部 Bug"

# 2. 回到主仓库
cd d:/Code/Agent/second_genimi_agent

# 3. 确保 main 是最新的
git checkout main
git pull origin main  # 如果有远程的话

# 4. 合并（如果 Sprint 间无冲突）
git merge worktree-sprint0-bugs

# 5. 合并后验证
bash scripts/test_runner.sh quick

# 6. 如果验证通过，推送
git push origin main
```

### 如果有冲突

```bash
# 查看冲突文件
git diff --name-only --diff-filter=U

# 手动解决冲突后
git add <冲突文件>
git commit -m "merge: 解决 Sprint X 合并冲突"
```

### Rebase（当 main 已有其他 Sprint 的更改时）

```bash
# 在 worktree 中 rebase
cd .claude/worktrees/sprint1-retrieval
git rebase main

# 如果有冲突，解决后继续
git add <冲突文件>
git rebase --continue
```

---

## 代码规范

- 中文注释和中文日志
- 函数必须有 docstring
- 新配置项通过环境变量控制，有合理默认值
- 不删除现有功能，新功能通过开关启用
- commit message 格式：`fix: ...` / `feat: ...` / `refactor: ...`
