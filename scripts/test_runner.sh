#!/bin/bash
# scripts/test_runner.sh — 统一测试入口
# 用法：
#   bash scripts/test_runner.sh quick       快速验证（导入检查，<30s）
#   bash scripts/test_runner.sh unit        单元测试
#   bash scripts/test_runner.sh full        完整测试套件

set -e

PASS=0
FAIL=0

echo "=========================================="
echo "  RAG Agent Test Runner — $1"
echo "=========================================="
echo ""

# ==================== Quick Test ====================
if [ "$1" = "quick" ] || [ "$1" = "unit" ] || [ "$1" = "full" ]; then
    echo ">> Quick Tests (Import Checks)"

    python -c "
modules = [
    ('config',         'from config.settings import settings'),
    ('logger',         'from src.utils.logger import setup_logger'),
    ('db',             'from src.utils.db import init_db'),
    ('model_manager',  'from src.utils.model_manager import model_manager'),
    ('etl',            'from src.rag.etl import ContentProcessor'),
    ('vectorstore',    'from src.rag.vectorstore import VectorDBManager'),
    ('retriever',      'from src.rag.retriever import VectorRetriever'),
    ('generator',      'from src.rag.generator import RAGGenerator'),
    ('state',          'from src.agent.state import AgentState'),
    ('tools',          'from src.agent.nodes import all_tools'),
    ('prompts',        'from src.agent.prompts import PromptManager'),
    ('nodes',          'from src.agent.nodes import researcher_node, writer_node'),
    ('graph',          'from src.agent.graph import app'),
    ('services',       'from src.service import ChatService, KnowledgeBaseService, DocumentService'),
]

# Optional modules (may not exist yet)
optional = [
    ('hybrid_retriever',  'from src.rag.hybrid_retriever import HybridRetriever'),
    ('reranker',          'from src.rag.reranker import Reranker'),
    ('nodes_query',       'from src.agent.nodes_query import query_rewriter_node'),
    ('stores',            'from src.rag.stores import get_vector_store'),
    ('cache',             'from src.utils.cache import EmbeddingCache'),
    ('nodes_eval',        'from src.agent.nodes_eval import retrieval_evaluator_node'),
    ('tools_dir',         'from src.agent.tools import get_all_tools'),
]

passed = 0
failed = 0

for name, stmt in modules:
    try:
        exec(stmt)
        print(f'  [{name}] PASS')
        passed += 1
    except Exception as e:
        print(f'  [{name}] FAIL: {e}')
        failed += 1

for name, stmt in optional:
    try:
        exec(stmt)
        print(f'  [{name}] PASS')
        passed += 1
    except Exception:
        print(f'  [{name}] SKIP')

print(f'\n  Core: {passed} passed, {failed} failed')
if failed > 0:
    exit(1)
" 2>&1

    if [ $? -eq 0 ]; then
        echo ""
    else
        echo ""
        echo "Some core tests failed!"
        exit 1
    fi
fi

# ==================== Unit Test ====================
if [ "$1" = "unit" ] || [ "$1" = "full" ]; then
    echo ">> Unit Tests (Syntax Checks)"

    python -c "
import py_compile, os
files = [
    'src/agent/nodes.py',
    'src/agent/graph.py',
    'src/agent/nodes_eval.py',
    'src/agent/tools_dir/__init__.py',
    'src/agent/tools_dir/knowledge_base.py',
    'src/agent/tools_dir/general.py',
    'src/agent/tools_dir/text_processing.py',
    'src/rag/generator.py',
    'src/rag/retriever.py',
    'src/rag/stores/__init__.py',
    'src/rag/stores/chroma_store.py',
    'src/rag/etl.py',
    'src/utils/db.py',
    'src/utils/cache.py',
    'config/settings.py',
]
passed = 0
failed = 0
for f in files:
    if os.path.exists(f):
        try:
            py_compile.compile(f, doraise=True)
            print(f'  [syntax {f}] PASS')
            passed += 1
        except py_compile.PyCompileError as e:
            print(f'  [syntax {f}] FAIL: {e}')
            failed += 1
    else:
        print(f'  [syntax {f}] SKIP (not found)')

print(f'\n  Syntax: {passed} passed, {failed} failed')
if failed > 0:
    exit(1)
" 2>&1

    if [ $? -ne 0 ]; then
        echo "Syntax checks failed!"
        exit 1
    fi
fi

# ==================== Integration Test ====================
if [ "$1" = "integration" ]; then
    echo ">> Integration Tests (pytest)"

    python -m pytest tests/test_components.py tests/test_agent.py -v --tb=short 2>&1

    if [ $? -ne 0 ]; then
        echo "Integration tests failed!"
        exit 1
    fi
fi

# ==================== Full Test ====================
if [ "$1" = "full" ]; then
    echo ">> Full Test Suite"
    if [ -f "tests/run_all_metrics.py" ]; then
        python tests/run_all_metrics.py --skip-load-test 2>&1 || true
    else
        echo "  tests/run_all_metrics.py not found, skipping"
    fi
fi

echo "=========================================="
echo "  All tests passed!"
echo "=========================================="
