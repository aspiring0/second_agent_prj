# scripts/run_rag_test.py
"""
执行RAG系统完整测试
1. 将PROJECT_DOCUMENTATION.md入库
2. 运行25个标准测试用例
3. 输出测试报告
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import shutil
from pathlib import Path

from config.settings import settings
from src.rag.etl import ContentProcessor
from src.rag.vectorstore import VectorDBManager
from src.rag.retriever import VectorRetriever
from src.rag.generator import RAGGenerator
from src.utils.db import init_db, create_project, add_project_file_record
from src.utils.logger import setup_logger

logger = setup_logger("RAG_TEST_RUNNER")


def setup_test_knowledge_base(project_id: str = "rag_test_kb"):
    """设置测试知识库，将PROJECT_DOCUMENTATION.md入库"""
    print("\n" + "="*70)
    print("📦 步骤1: 设置测试知识库")
    print("="*70)
    
    # 初始化数据库
    init_db()
    
    # 创建测试项目
    create_project(project_id, "RAG测试知识库")
    print(f"✅ 创建测试项目: {project_id}")
    
    # 准备文档
    doc_path = settings.BASE_DIR / "PROJECT_DOCUMENTATION.md"
    
    if not doc_path.exists():
        print(f"❌ 文档不存在: {doc_path}")
        return False
    
    print(f"📄 使用文档: {doc_path}")
    
    # 处理文档
    processor = ContentProcessor()
    
    # 模拟上传文件
    class UploadedFile:
        def __init__(self, path):
            self.name = path.name
            self._path = path
        
        def getvalue(self):
            with open(self._path, 'rb') as f:
                return f.read()
    
    uploaded_file = UploadedFile(doc_path)
    
    # 加载文档
    print("📖 加载文档...")
    documents = processor.load_uploaded_files([uploaded_file])
    print(f"   加载了 {len(documents)} 个文档片段")
    
    # 切分文档
    print("✂️ 切分文档...")
    chunks = processor.split_documents(documents)
    print(f"   切分成 {len(chunks)} 个块")
    
    # 添加project_id到metadata
    for chunk in chunks:
        chunk.metadata["project_id"] = project_id
    
    # 入库到向量数据库
    print("💾 入库到向量数据库...")
    vector_db_manager = VectorDBManager()
    vector_db = vector_db_manager.create_vector_db(chunks, project_id)
    
    # 记录文件
    add_project_file_record(project_id, "PROJECT_DOCUMENTATION.md", "md", len(chunks))
    
    print(f"✅ 知识库设置完成: {len(chunks)} 个文档块已入库")
    
    return True


def run_rag_tests(project_id: str = "rag_test_kb"):
    """运行RAG测试"""
    print("\n" + "="*70)
    print("🧪 步骤2: 运行RAG测试")
    print("="*70)
    
    from tests.rag_test_suite import RAGTestSuite, PROJECT_DOC_TEST_CASES
    
    # 初始化组件
    retriever = VectorRetriever()
    suite = RAGTestSuite()
    
    # 定义检索函数
    def retriever_func(query, proj_id, top_k):
        try:
            results = retriever.query(query, proj_id, top_k)
            return results
        except Exception as e:
            print(f"   ❌ 检索失败: {e}")
            return []
    
    # 定义生成函数（使用简化版本，直接基于上下文生成）
    def generator_func(query, context, proj_id):
        if not context:
            return "抱歉，未找到相关信息。"
        
        # 简单的基于上下文的回答（实际应调用LLM）
        # 这里我们用上下文的相关部分作为回答
        context_parts = context[:500]
        return f"基于文档内容：{context_parts}..."
    
    # 运行测试
    print(f"📝 共 {len(PROJECT_DOC_TEST_CASES)} 个测试用例")
    print("-"*70)
    
    results = suite.run_test(
        retriever_func=retriever_func,
        generator_func=generator_func,
        project_id=project_id,
        top_k=3
    )
    
    # 打印摘要
    suite.print_summary()
    
    # 导出报告
    report_path = suite.export_report()
    
    return suite


def run_retrieval_only_test(project_id: str = "rag_test_kb"):
    """仅运行检索测试（不需要LLM）"""
    print("\n" + "="*70)
    print("🔍 检索测试（无需LLM）")
    print("="*70)
    
    from tests.rag_test_suite import PROJECT_DOC_TEST_CASES
    
    retriever = VectorRetriever()
    
    results = []
    total_precision = 0
    total_recall = 0
    total_mrr = 0
    total_latency = 0
    
    print(f"\n📝 运行 {len(PROJECT_DOC_TEST_CASES)} 个测试查询...\n")
    
    for i, case in enumerate(PROJECT_DOC_TEST_CASES):
        query = case["query"]
        keywords = case.get("keywords", [])
        
        # 执行检索
        start_time = time.time()
        try:
            retrieved = retriever.query(query, project_id, top_k=3)
            latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            print(f"   [{case['id']}] ❌ 检索失败: {e}")
            continue
        
        # 计算指标
        found_keywords = set()
        for doc, score in retrieved:
            content_lower = doc.page_content.lower()
            for kw in keywords:
                if kw.lower() in content_lower:
                    found_keywords.add(kw)
        
        precision = len(found_keywords) / len(retrieved) if retrieved else 0
        recall = len(found_keywords) / len(keywords) if keywords else 0
        
        # 计算MRR
        mrr = 0.0
        for rank, (doc, score) in enumerate(retrieved):
            content_lower = doc.page_content.lower()
            for kw in keywords:
                if kw.lower() in content_lower:
                    mrr = 1.0 / (rank + 1)
                    break
            if mrr > 0:
                break
        
        total_precision += precision
        total_recall += recall
        total_mrr += mrr
        total_latency += latency_ms
        
        results.append({
            "id": case["id"],
            "query": query[:40],
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "latency_ms": latency_ms,
            "retrieved_count": len(retrieved)
        })
        
        # 打印进度
        status = "✅" if recall > 0.5 else "⚠️"
        print(f"   [{i+1:2d}/{len(PROJECT_DOC_TEST_CASES)}] {status} {case['id']}: "
              f"P={precision:.2f} R={recall:.2f} MRR={mrr:.2f} ({latency_ms:.0f}ms)")
    
    # 计算平均值
    n = len(results)
    avg_precision = total_precision / n if n > 0 else 0
    avg_recall = total_recall / n if n > 0 else 0
    avg_mrr = total_mrr / n if n > 0 else 0
    avg_latency = total_latency / n if n > 0 else 0
    
    # 打印汇总
    print("\n" + "="*70)
    print("📊 检索测试结果汇总")
    print("="*70)
    
    print(f"\n📈 检索质量指标:")
    print(f"   平均精确率 (Precision): {avg_precision:.3f}")
    print(f"   平均召回率 (Recall):    {avg_recall:.3f}")
    print(f"   平均MRR:               {avg_mrr:.3f}")
    
    print(f"\n⚡ 性能指标:")
    print(f"   平均检索延迟: {avg_latency:.2f} ms")
    
    # 按分类统计
    print(f"\n📂 按分类统计:")
    by_category = {}
    for r, case in zip(results, PROJECT_DOC_TEST_CASES[:len(results)]):
        cat = case.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"count": 0, "recall_sum": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["recall_sum"] += r["recall"]
    
    for cat, stats in by_category.items():
        avg_cat_recall = stats["recall_sum"] / stats["count"]
        print(f"   {cat}: {stats['count']}个用例, 平均召回率 {avg_cat_recall:.3f}")
    
    print("="*70)
    
    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_mrr": avg_mrr,
        "avg_latency_ms": avg_latency,
        "total_tests": n,
        "by_category": by_category
    }


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 RAG系统测试")
    print("="*70)
    
    project_id = "rag_test_kb"
    
    # 步骤1: 设置知识库
    if not setup_test_knowledge_base(project_id):
        print("❌ 知识库设置失败，退出测试")
        return
    
    # 步骤2: 运行检索测试（不需要LLM）
    retrieval_results = run_retrieval_only_test(project_id)
    
    # 保存结果
    import json
    from datetime import datetime
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "project_id": project_id,
        "test_type": "retrieval_only",
        "summary": retrieval_results
    }
    
    report_dir = settings.BASE_DIR / "metrics" / "rag_tests"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"rag_retrieval_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 测试报告已保存: {report_path}")
    
    return retrieval_results


if __name__ == "__main__":
    main()