# tests/test_rag_integration.py
"""
RAG集成测试 - 将多语言代码文件入库并测试检索效果
测试流程：
1. 加载 tests/test_files/ 下的所有代码文件
2. 使用ETL模块切分文档
3. 入库到向量数据库
4. 执行检索测试
5. 评估检索质量
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import json
import time

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.rag.etl import (
    ContentProcessor,
    PYTHON_SEPARATORS,
    JAVA_SEPARATORS,
    JS_SEPARATORS,
    GO_SEPARATORS,
    C_SEPARATORS,
    RUST_SEPARATORS
)
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("RAG_INTEGRATION_TEST")

# 测试文件目录
TEST_FILES_DIR = Path(__file__).parent / "test_files"

# 测试用的project_id
TEST_PROJECT_ID = "test_multilang_code"


@dataclass
class RetrievalTestResult:
    """检索测试结果"""
    question: str
    expected_file: str
    retrieved_files: List[str]
    top1_hit: bool  # 第一条是否命中
    top3_hit: bool  # 前三条是否命中
    scores: List[float]


# 文件类型到分隔符的映射
FILE_TYPE_SEPARATORS = {
    ".py": PYTHON_SEPARATORS,
    ".java": JAVA_SEPARATORS,
    ".c": C_SEPARATORS,
    ".js": JS_SEPARATORS,
    ".go": GO_SEPARATORS,
    ".rs": RUST_SEPARATORS,
}


class RAGIntegrationTester:
    """RAG集成测试器"""
    
    def __init__(self):
        self.processor = ContentProcessor()
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        self.vector_db = None
        self.test_questions = self._create_test_questions()
    
    def _create_test_questions(self) -> List[Dict]:
        """创建测试问题集 - 针对测试文件内容设计"""
        return [
            # Python相关
            {"question": "Python中VectorStore类是如何实现的？", "expected": "test.py"},
            {"question": "如何在Python中添加文档到向量存储？", "expected": "test.py"},
            {"question": "Python的RAGPipeline类有什么功能？", "expected": "test.py"},
            {"question": "Python的ETLProcessor如何切分文档？", "expected": "test.py"},
            
            # Java相关
            {"question": "Java中RAGService类的query方法怎么实现？", "expected": "Test.java"},
            {"question": "Java的VectorStore如何搜索文档？", "expected": "Test.java"},
            {"question": "Java中如何构建提示词？", "expected": "Test.java"},
            
            # C相关
            {"question": "C语言中如何创建向量存储？", "expected": "test.c"},
            {"question": "C语言中search_similar函数的实现", "expected": "test.c"},
            {"question": "C语言中如何生成文本向量？", "expected": "test.c"},
            
            # JavaScript相关
            {"question": "JavaScript中VectorStore类的构造函数", "expected": "test.js"},
            {"question": "JS中如何计算余弦相似度？", "expected": "test.js"},
            {"question": "JavaScript的RAGPipeline如何执行查询？", "expected": "test.js"},
            
            # Go相关
            {"question": "Go语言中VectorStore的AddDocuments方法", "expected": "test.go"},
            {"question": "Go语言如何实现RAG查询？", "expected": "test.go"},
            {"question": "Go语言的ETLProcessor如何切分文档？", "expected": "test.go"},
            
            # Rust相关
            {"question": "Rust中VectorStore如何搜索？", "expected": "test.rs"},
            {"question": "Rust的RAGPipeline如何构建提示词？", "expected": "test.rs"},
            {"question": "Rust中Document结构体的定义", "expected": "test.rs"},
        ]
    
    def run_full_test(self) -> Dict:
        """运行完整的RAG集成测试"""
        print("\n" + "="*70)
        print("🔄 RAG集成测试 (多语言代码)")
        print("="*70)
        
        results = {
            "ingestion": None,
            "retrieval": [],
            "summary": {}
        }
        
        # Step 1: 加载和切分文档
        print("\n📥 Step 1: 加载和切分文档")
        print("-"*70)
        chunks = self._load_and_split_files()
        
        if not chunks:
            print("❌ 没有加载到任何文档，测试终止")
            return results
        
        results["ingestion"] = {
            "total_chunks": len(chunks),
            "files_processed": len(set(c.metadata.get("source", "") for c in chunks))
        }
        
        # Step 2: 入库到向量数据库
        print("\n📥 Step 2: 入库到向量数据库")
        print("-"*70)
        self._ingest_to_vectordb(chunks)
        
        # Step 3: 执行检索测试
        print("\n🔍 Step 3: 执行检索测试")
        print("-"*70)
        retrieval_results = self._run_retrieval_tests()
        results["retrieval"] = retrieval_results
        
        # Step 4: 计算汇总指标
        print("\n📊 Step 4: 计算汇总指标")
        print("-"*70)
        summary = self._calculate_summary(retrieval_results)
        results["summary"] = summary
        
        # 打印最终结果
        self._print_final_summary(summary)
        
        return results
    
    def _load_and_split_files(self) -> List:
        """加载并切分所有测试文件"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        all_chunks = []
        
        if not TEST_FILES_DIR.exists():
            print(f"❌ 测试文件目录不存在: {TEST_FILES_DIR}")
            return []
        
        test_files = list(TEST_FILES_DIR.glob("*"))
        test_files = [f for f in test_files if f.is_file() and f.suffix in FILE_TYPE_SEPARATORS]
        
        for file_path in test_files:
            suffix = file_path.suffix
            separators = FILE_TYPE_SEPARATORS.get(suffix, [])
            
            print(f"   📄 {file_path.name} (分隔符: {len(separators)}个)")
            
            try:
                # 加载文档
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                
                if not docs:
                    continue
                
                # 切分
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    separators=separators
                )
                chunks = splitter.split_documents(docs)
                
                # 添加元数据
                for chunk in chunks:
                    chunk.metadata["source"] = file_path.name
                    chunk.metadata["file_type"] = suffix
                    chunk.metadata["project_id"] = TEST_PROJECT_ID
                
                all_chunks.extend(chunks)
                print(f"      → {len(chunks)} 个片段")
                
            except Exception as e:
                print(f"      ❌ 处理失败: {e}")
        
        print(f"\n   ✅ 总计: {len(all_chunks)} 个片段")
        return all_chunks
    
    def _ingest_to_vectordb(self, chunks: List):
        """入库到向量数据库"""
        if not chunks:
            return
        
        # 使用临时目录存储向量数据库
        persist_dir = Path("data/vector_db_test")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 创建向量数据库
            self.vector_db = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
                collection_name=TEST_PROJECT_ID
            )
            
            # 添加文档
            self.vector_db.add_documents(chunks)
            
            print(f"   ✅ 入库完成: {len(chunks)} 个片段")
            print(f"   📁 存储位置: {persist_dir}")
            
        except Exception as e:
            print(f"   ❌ 入库失败: {e}")
            raise
    
    def _run_retrieval_tests(self) -> List[RetrievalTestResult]:
        """执行检索测试"""
        results = []
        
        for i, test_case in enumerate(self.test_questions):
            question = test_case["question"]
            expected = test_case["expected"]
            
            try:
                # 执行检索
                retrieved = self.vector_db.similarity_search_with_score(
                    question,
                    k=3,
                    filter={"project_id": TEST_PROJECT_ID}
                )
                
                if not retrieved:
                    print(f"   ⚠️ Q{i+1}: 无结果")
                    continue
                
                # 提取结果
                retrieved_files = [doc.metadata.get("source", "") for doc, score in retrieved]
                scores = [float(score) for doc, score in retrieved]
                
                # 检查命中
                top1_hit = retrieved_files[0] == expected if retrieved_files else False
                top3_hit = expected in retrieved_files
                
                result = RetrievalTestResult(
                    question=question,
                    expected_file=expected,
                    retrieved_files=retrieved_files,
                    top1_hit=top1_hit,
                    top3_hit=top3_hit,
                    scores=scores
                )
                results.append(result)
                
                # 打印结果
                status = "✅" if top1_hit else ("🔶" if top3_hit else "❌")
                print(f"   {status} Q{i+1}: {question[:40]}...")
                print(f"      期望: {expected} | Top1: {retrieved_files[0] if retrieved_files else 'N/A'}")
                
            except Exception as e:
                print(f"   ❌ Q{i+1}: 检索失败 - {e}")
        
        return results
    
    def _calculate_summary(self, results: List[RetrievalTestResult]) -> Dict:
        """计算汇总指标"""
        if not results:
            return {}
        
        total = len(results)
        top1_hits = sum(1 for r in results if r.top1_hit)
        top3_hits = sum(1 for r in results if r.top3_hit)
        
        # 按文件类型统计
        by_file_type = {}
        for r in results:
            file_type = r.expected_file.split('.')[-1]
            if file_type not in by_file_type:
                by_file_type[file_type] = {"total": 0, "top1": 0, "top3": 0}
            by_file_type[file_type]["total"] += 1
            if r.top1_hit:
                by_file_type[file_type]["top1"] += 1
            if r.top3_hit:
                by_file_type[file_type]["top3"] += 1
        
        return {
            "total_questions": total,
            "top1_accuracy": top1_hits / total if total > 0 else 0,
            "top3_accuracy": top3_hits / total if total > 0 else 0,
            "top1_hits": top1_hits,
            "top3_hits": top3_hits,
            "by_file_type": by_file_type
        }
    
    def _print_final_summary(self, summary: Dict):
        """打印最终汇总"""
        print("\n" + "="*70)
        print("📊 RAG集成测试结果汇总")
        print("="*70)
        
        print(f"\n总体准确率:")
        print(f"   Top-1 准确率: {summary['top1_accuracy']*100:.1f}% ({summary['top1_hits']}/{summary['total_questions']})")
        print(f"   Top-3 准确率: {summary['top3_accuracy']*100:.1f}% ({summary['top3_hits']}/{summary['total_questions']})")
        
        print(f"\n按文件类型:")
        for file_type, stats in summary.get("by_file_type", {}).items():
            top1_rate = stats["top1"] / stats["total"] * 100 if stats["total"] > 0 else 0
            top3_rate = stats["top3"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"   .{file_type:<4} Top-1: {top1_rate:.0f}%  Top-3: {top3_rate:.0f}%  ({stats['total']}个问题)")


def main():
    """主函数"""
    print("="*70)
    print("⚠️  注意: 此测试需要有效的OpenAI API配置")
    print("="*70)
    
    # 检查API配置
    if not settings.OPENAI_API_KEY:
        print("❌ 未配置 OPENAI_API_KEY，无法运行集成测试")
        print("   请在 .env 文件中配置: OPENAI_API_KEY=your-key")
        return
    
    tester = RAGIntegrationTester()
    results = tester.run_full_test()
    
    # 保存报告
    report_path = Path("metrics/reports/rag_integration_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换结果为可序列化格式
    report = {
        "ingestion": results.get("ingestion", {}),
        "summary": results.get("summary", {}),
        "retrieval_details": [
            {
                "question": r.question,
                "expected": r.expected_file,
                "retrieved": r.retrieved_files,
                "top1_hit": r.top1_hit,
                "top3_hit": r.top3_hit,
                "scores": r.scores
            }
            for r in results.get("retrieval", [])
        ]
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 报告已保存: {report_path}")


if __name__ == "__main__":
    main()