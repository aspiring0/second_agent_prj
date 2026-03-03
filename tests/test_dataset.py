# tests/test_dataset.py
"""
RAG系统测试数据集 - Test Dataset
包含训练集、验证集和测试集，用于评估系统性能
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class TestCase:
    """单个测试用例"""
    id: str
    query: str  # 用户查询
    expected_docs: List[str] = field(default_factory=list)  # 期望检索到的文档
    ground_truth: str = ""  # 期望的回答
    category: str = "general"  # 分类：general, technical, factual, reasoning
    difficulty: str = "medium"  # 难度：easy, medium, hard
    tags: List[str] = field(default_factory=list)


@dataclass
class TestDataset:
    """测试数据集"""
    name: str
    description: str
    version: str
    created_at: str
    test_cases: List[TestCase]
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "test_cases": [asdict(tc) for tc in self.test_cases],
            "statistics": {
                "total_cases": len(self.test_cases),
                "by_category": self._count_by("category"),
                "by_difficulty": self._count_by("difficulty")
            }
        }
    
    def _count_by(self, field_name: str) -> Dict[str, int]:
        counts = {}
        for tc in self.test_cases:
            val = getattr(tc, field_name, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TestDataset':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = [
            TestCase(**tc) for tc in data["test_cases"]
        ]
        
        return cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            created_at=data["created_at"],
            test_cases=test_cases
        )


# ==================== 预定义测试数据集 ====================

def get_default_training_set() -> TestDataset:
    """获取默认训练集"""
    test_cases = [
        # 简单事实类问题
        TestCase(
            id="train_001",
            query="Python是什么？",
            expected_docs=["python_intro.txt"],
            ground_truth="Python是一种高级编程语言，由Guido van Rossum于1991年创建。它以简洁、易读的语法著称，支持多种编程范式。",
            category="factual",
            difficulty="easy",
            tags=["python", "programming", "basics"]
        ),
        TestCase(
            id="train_002",
            query="什么是机器学习？",
            expected_docs=["ml_basics.txt", "ai_overview.txt"],
            ground_truth="机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进性能，而无需显式编程。",
            category="factual",
            difficulty="easy",
            tags=["machine_learning", "ai", "basics"]
        ),
        TestCase(
            id="train_003",
            query="向量数据库的作用是什么？",
            expected_docs=["vector_db.txt", "embeddings.txt"],
            ground_truth="向量数据库用于存储和检索高维向量，常用于语义搜索、推荐系统和RAG应用中，支持高效的相似度查询。",
            category="factual",
            difficulty="medium",
            tags=["database", "vector", "rag"]
        ),
        
        # 技术类问题
        TestCase(
            id="train_004",
            query="如何实现文本切分？",
            expected_docs=["text_splitting.txt", "chunking_strategies.txt"],
            ground_truth="文本切分可以通过固定大小切分、递归字符切分、语义切分等方式实现。常用的方法是按字符数或token数切分，并保留重叠部分以保持上下文连贯。",
            category="technical",
            difficulty="medium",
            tags=["nlp", "chunking", "preprocessing"]
        ),
        TestCase(
            id="train_005",
            query="RAG系统的工作流程是什么？",
            expected_docs=["rag_pipeline.txt", "retrieval_generation.txt"],
            ground_truth="RAG系统的工作流程包括：1)文档预处理和向量化；2)用户查询向量化；3)语义检索相关文档；4)将检索结果与查询组合；5)LLM生成最终回答。",
            category="technical",
            difficulty="medium",
            tags=["rag", "pipeline", "architecture"]
        ),
        
        # 推理类问题
        TestCase(
            id="train_006",
            query="为什么需要知识库隔离？",
            expected_docs=["kb_isolation.txt", "multi_tenant.txt"],
            ground_truth="知识库隔离可以防止不同项目的数据混淆，确保检索结果的相关性和准确性，同时保护数据隐私，支持多租户场景。",
            category="reasoning",
            difficulty="medium",
            tags=["architecture", "isolation", "multi-tenant"]
        ),
        
        # 复杂问题
        TestCase(
            id="train_007",
            query="如何评估RAG系统的性能？",
            expected_docs=["rag_evaluation.txt", "metrics.txt"],
            ground_truth="RAG系统评估包括：检索质量评估（精确率、召回率、MRR）、生成质量评估（忠实度、相关性、完整性）、性能指标（延迟、吞吐量）以及端到端测试。",
            category="reasoning",
            difficulty="hard",
            tags=["evaluation", "metrics", "performance"]
        ),
    ]
    
    return TestDataset(
        name="RAG训练集",
        description="用于RAG系统基础测试的训练数据集",
        version="1.0",
        created_at=datetime.now().isoformat(),
        test_cases=test_cases
    )


def get_default_validation_set() -> TestDataset:
    """获取默认验证集"""
    test_cases = [
        TestCase(
            id="val_001",
            query="LangChain框架有什么优势？",
            expected_docs=["langchain_intro.txt"],
            ground_truth="LangChain提供了丰富的组件和工具，支持多种LLM和向量数据库，简化了RAG应用开发，具有良好的扩展性和模块化设计。",
            category="factual",
            difficulty="easy",
            tags=["langchain", "framework"]
        ),
        TestCase(
            id="val_002",
            query="ChromaDB如何实现持久化存储？",
            expected_docs=["chromadb_storage.txt", "vector_persistence.txt"],
            ground_truth="ChromaDB通过PersistentClient实现持久化，将向量数据存储在指定目录中，支持增量更新和快速加载。",
            category="technical",
            difficulty="medium",
            tags=["chromadb", "persistence", "storage"]
        ),
        TestCase(
            id="val_003",
            query="Embedding模型如何选择？",
            expected_docs=["embedding_models.txt", "model_comparison.txt"],
            ground_truth="选择Embedding模型需要考虑：向量维度、语言支持、领域适配性、计算成本和性能基准。常用模型包括OpenAI text-embedding-3系列、Sentence Transformers等。",
            category="reasoning",
            difficulty="medium",
            tags=["embedding", "model_selection"]
        ),
        TestCase(
            id="val_004",
            query="多智能体协作的优缺点是什么？",
            expected_docs=["multi_agent.txt", "agent_collaboration.txt"],
            ground_truth="多智能体协作优点：专业化分工、并行处理、可扩展性强；缺点：复杂度增加、通信开销、协调困难。",
            category="reasoning",
            difficulty="hard",
            tags=["multi_agent", "collaboration", "tradeoffs"]
        ),
        TestCase(
            id="val_005",
            query="如何处理检索结果不相关的情况？",
            expected_docs=["retrieval_improvement.txt", "fallback_strategies.txt"],
            ground_truth="可以通过以下方式处理：1)降低检索阈值；2)使用查询重写；3)混合检索策略；4)引入LLM判断是否需要重新检索；5)提供兜底回答。",
            category="technical",
            difficulty="hard",
            tags=["retrieval", "edge_cases", "strategies"]
        ),
    ]
    
    return TestDataset(
        name="RAG验证集",
        description="用于验证RAG系统性能的测试数据集",
        version="1.0",
        created_at=datetime.now().isoformat(),
        test_cases=test_cases
    )


def get_default_test_set() -> TestDataset:
    """获取默认测试集"""
    test_cases = [
        TestCase(
            id="test_001",
            query="解释一下RAG中的检索增强机制",
            expected_docs=["rag_mechanism.txt", "retrieval_augmentation.txt"],
            ground_truth="检索增强机制通过将外部知识库中的相关信息注入到LLM的上下文中，增强模型的回答能力和准确性，减少幻觉。",
            category="technical",
            difficulty="medium",
            tags=["rag", "mechanism", "augmentation"]
        ),
        TestCase(
            id="test_002",
            query="什么是幻觉检测？",
            expected_docs=["hallucination.txt", "fact_checking.txt"],
            ground_truth="幻觉检测是识别LLM生成内容中与事实不符或无法验证的部分的技术，可通过NLI模型、事实核查或与源文档比对来实现。",
            category="factual",
            difficulty="medium",
            tags=["hallucination", "detection", "quality"]
        ),
        TestCase(
            id="test_003",
            query="对比不同的文本切分策略",
            expected_docs=["chunking_comparison.txt", "splitting_strategies.txt"],
            ground_truth="固定大小切分简单但可能破坏语义；递归切分更智能但参数敏感；语义切分保持完整性但计算成本高；选择需根据具体场景权衡。",
            category="reasoning",
            difficulty="hard",
            tags=["chunking", "comparison", "strategies"]
        ),
        TestCase(
            id="test_004",
            query="如何优化RAG系统的响应延迟？",
            expected_docs=["latency_optimization.txt", "performance_tuning.txt"],
            ground_truth="优化方法包括：使用缓存、预计算Embedding、优化检索算法、减少LLM调用次数、使用流式输出、选择更快的模型等。",
            category="technical",
            difficulty="hard",
            tags=["optimization", "latency", "performance"]
        ),
        TestCase(
            id="test_005",
            query="流式输出的实现原理",
            expected_docs=["streaming.txt", "sse.txt"],
            ground_truth="流式输出通过Server-Sent Events(SSE)或WebSocket实现，LLM逐步生成token并实时推送给客户端，减少首字节延迟，提升用户体验。",
            category="technical",
            difficulty="medium",
            tags=["streaming", "sse", "implementation"]
        ),
    ]
    
    return TestDataset(
        name="RAG测试集",
        description="用于最终评估RAG系统性能的测试数据集",
        version="1.0",
        created_at=datetime.now().isoformat(),
        test_cases=test_cases
    )


def get_edge_case_test_set() -> TestDataset:
    """获取边缘情况测试集"""
    test_cases = [
        TestCase(
            id="edge_001",
            query="",  # 空查询
            expected_docs=[],
            ground_truth="请提供有效的问题。",
            category="edge_case",
            difficulty="easy",
            tags=["empty", "validation"]
        ),
        TestCase(
            id="edge_002",
            query="这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的查询语句，用于测试系统对超长输入的处理能力，看看是否能够正确截断或者处理这种异常情况",
            expected_docs=[],
            ground_truth="您的查询过长，请简化问题后重试。",
            category="edge_case",
            difficulty="medium",
            tags=["long_input", "validation"]
        ),
        TestCase(
            id="edge_003",
            query="asdfghjkl qwertyuiop",  # 无意义输入
            expected_docs=[],
            ground_truth="抱歉，我无法理解您的问题。请使用更有意义的语言描述您的问题。",
            category="edge_case",
            difficulty="easy",
            tags=["nonsense", "validation"]
        ),
        TestCase(
            id="edge_004",
            query="文档中完全没有提到的冷门话题xyz123",
            expected_docs=[],
            ground_truth="抱歉，我在知识库中没有找到相关信息。请尝试其他问题或上传相关文档。",
            category="edge_case",
            difficulty="medium",
            tags=["no_match", "fallback"]
        ),
        TestCase(
            id="edge_005",
            query="帮我写一段代码",  # 模糊查询
            expected_docs=[],
            ground_truth="请提供更具体的需求，例如您想使用什么编程语言，实现什么功能。",
            category="edge_case",
            difficulty="medium",
            tags=["ambiguous", "clarification"]
        ),
    ]
    
    return TestDataset(
        name="边缘情况测试集",
        description="用于测试系统对异常输入的处理能力",
        version="1.0",
        created_at=datetime.now().isoformat(),
        test_cases=test_cases
    )


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, data_dir: str = None):
        from config.settings import settings
        self.data_dir = Path(data_dir or (settings.BASE_DIR / "data" / "test_datasets"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets: Dict[str, TestDataset] = {}
        self._load_all()
    
    def _load_all(self):
        """加载所有数据集"""
        for json_file in self.data_dir.glob("*.json"):
            try:
                dataset = TestDataset.load(str(json_file))
                self.datasets[dataset.name] = dataset
            except Exception as e:
                print(f"加载数据集失败 {json_file}: {e}")
    
    def initialize_default_datasets(self):
        """初始化默认数据集"""
        datasets = [
            ("training_set.json", get_default_training_set()),
            ("validation_set.json", get_default_validation_set()),
            ("test_set.json", get_default_test_set()),
            ("edge_case_set.json", get_edge_case_test_set()),
        ]
        
        for filename, dataset in datasets:
            path = self.data_dir / filename
            dataset.save(str(path))
            self.datasets[dataset.name] = dataset
            print(f"✅ 已创建数据集: {dataset.name} ({len(dataset.test_cases)} 个用例)")
    
    def get_dataset(self, name: str) -> Optional[TestDataset]:
        """获取数据集"""
        return self.datasets.get(name)
    
    def get_all_queries(self, dataset_names: List[str] = None) -> List[Dict]:
        """获取所有测试查询"""
        queries = []
        
        datasets_to_use = (
            {k: v for k, v in self.datasets.items() if k in dataset_names}
            if dataset_names else self.datasets
        )
        
        for dataset in datasets_to_use.values():
            for tc in dataset.test_cases:
                queries.append({
                    "query": tc.query,
                    "expected_docs": tc.expected_docs,
                    "ground_truth": tc.ground_truth,
                    "category": tc.category,
                    "difficulty": tc.difficulty,
                    "dataset": dataset.name
                })
        
        return queries
    
    def get_queries_for_evaluation(self) -> List[Dict]:
        """获取用于评估的查询（排除边缘情况）"""
        return self.get_all_queries(
            ["RAG训练集", "RAG验证集", "RAG测试集"]
        )
    
    def get_statistics(self) -> Dict:
        """获取数据集统计"""
        stats = {
            "total_datasets": len(self.datasets),
            "total_test_cases": sum(len(ds.test_cases) for ds in self.datasets.values()),
            "by_dataset": {}
        }
        
        for name, dataset in self.datasets.items():
            stats["by_dataset"][name] = {
                "count": len(dataset.test_cases),
                "categories": dataset._count_by("category"),
                "difficulties": dataset._count_by("difficulty")
            }
        
        return stats


# 便捷函数
def get_test_queries() -> List[Dict]:
    """快速获取测试查询"""
    manager = DatasetManager()
    
    # 如果没有数据集，初始化默认数据集
    if not manager.datasets:
        manager.initialize_default_datasets()
    
    return manager.get_queries_for_evaluation()


if __name__ == "__main__":
    # 初始化并保存默认数据集
    manager = DatasetManager()
    manager.initialize_default_datasets()
    
    # 打印统计信息
    print("\n📊 数据集统计:")
    stats = manager.get_statistics()
    print(f"  总数据集数: {stats['total_datasets']}")
    print(f"  总测试用例数: {stats['total_test_cases']}")
    
    for name, ds_stats in stats["by_dataset"].items():
        print(f"\n  {name}:")
        print(f"    用例数: {ds_stats['count']}")
        print(f"    分类: {ds_stats['categories']}")
        print(f"    难度: {ds_stats['difficulties']}")