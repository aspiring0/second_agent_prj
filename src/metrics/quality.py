# src/metrics/quality.py
"""
质量评估器 - Quality Evaluator
评估RAG系统的检索质量和生成质量
"""
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sqlite3

from src.metrics.collector import metrics_collector
from src.utils.logger import setup_logger

logger = setup_logger("QUALITY")


@dataclass
class RetrievalEvaluation:
    """检索评估结果"""
    query: str
    retrieved_docs: List[str]
    relevance_scores: List[float]
    precision: float
    recall: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain


@dataclass
class GenerationEvaluation:
    """生成评估结果"""
    query: str
    response: str
    ground_truth: Optional[str]
    faithfulness_score: float  # 忠实度（是否基于上下文）
    relevance_score: float  # 相关性
    completeness_score: float  # 完整性
    coherence_score: float  # 连贯性
    hallucination_detected: bool  # 是否检测到幻觉


@dataclass
class TestCase:
    """测试用例"""
    id: str
    query: str
    expected_docs: List[str] = field(default_factory=list)  # 期望检索到的文档
    ground_truth: Optional[str] = None  # 期望的回答
    tags: List[str] = field(default_factory=list)


class QualityEvaluator:
    """质量评估器"""
    
    _instance = None
    _lock = None  # 将在 __new__ 中初始化
    
    def __new__(cls):
        if cls._instance is None:
            cls._lock = type('_lock', (), {})()  # 创建锁对象占位
            import threading
            cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        
        # 测试用例存储
        self._test_cases: Dict[str, TestCase] = {}
        
        # 评估结果存储
        self._retrieval_results: List[RetrievalEvaluation] = []
        self._generation_results: List[GenerationEvaluation] = []
        
        # 初始化数据库
        self._init_quality_db()
        
        logger.info("✅ 质量评估器初始化完成")
    
    def _init_quality_db(self):
        """初始化质量评估数据库"""
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        db_path = metrics_dir / "quality.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 测试用例表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                expected_docs TEXT,
                ground_truth TEXT,
                tags TEXT,
                created_at TEXT
            )
        """)
        
        # 检索评估结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_case_id TEXT,
                query TEXT NOT NULL,
                retrieved_docs TEXT,
                relevance_scores TEXT,
                precision REAL,
                recall REAL,
                mrr REAL,
                ndcg REAL,
                evaluated_at TEXT
            )
        """)
        
        # 生成评估结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_case_id TEXT,
                query TEXT NOT NULL,
                response TEXT,
                ground_truth TEXT,
                faithfulness_score REAL,
                relevance_score REAL,
                completeness_score REAL,
                coherence_score REAL,
                hallucination_detected INTEGER,
                evaluated_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_test_case(
        self, 
        test_id: str, 
        query: str, 
        expected_docs: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """添加测试用例"""
        test_case = TestCase(
            id=test_id,
            query=query,
            expected_docs=expected_docs or [],
            ground_truth=ground_truth,
            tags=tags or []
        )
        
        self._test_cases[test_id] = test_case
        
        # 持久化
        self._save_test_case(test_case)
        
        logger.info(f"📝 添加测试用例: {test_id}")
    
    def _save_test_case(self, test_case: TestCase):
        """保存测试用例到数据库"""
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        db_path = metrics_dir / "quality.db"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO test_cases 
            (id, query, expected_docs, ground_truth, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            test_case.id,
            test_case.query,
            json.dumps(test_case.expected_docs),
            test_case.ground_truth,
            json.dumps(test_case.tags),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def load_test_cases(self) -> Dict[str, TestCase]:
        """从数据库加载测试用例"""
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        db_path = metrics_dir / "quality.db"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, query, expected_docs, ground_truth, tags FROM test_cases")
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            test_case = TestCase(
                id=row[0],
                query=row[1],
                expected_docs=json.loads(row[2]) if row[2] else [],
                ground_truth=row[3],
                tags=json.loads(row[4]) if row[4] else []
            )
            self._test_cases[row[0]] = test_case
        
        return self._test_cases
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[str],
        expected_docs: List[str],
        relevance_scores: Optional[List[float]] = None
    ) -> RetrievalEvaluation:
        """
        评估检索质量
        
        Args:
            query: 查询文本
            retrieved_docs: 检索到的文档列表
            expected_docs: 期望的文档列表
            relevance_scores: 相关性分数列表
        """
        if relevance_scores is None:
            relevance_scores = [1.0] * len(retrieved_docs)
        
        # 计算精确率
        retrieved_set = set(retrieved_docs)
        expected_set = set(expected_docs)
        
        true_positives = len(retrieved_set & expected_set)
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(expected_set) if expected_set else 0
        
        # 计算 MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, doc in enumerate(retrieved_docs):
            if doc in expected_set:
                mrr = 1.0 / (i + 1)
                break
        
        # 计算 NDCG (Normalized Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs):
            rel = 1 if doc in expected_set else 0
            dcg += rel / (i + 1)  # log2(i+2) 简化为 i+1
        
        ideal_dcg = sum(1 / (i + 1) for i in range(min(len(expected_docs), len(retrieved_docs))))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        evaluation = RetrievalEvaluation(
            query=query,
            retrieved_docs=retrieved_docs,
            relevance_scores=relevance_scores,
            precision=precision,
            recall=recall,
            mrr=mrr,
            ndcg=ndcg
        )
        
        self._retrieval_results.append(evaluation)
        
        # 记录指标
        metrics_collector.record(
            metric_type="quality",
            metric_name="retrieval_precision",
            value=precision,
            unit="ratio"
        )
        
        metrics_collector.record(
            metric_type="quality",
            metric_name="retrieval_recall",
            value=recall,
            unit="ratio"
        )
        
        metrics_collector.record(
            metric_type="quality",
            metric_name="retrieval_mrr",
            value=mrr,
            unit="ratio"
        )
        
        metrics_collector.record(
            metric_type="quality",
            metric_name="retrieval_ndcg",
            value=ndcg,
            unit="ratio"
        )
        
        logger.debug(f"📊 检索评估: Precision={precision:.3f}, Recall={recall:.3f}, MRR={mrr:.3f}")
        
        return evaluation
    
    def evaluate_generation(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> GenerationEvaluation:
        """
        评估生成质量
        
        Args:
            query: 用户查询
            response: 生成的回答
            context: 检索到的上下文
            ground_truth: 标准答案
        """
        # 忠实度评估（回答是否基于上下文）
        faithfulness_score = self._evaluate_faithfulness(response, context)
        
        # 相关性评估
        relevance_score = self._evaluate_relevance(query, response)
        
        # 完整性评估
        completeness_score = self._evaluate_completeness(query, response, ground_truth)
        
        # 连贯性评估
        coherence_score = self._evaluate_coherence(response)
        
        # 幻觉检测
        hallucination_detected = self._detect_hallucination(response, context)
        
        evaluation = GenerationEvaluation(
            query=query,
            response=response,
            ground_truth=ground_truth,
            faithfulness_score=faithfulness_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            coherence_score=coherence_score,
            hallucination_detected=hallucination_detected
        )
        
        self._generation_results.append(evaluation)
        
        # 记录指标
        metrics_collector.record(
            metric_type="quality",
            metric_name="generation_faithfulness",
            value=faithfulness_score,
            unit="score"
        )
        
        metrics_collector.record(
            metric_type="quality",
            metric_name="generation_relevance",
            value=relevance_score,
            unit="score"
        )
        
        metrics_collector.record(
            metric_type="quality",
            metric_name="generation_completeness",
            value=completeness_score,
            unit="score"
        )
        
        metrics_collector.record(
            metric_type="quality",
            metric_name="hallucination_rate",
            value=1.0 if hallucination_detected else 0.0,
            unit="boolean"
        )
        
        logger.debug(
            f"📊 生成评估: Faithfulness={faithfulness_score:.3f}, "
            f"Relevance={relevance_score:.3f}, Hallucination={hallucination_detected}"
        )
        
        return evaluation
    
    def _evaluate_faithfulness(self, response: str, context: Optional[str]) -> float:
        """评估忠实度（回答是否基于上下文）"""
        if not context:
            return 0.5  # 无上下文时返回中等分数
        
        # 简单的关键词匹配方法
        # 实际项目中可以使用 LLM 或 NLI 模型
        response_words = set(self._tokenize(response.lower()))
        context_words = set(self._tokenize(context.lower()))
        
        if not response_words:
            return 0.0
        
        overlap = response_words & context_words
        return len(overlap) / len(response_words)
    
    def _evaluate_relevance(self, query: str, response: str) -> float:
        """评估相关性"""
        query_words = set(self._tokenize(query.lower()))
        response_words = set(self._tokenize(response.lower()))
        
        if not query_words:
            return 0.5
        
        # 检查回答中是否包含查询关键词
        overlap = query_words & response_words
        return min(1.0, len(overlap) / len(query_words) + 0.3)
    
    def _evaluate_completeness(
        self, 
        query: str, 
        response: str, 
        ground_truth: Optional[str]
    ) -> float:
        """评估完整性"""
        if not ground_truth:
            # 无标准答案时，根据回答长度和结构评估
            word_count = len(self._tokenize(response))
            if word_count < 10:
                return 0.3
            elif word_count < 30:
                return 0.6
            else:
                return 0.8
        
        # 有标准答案时，计算重叠度
        truth_words = set(self._tokenize(ground_truth.lower()))
        response_words = set(self._tokenize(response.lower()))
        
        if not truth_words:
            return 0.5
        
        overlap = truth_words & response_words
        return len(overlap) / len(truth_words)
    
    def _evaluate_coherence(self, response: str) -> float:
        """评估连贯性"""
        sentences = re.split(r'[。！？.!?]', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.7
        
        # 检查句子长度是否合理
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        
        if avg_length < 5:
            return 0.4
        elif avg_length > 100:
            return 0.5
        else:
            return 0.8
    
    def _detect_hallucination(self, response: str, context: Optional[str]) -> bool:
        """检测幻觉"""
        if not context:
            return False  # 无上下文无法判断
        
        # 简单的启发式方法
        # 检查回答中是否有明确的断言但上下文中没有相关信息
        
        # 知识截止提示词
        cutoff_phrases = [
            "据我所知", "我了解到", "根据我的知识",
            "在最新的", "截止到目前"
        ]
        
        response_lower = response.lower()
        for phrase in cutoff_phrases:
            if phrase in response:
                return True
        
        return False
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 中英文混合分词
        words = []
        # 按空格分
        for part in text.split():
            # 中文按字符分
            for char in part:
                if '\u4e00' <= char <= '\u9fff':
                    words.append(char)
                else:
                    if words and not '\u4e00' <= words[-1][-1] <= '\u9fff':
                        words[-1] += char
                    else:
                        words.append(char)
        return words
    
    def get_retrieval_summary(self) -> Dict[str, float]:
        """获取检索评估摘要"""
        if not self._retrieval_results:
            return {}
        
        precisions = [r.precision for r in self._retrieval_results]
        recalls = [r.recall for r in self._retrieval_results]
        mrrs = [r.mrr for r in self._retrieval_results]
        ndcgs = [r.ndcg for r in self._retrieval_results]
        
        return {
            "total_evaluations": len(self._retrieval_results),
            "avg_precision": sum(precisions) / len(precisions),
            "avg_recall": sum(recalls) / len(recalls),
            "avg_mrr": sum(mrrs) / len(mrrs),
            "avg_ndcg": sum(ndcgs) / len(ndcgs)
        }
    
    def get_generation_summary(self) -> Dict[str, float]:
        """获取生成评估摘要"""
        if not self._generation_results:
            return {}
        
        faithfulness = [g.faithfulness_score for g in self._generation_results]
        relevance = [g.relevance_score for g in self._generation_results]
        completeness = [g.completeness_score for g in self._generation_results]
        coherence = [g.coherence_score for g in self._generation_results]
        hallucinations = sum(1 for g in self._generation_results if g.hallucination_detected)
        
        return {
            "total_evaluations": len(self._generation_results),
            "avg_faithfulness": sum(faithfulness) / len(faithfulness),
            "avg_relevance": sum(relevance) / len(relevance),
            "avg_completeness": sum(completeness) / len(completeness),
            "avg_coherence": sum(coherence) / len(coherence),
            "hallucination_rate": hallucinations / len(self._generation_results)
        }
    
    def get_full_report(self) -> Dict[str, Any]:
        """获取完整质量报告"""
        return {
            "timestamp": datetime.now().isoformat(),
            "retrieval_summary": self.get_retrieval_summary(),
            "generation_summary": self.get_generation_summary(),
            "test_cases_count": len(self._test_cases)
        }
    
    def export_report(self, output_path: str) -> Dict:
        """导出报告"""
        report = self.get_full_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 质量报告已导出: {output_path}")
        return report
    
    def reset(self):
        """重置评估数据"""
        self._retrieval_results.clear()
        self._generation_results.clear()
        logger.info("🔄 质量评估器已重置")


# 全局单例
quality_evaluator = QualityEvaluator()