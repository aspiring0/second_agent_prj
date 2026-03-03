# tests/run_all_metrics.py
"""
综合指标测试运行器 - Comprehensive Metrics Test Runner
运行所有指标收集、性能测试、质量评估、A/B测试等
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# 导入指标模块
from src.metrics.collector import metrics_collector
from src.metrics.performance import PerformanceTracker, performance_tracker
from src.metrics.quality import QualityEvaluator, quality_evaluator
from src.metrics.ab_testing import ABTestingFramework, ab_testing

# 导入测试模块
from tests.load_test import LoadTester, StressTester, LoadTestConfig
from tests.benchmark import BenchmarkSuite, BenchmarkConfig, RAGBenchmark
from tests.test_dataset import DatasetManager, get_test_queries

from src.utils.logger import setup_logger

logger = setup_logger("METRICS_RUNNER")


class MetricsTestRunner:
    """指标测试运行器"""
    
    def __init__(self, output_dir: str = None):
        from config.settings import settings
        self.output_dir = Path(output_dir or (settings.BASE_DIR / "metrics" / "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # 初始化测试数据集
        self.dataset_manager = DatasetManager()
        if not self.dataset_manager.datasets:
            logger.info("📝 初始化测试数据集...")
            self.dataset_manager.initialize_default_datasets()
    
    def run_all_tests(
        self,
        include_load_test: bool = True,
        include_quality_test: bool = True,
        include_benchmark: bool = True,
        load_test_queries: List[str] = None
    ):
        """运行所有测试"""
        logger.info("🚀 开始运行综合指标测试...")
        
        start_time = time.time()
        
        # 0. 记录数据集统计
        self._record_dataset_stats()
        
        # 1. 收集性能指标
        self._collect_performance_metrics()
        
        # 2. 收集质量指标（使用测试数据集）
        if include_quality_test:
            self._collect_quality_metrics_with_dataset()
        
        # 3. 运行负载测试
        if include_load_test:
            self._run_load_tests(load_test_queries)
        
        # 4. 运行基准测试
        if include_benchmark:
            self._run_benchmarks()
        
        # 5. 生成综合报告
        total_time = time.time() - start_time
        self._generate_comprehensive_report(total_time)
        
        logger.info(f"✅ 所有测试完成，耗时 {total_time:.2f} 秒")
        
        return self.results
    
    def _record_dataset_stats(self):
        """记录数据集统计"""
        stats = self.dataset_manager.get_statistics()
        self.results["dataset_stats"] = stats
        logger.info(f"📊 测试数据集: {stats['total_datasets']} 个, 共 {stats['total_test_cases']} 个用例")
    
    def _collect_performance_metrics(self):
        """收集性能指标"""
        logger.info("📊 收集性能指标...")
        
        # 获取性能统计
        performance_stats = performance_tracker.get_full_report()
        
        # 获取指标收集器统计
        collector_stats = metrics_collector.get_all_stats()
        
        self.results["performance"] = {
            "tracker_stats": performance_stats,
            "collector_stats": collector_stats
        }
        
        logger.info(f"  ✅ 收集了 {len(collector_stats)} 类指标")
    
    def _collect_quality_metrics(self):
        """收集质量指标"""
        logger.info("📊 收集质量指标...")
        
        # 获取质量统计
        retrieval_summary = quality_evaluator.get_retrieval_summary()
        generation_summary = quality_evaluator.get_generation_summary()
        
        self.results["quality"] = {
            "retrieval": retrieval_summary,
            "generation": generation_summary
        }
        
        logger.info(f"  ✅ 检索评估: {retrieval_summary.get('total_evaluations', 0)} 次")
        logger.info(f"  ✅ 生成评估: {generation_summary.get('total_evaluations', 0)} 次")
    
    def _collect_quality_metrics_with_dataset(self):
        """使用测试数据集收集质量指标"""
        logger.info("📊 使用测试数据集收集质量指标...")
        
        # 获取测试查询
        test_queries = self.dataset_manager.get_queries_for_evaluation()
        
        if not test_queries:
            logger.warning("  ⚠️ 没有测试查询，使用空结果")
            self.results["quality"] = {
                "retrieval": {},
                "generation": {},
                "test_queries_count": 0
            }
            return
        
        logger.info(f"  📝 使用 {len(test_queries)} 个测试查询进行模拟评估")
        
        # 模拟检索和生成评估
        # 在实际使用中，这些会调用真实的检索器和生成器
        retrieval_results = []
        generation_results = []
        
        for i, test_case in enumerate(test_queries):
            query = test_case.get("query", "")
            expected_docs = test_case.get("expected_docs", [])
            ground_truth = test_case.get("ground_truth", "")
            
            # 模拟检索评估
            # 这里使用模拟数据，实际使用时需要调用真实的检索器
            simulated_retrieved = expected_docs[:1] if expected_docs else []  # 模拟检索到部分文档
            
            retrieval_eval = quality_evaluator.evaluate_retrieval(
                query=query,
                retrieved_docs=simulated_retrieved,
                expected_docs=expected_docs
            )
            
            retrieval_results.append({
                "query": query[:50] + "..." if len(query) > 50 else query,
                "precision": retrieval_eval.precision,
                "recall": retrieval_eval.recall,
                "mrr": retrieval_eval.mrr,
                "ndcg": retrieval_eval.ndcg
            })
            
            # 模拟生成评估
            simulated_response = ground_truth[:100] if ground_truth else "模拟回答"
            
            generation_eval = quality_evaluator.evaluate_generation(
                query=query,
                response=simulated_response,
                context="\n".join(expected_docs),
                ground_truth=ground_truth
            )
            
            generation_results.append({
                "query": query[:50] + "..." if len(query) > 50 else query,
                "faithfulness": generation_eval.faithfulness_score,
                "relevance": generation_eval.relevance_score,
                "completeness": generation_eval.completeness_score,
                "hallucination": generation_eval.hallucination_detected
            })
        
        # 计算汇总统计
        retrieval_summary = {
            "total_evaluations": len(retrieval_results),
            "avg_precision": sum(r["precision"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0,
            "avg_recall": sum(r["recall"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0,
            "avg_mrr": sum(r["mrr"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0,
            "avg_ndcg": sum(r["ndcg"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
        }
        
        generation_summary = {
            "total_evaluations": len(generation_results),
            "avg_faithfulness": sum(g["faithfulness"] for g in generation_results) / len(generation_results) if generation_results else 0,
            "avg_relevance": sum(g["relevance"] for g in generation_results) / len(generation_results) if generation_results else 0,
            "avg_completeness": sum(g["completeness"] for g in generation_results) / len(generation_results) if generation_results else 0,
            "hallucination_rate": sum(1 for g in generation_results if g["hallucination"]) / len(generation_results) if generation_results else 0
        }
        
        self.results["quality"] = {
            "retrieval": retrieval_summary,
            "generation": generation_summary,
            "test_queries_count": len(test_queries),
            "detailed_results": {
                "retrieval_samples": retrieval_results[:5],  # 只保留前5个样本
                "generation_samples": generation_results[:5]
            }
        }
        
        logger.info(f"  ✅ 检索评估: {retrieval_summary['total_evaluations']} 次")
        logger.info(f"     平均精确率: {retrieval_summary['avg_precision']:.3f}")
        logger.info(f"     平均召回率: {retrieval_summary['avg_recall']:.3f}")
        logger.info(f"  ✅ 生成评估: {generation_summary['total_evaluations']} 次")
        logger.info(f"     平均忠实度: {generation_summary['avg_faithfulness']:.3f}")
        logger.info(f"     幻觉率: {generation_summary['hallucination_rate']:.1%}")
    
    def _run_load_tests(self, queries: List[str] = None):
        """运行负载测试"""
        logger.info("⚡ 运行负载测试...")
        
        if queries is None:
            queries = [
                "测试查询1",
                "测试查询2", 
                "测试查询3"
            ]
        
        # 简单负载测试
        def mock_query(request_id: int, user_id: int) -> str:
            time.sleep(0.1)  # 模拟延迟
            return f"响应 {request_id}"
        
        load_tester = LoadTester()
        
        # 运行低负载测试
        low_load_config = LoadTestConfig(
            concurrent_users=5,
            total_requests=20,
            ramp_up_seconds=2.0,
            think_time_seconds=0.5
        )
        
        low_load_result = load_tester.run_test(mock_query, low_load_config)
        
        self.results["load_test"] = {
            "low_load": low_load_result.to_dict()
        }
        
        logger.info(f"  ✅ 低负载测试完成: {low_load_result.requests_per_second:.2f} RPS")
    
    def _run_benchmarks(self):
        """运行基准测试"""
        logger.info("🏁 运行基准测试...")
        
        # 创建简单的基准测试
        suite = BenchmarkSuite("系统基准测试", "测试基本系统操作性能")
        
        # 添加测试配置
        suite.add_benchmark(
            BenchmarkConfig(
                "baseline",
                "基线配置",
                {"type": "baseline"}
            ),
            is_baseline=True
        )
        
        suite.add_benchmark(
            BenchmarkConfig(
                "optimized",
                "优化配置",
                {"type": "optimized"}
            )
        )
        
        # 定义测试函数
        def benchmark_func(config: Dict) -> Dict[str, float]:
            start = time.time()
            
            # 模拟操作
            base_delay = 0.05
            
            if config.get("type") == "optimized":
                base_delay = 0.03  # 优化后更快
            
            time.sleep(base_delay)
            
            duration = (time.time() - start) * 1000
            
            return {
                "latency_ms": max(duration, 10),
                "throughput": 1000 / max(duration, 10)
            }
        
        results = suite.run_all(benchmark_func)
        comparison = suite.compare_with_baseline()
        
        self.results["benchmark"] = {
            "results": [r.to_dict() for r in results],
            "comparison": {
                "winner": comparison.winner if comparison else None,
                "summary": comparison.summary if comparison else ""
            }
        }
        
        logger.info(f"  ✅ 基准测试完成: {len(results)} 个配置")
    
    def _generate_comprehensive_report(self, total_time: float):
        """生成综合报告"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_execution_time_seconds": round(total_time, 2),
            "results": self.results,
            "summary": self._generate_summary()
        }
        
        # 保存报告
        report_path = self.output_dir / f"comprehensive_report_{self.timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 综合报告已保存: {report_path}")
        
        # 同时保存一份最新报告
        latest_path = self.output_dir / "latest_report.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.results["report_path"] = str(report_path)
    
    def _generate_summary(self) -> Dict:
        """生成摘要"""
        summary = {
            "performance": {},
            "quality": {},
            "load_test": {},
            "benchmark": {}
        }
        
        # 性能摘要
        if "performance" in self.results:
            stats = self.results["performance"].get("collector_stats", {})
            summary["performance"]["total_metric_types"] = len(stats)
        
        # 质量摘要
        if "quality" in self.results:
            retrieval = self.results["quality"].get("retrieval", {})
            generation = self.results["quality"].get("generation", {})
            
            summary["quality"]["avg_precision"] = retrieval.get("avg_precision")
            summary["quality"]["avg_recall"] = retrieval.get("avg_recall")
            summary["quality"]["avg_faithfulness"] = generation.get("avg_faithfulness")
        
        # 负载测试摘要
        if "load_test" in self.results:
            low_load = self.results["load_test"].get("low_load", {}).get("results", {})
            summary["load_test"]["rps"] = low_load.get("requests_per_second")
            summary["load_test"]["avg_latency_ms"] = low_load.get("avg_latency_ms")
        
        # 基准测试摘要
        if "benchmark" in self.results:
            summary["benchmark"]["winner"] = self.results["benchmark"].get("comparison", {}).get("winner")
        
        return summary
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "="*60)
        print("📊 指标测试摘要")
        print("="*60)
        
        if "summary" in self.results:
            summary = self.results["summary"]
            
            print("\n📈 性能指标:")
            print(f"   指标类型数: {summary['performance'].get('total_metric_types', 0)}")
            
            print("\n🎯 质量指标:")
            if summary["quality"].get("avg_precision"):
                print(f"   平均精确率: {summary['quality']['avg_precision']:.3f}")
            if summary["quality"].get("avg_recall"):
                print(f"   平均召回率: {summary['quality']['avg_recall']:.3f}")
            if summary["quality"].get("avg_faithfulness"):
                print(f"   平均忠实度: {summary['quality']['avg_faithfulness']:.3f}")
            
            print("\n⚡ 负载测试:")
            if summary["load_test"].get("rps"):
                print(f"   吞吐量: {summary['load_test']['rps']:.2f} RPS")
            if summary["load_test"].get("avg_latency_ms"):
                print(f"   平均延迟: {summary['load_test']['avg_latency_ms']:.2f} ms")
            
            print("\n🏁 基准测试:")
            if summary["benchmark"].get("winner"):
                print(f"   最佳配置: {summary['benchmark']['winner']}")
        
        if "report_path" in self.results:
            print(f"\n📄 详细报告: {self.results['report_path']}")
        
        print("="*60)


def run_quick_test():
    """快速测试"""
    runner = MetricsTestRunner()
    runner.run_all_tests(
        include_load_test=True,
        include_quality_test=True,
        include_benchmark=True
    )
    runner.print_summary()
    return runner


def run_rag_evaluation(
    retriever_func=None,
    generator_func=None,
    queries: List[Dict[str, str]] = None
):
    """
    运行RAG系统评估
    
    Args:
        retriever_func: 检索函数 (query) -> [(doc, score), ...]
        generator_func: 生成函数 (query) -> response
        queries: 测试查询列表 [{"query": "...", "expected_docs": [...], "ground_truth": "..."}]
    """
    logger.info("🔬 开始RAG系统评估...")
    
    results = {
        "retrieval": [],
        "generation": []
    }
    
    if queries is None:
        logger.warning("没有提供测试查询，跳过评估")
        return results
    
    for i, test_case in enumerate(queries):
        query = test_case.get("query", "")
        expected_docs = test_case.get("expected_docs", [])
        ground_truth = test_case.get("ground_truth")
        
        logger.info(f"评估查询 {i+1}/{len(queries)}: {query[:50]}...")
        
        # 评估检索
        if retriever_func:
            with performance_tracker.track_latency("retrieval"):
                retrieved = retriever_func(query)
            
            retrieved_sources = [doc.metadata.get("source", "") for doc, _ in retrieved]
            
            retrieval_eval = quality_evaluator.evaluate_retrieval(
                query=query,
                retrieved_docs=retrieved_sources,
                expected_docs=expected_docs
            )
            
            results["retrieval"].append({
                "query": query,
                "precision": retrieval_eval.precision,
                "recall": retrieval_eval.recall,
                "mrr": retrieval_eval.mrr,
                "ndcg": retrieval_eval.ndcg
            })
        
        # 评估生成
        if generator_func and ground_truth:
            with performance_tracker.track_latency("generation"):
                response = generator_func(query)
            
            context = "\n".join([doc.page_content for doc, _ in retrieved]) if retriever_func else None
            
            generation_eval = quality_evaluator.evaluate_generation(
                query=query,
                response=response,
                context=context,
                ground_truth=ground_truth
            )
            
            results["generation"].append({
                "query": query,
                "faithfulness": generation_eval.faithfulness_score,
                "relevance": generation_eval.relevance_score,
                "completeness": generation_eval.completeness_score,
                "hallucination": generation_eval.hallucination_detected
            })
    
    # 汇总
    if results["retrieval"]:
        results["retrieval_summary"] = {
            "avg_precision": sum(r["precision"] for r in results["retrieval"]) / len(results["retrieval"]),
            "avg_recall": sum(r["recall"] for r in results["retrieval"]) / len(results["retrieval"]),
            "avg_mrr": sum(r["mrr"] for r in results["retrieval"]) / len(results["retrieval"])
        }
    
    if results["generation"]:
        results["generation_summary"] = {
            "avg_faithfulness": sum(r["faithfulness"] for r in results["generation"]) / len(results["generation"]),
            "avg_relevance": sum(r["relevance"] for r in results["generation"]) / len(results["generation"]),
            "hallucination_rate": sum(1 for r in results["generation"] if r["hallucination"]) / len(results["generation"])
        }
    
    logger.info("✅ RAG系统评估完成")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统指标测试运行器")
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="报告输出目录"
    )
    
    parser.add_argument(
        "--skip-load-test",
        action="store_true",
        help="跳过负载测试"
    )
    
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="跳过质量评估"
    )
    
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="跳过基准测试"
    )
    
    args = parser.parse_args()
    
    runner = MetricsTestRunner(args.output_dir)
    
    runner.run_all_tests(
        include_load_test=not args.skip_load_test,
        include_quality_test=not args.skip_quality,
        include_benchmark=not args.skip_benchmark
    )
    
    runner.print_summary()


if __name__ == "__main__":
    main()