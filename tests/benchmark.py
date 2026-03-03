# tests/benchmark.py
"""
基准测试 - Benchmark Testing
对比不同配置、模型、参数的系统性能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.metrics.collector import metrics_collector
from src.metrics.performance import performance_tracker
from src.metrics.quality import quality_evaluator
from src.utils.logger import setup_logger

logger = setup_logger("BENCHMARK")


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    name: str
    description: str
    config: Dict[str, Any]  # 配置参数
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    config: BenchmarkConfig
    metrics: Dict[str, float]
    execution_time_seconds: float
    timestamp: str
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "config": self.config.config,
                "tags": self.config.tags
            },
            "metrics": self.metrics,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "timestamp": self.timestamp
        }


@dataclass
class ComparisonResult:
    """对比结果"""
    baseline_name: str
    baseline_metrics: Dict[str, float]
    comparisons: List[Dict[str, Any]]
    winner: Optional[str] = None
    summary: str = ""


class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.benchmarks: List[BenchmarkConfig] = []
        self.results: List[BenchmarkResult] = []
        self.baseline: Optional[BenchmarkConfig] = None
    
    def add_benchmark(self, config: BenchmarkConfig, is_baseline: bool = False):
        """添加基准测试配置"""
        self.benchmarks.append(config)
        
        if is_baseline:
            self.baseline = config
        
        logger.info(f"📝 添加基准测试: {config.name}")
    
    def run_benchmark(
        self,
        config: BenchmarkConfig,
        test_func: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> BenchmarkResult:
        """
        运行单个基准测试
        
        Args:
            config: 测试配置
            test_func: 测试函数，接收配置参数，返回指标字典
        """
        logger.info(f"🏃 运行基准测试: {config.name}")
        
        start_time = time.time()
        
        try:
            with performance_tracker.track_latency(f"benchmark_{config.name}"):
                metrics = test_func(config.config)
            
            success = True
        except Exception as e:
            logger.error(f"❌ 基准测试失败: {config.name} - {e}")
            metrics = {"error": 1, "error_message": str(e)}
            success = False
        
        end_time = time.time()
        
        result = BenchmarkResult(
            config=config,
            metrics=metrics,
            execution_time_seconds=end_time - start_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        
        # 记录指标
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_collector.record(
                    metric_type="benchmark",
                    metric_name=f"{config.name}.{metric_name}",
                    value=value,
                    tags={"benchmark": config.name}
                )
        
        return result
    
    def run_all(
        self,
        test_func: Callable[[Dict[str, Any]], Dict[str, float]]
    ) -> List[BenchmarkResult]:
        """运行所有基准测试"""
        logger.info(f"🚀 开始基准测试套件: {self.name}")
        
        self.results = []
        
        for config in self.benchmarks:
            result = self.run_benchmark(config, test_func)
            self.results.append(result)
        
        logger.info(f"✅ 基准测试套件完成: {len(self.results)} 个测试")
        
        return self.results
    
    def compare_with_baseline(self) -> Optional[ComparisonResult]:
        """与基线对比"""
        if not self.baseline or not self.results:
            return None
        
        baseline_result = None
        for r in self.results:
            if r.config.name == self.baseline.name:
                baseline_result = r
                break
        
        if not baseline_result:
            return None
        
        comparisons = []
        
        for result in self.results:
            if result.config.name == self.baseline.name:
                continue
            
            comparison = {
                "name": result.config.name,
                "metrics_comparison": {},
                "overall_improvement": 0.0
            }
            
            improvements = []
            
            for metric_name, value in result.metrics.items():
                baseline_value = baseline_result.metrics.get(metric_name, 0)
                
                if baseline_value > 0:
                    change_pct = (value - baseline_value) / baseline_value * 100
                    
                    # 对于延迟类指标，负变化是好的
                    if "latency" in metric_name or "time" in metric_name:
                        improvement = -change_pct
                    else:
                        improvement = change_pct
                else:
                    change_pct = 0
                    improvement = 0
                
                comparison["metrics_comparison"][metric_name] = {
                    "value": value,
                    "baseline_value": baseline_value,
                    "change_percent": round(change_pct, 2),
                    "improvement": improvement > 0
                }
                
                improvements.append(improvement)
            
            if improvements:
                comparison["overall_improvement"] = sum(improvements) / len(improvements)
            
            comparisons.append(comparison)
        
        # 确定最佳配置
        best = max(comparisons, key=lambda x: x["overall_improvement"])
        winner = best["name"] if best["overall_improvement"] > 5 else self.baseline.name
        
        summary = self._generate_comparison_summary(baseline_result, comparisons)
        
        return ComparisonResult(
            baseline_name=self.baseline.name,
            baseline_metrics=baseline_result.metrics,
            comparisons=comparisons,
            winner=winner,
            summary=summary
        )
    
    def _generate_comparison_summary(
        self,
        baseline: BenchmarkResult,
        comparisons: List[Dict]
    ) -> str:
        """生成对比摘要"""
        lines = [f"基准线: {baseline.config.name}"]
        lines.append(f"指标: {json.dumps(baseline.metrics, ensure_ascii=False)}")
        lines.append("")
        
        for comp in comparisons:
            lines.append(f"对比: {comp['name']}")
            lines.append(f"  整体改进: {comp['overall_improvement']:.1f}%")
            
            for metric, data in comp["metrics_comparison"].items():
                symbol = "✅" if data["improvement"] else "❌"
                value = data['value'] if isinstance(data['value'], (int, float)) else 0
                baseline_value = data['baseline_value'] if isinstance(data['baseline_value'], (int, float)) else 0
                lines.append(
                    f"  {symbol} {metric}: {value:.2f} vs {baseline_value:.2f} "
                    f"({data['change_percent']:+.1f}%)"
                )
        
        return "\n".join(lines)
    
    def export_report(self, output_path: str) -> Dict:
        """导出报告"""
        comparison = self.compare_with_baseline()
        
        report = {
            "suite_name": self.name,
            "suite_description": self.description,
            "generated_at": datetime.now().isoformat(),
            "benchmarks": [r.to_dict() for r in self.results],
            "comparison": {
                "baseline_name": comparison.baseline_name if comparison else None,
                "baseline_metrics": comparison.baseline_metrics if comparison else None,
                "comparisons": comparison.comparisons if comparison else [],
                "winner": comparison.winner if comparison else None,
                "summary": comparison.summary if comparison else ""
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 基准测试报告已导出: {output_path}")
        
        return report


class RAGBenchmark:
    """RAG系统专用基准测试"""
    
    # 预定义的配置模板
    CONFIG_TEMPLATES = {
        "gpt35_turbo": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "chunk_size": 800,
            "chunk_overlap": 100,
            "top_k": 3
        },
        "gpt35_turbo_large_chunks": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "chunk_size": 1200,
            "chunk_overlap": 200,
            "top_k": 3
        },
        "gpt35_turbo_more_chunks": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "chunk_size": 800,
            "chunk_overlap": 100,
            "top_k": 5
        },
        "gpt4": {
            "model": "gpt-4",
            "temperature": 0.1,
            "chunk_size": 800,
            "chunk_overlap": 100,
            "top_k": 3
        }
    }
    
    def __init__(self):
        self.suite = BenchmarkSuite(
            "RAG Benchmark Suite",
            "RAG系统性能基准测试"
        )
    
    def setup_from_templates(
        self,
        template_names: List[str],
        baseline_template: str = "gpt35_turbo"
    ):
        """从模板设置基准测试"""
        for name in template_names:
            if name in self.CONFIG_TEMPLATES:
                config = BenchmarkConfig(
                    name=name,
                    description=f"配置: {name}",
                    config=self.CONFIG_TEMPLATES[name].copy(),
                    tags=["rag", "predefined"]
                )
                self.suite.add_benchmark(
                    config,
                    is_baseline=(name == baseline_template)
                )
    
    def run_retrieval_benchmark(
        self,
        queries: List[str],
        retriever_func: Callable[[str, Dict], List[Tuple]],
        iterations: int = 3
    ) -> List[BenchmarkResult]:
        """
        运行检索基准测试
        
        Args:
            queries: 测试查询列表
            retriever_func: 检索函数 (query, config) -> [(doc, score), ...]
            iterations: 每个配置运行的次数
        """
        def test_func(config: Dict) -> Dict[str, float]:
            total_time = 0
            total_results = 0
            
            for _ in range(iterations):
                for query in queries:
                    start = time.time()
                    results = retriever_func(query, config)
                    total_time += time.time() - start
                    total_results += len(results)
            
            avg_latency_ms = (total_time / (len(queries) * iterations)) * 1000
            avg_results = total_results / (len(queries) * iterations)
            
            return {
                "avg_latency_ms": avg_latency_ms,
                "avg_results_count": avg_results,
                "total_time_seconds": total_time
            }
        
        return self.suite.run_all(test_func)
    
    def run_generation_benchmark(
        self,
        queries: List[str],
        generator_func: Callable[[str, Dict], str],
        iterations: int = 2
    ) -> List[BenchmarkResult]:
        """
        运行生成基准测试
        
        Args:
            queries: 测试查询列表
            generator_func: 生成函数 (query, config) -> response
            iterations: 每个配置运行的次数
        """
        def test_func(config: Dict) -> Dict[str, float]:
            total_time = 0
            total_tokens = 0
            
            for _ in range(iterations):
                for query in queries:
                    start = time.time()
                    response = generator_func(query, config)
                    total_time += time.time() - start
                    
                    # 粗略估计token数
                    total_tokens += len(response) / 2  # 中文约2字符/token
            
            avg_latency_ms = (total_time / (len(queries) * iterations)) * 1000
            avg_tokens = total_tokens / (len(queries) * iterations)
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            
            return {
                "avg_latency_ms": avg_latency_ms,
                "avg_response_tokens": avg_tokens,
                "tokens_per_second": tokens_per_second,
                "total_time_seconds": total_time
            }
        
        return self.suite.run_all(test_func)
    
    def run_end_to_end_benchmark(
        self,
        queries: List[Tuple[str, str]],  # (query, expected_answer)
        rag_func: Callable[[str, Dict], str],
        iterations: int = 2
    ) -> List[BenchmarkResult]:
        """
        运行端到端基准测试
        
        Args:
            queries: 测试查询和期望答案列表
            rag_func: RAG函数 (query, config) -> response
            iterations: 每个配置运行的次数
        """
        def test_func(config: Dict) -> Dict[str, float]:
            total_time = 0
            total_faithfulness = 0
            total_relevance = 0
            
            for _ in range(iterations):
                for query, expected in queries:
                    start = time.time()
                    response = rag_func(query, config)
                    total_time += time.time() - start
                    
                    # 评估质量
                    eval_result = quality_evaluator.evaluate_generation(
                        query=query,
                        response=response,
                        ground_truth=expected
                    )
                    
                    total_faithfulness += eval_result.faithfulness_score
                    total_relevance += eval_result.relevance_score
            
            count = len(queries) * iterations
            
            return {
                "avg_latency_ms": (total_time / count) * 1000,
                "avg_faithfulness": total_faithfulness / count,
                "avg_relevance": total_relevance / count,
                "total_time_seconds": total_time
            }
        
        return self.suite.run_all(test_func)
    
    def get_comparison(self) -> Optional[ComparisonResult]:
        """获取对比结果"""
        return self.suite.compare_with_baseline()
    
    def export_report(self, output_path: str) -> Dict:
        """导出报告"""
        return self.suite.export_report(output_path)


def run_quick_benchmark(
    test_func: Callable[[Dict], Dict[str, float]],
    configs: Dict[str, Dict],
    baseline_name: str = None
) -> Tuple[List[BenchmarkResult], Optional[ComparisonResult]]:
    """
    快速运行基准测试
    
    Args:
        test_func: 测试函数
        configs: 配置字典 {name: config}
        baseline_name: 基线配置名称
    """
    suite = BenchmarkSuite("Quick Benchmark", "快速基准测试")
    
    for name, config in configs.items():
        suite.add_benchmark(
            BenchmarkConfig(
                name=name,
                description=f"配置: {name}",
                config=config
            ),
            is_baseline=(name == baseline_name)
        )
    
    results = suite.run_all(test_func)
    comparison = suite.compare_with_baseline()
    
    return results, comparison


# 示例用法
if __name__ == "__main__":
    # 示例测试函数
    def sample_rag_test(config: Dict) -> Dict[str, float]:
        # 模拟延迟
        latency = 0.1 + config.get("chunk_size", 800) / 10000
        time.sleep(latency)
        
        return {
            "latency_ms": latency * 1000,
            "quality_score": 0.85 + config.get("top_k", 3) * 0.02
        }
    
    # 创建基准测试
    suite = BenchmarkSuite("RAG性能测试", "测试不同chunk_size和top_k的影响")
    
    # 添加配置
    suite.add_benchmark(
        BenchmarkConfig("baseline", "基线配置", {"chunk_size": 800, "top_k": 3}),
        is_baseline=True
    )
    
    suite.add_benchmark(
        BenchmarkConfig("large_chunks", "大块切分", {"chunk_size": 1200, "top_k": 3})
    )
    
    suite.add_benchmark(
        BenchmarkConfig("more_results", "更多结果", {"chunk_size": 800, "top_k": 5})
    )
    
    # 运行测试
    results = suite.run_all(sample_rag_test)
    
    # 对比
    comparison = suite.compare_with_baseline()
    
    print("\n" + "="*60)
    print("基准测试结果:")
    print("="*60)
    
    for result in results:
        print(f"\n{result.config.name}:")
        for metric, value in result.metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    if comparison:
        print(f"\n最佳配置: {comparison.winner}")
        print(f"\n{comparison.summary}")