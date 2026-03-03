# 指标与测试系统指南 - Metrics & Testing Guide

## 📋 概述

本项目实现了一套完整的指标收集和测试系统，包括：

- **性能指标收集**：响应延迟、吞吐量、并发数等
- **质量指标评估**：检索精确率/召回率、生成质量评分、幻觉检测等
- **A/B测试框架**：支持不同配置/模型的对比实验
- **高并发压力测试**：测试系统极限和瓶颈
- **基准测试**：与基线对比的性能评估

## 🏗️ 系统架构

```
src/metrics/
├── __init__.py           # 模块入口
├── collector.py          # 指标收集器（核心）
├── performance.py        # 性能追踪器
├── quality.py           # 质量评估器
└── ab_testing.py        # A/B测试框架

tests/
├── __init__.py          # 测试模块入口
├── load_test.py         # 负载/压力测试
├── benchmark.py         # 基准测试
└── run_all_metrics.py   # 综合测试运行器
```

## 📊 指标类型

### 1. 性能指标 (Performance Metrics)

| 指标名 | 描述 | 单位 |
|--------|------|------|
| `latency` | 请求响应延迟 | ms |
| `requests_per_second` | 每秒请求数 (RPS) | count/s |
| `concurrent_requests` | 当前并发请求数 | count |
| `p50/p90/p95/p99` | 延迟百分位数 | ms |

### 2. 质量指标 (Quality Metrics)

| 指标名 | 描述 | 范围 |
|--------|------|------|
| `retrieval_precision` | 检索精确率 | 0-1 |
| `retrieval_recall` | 检索召回率 | 0-1 |
| `retrieval_mrr` | 平均倒数排名 | 0-1 |
| `retrieval_ndcg` | 归一化折损累积增益 | 0-1 |
| `generation_faithfulness` | 生成忠实度 | 0-1 |
| `generation_relevance` | 生成相关性 | 0-1 |
| `hallucination_rate` | 幻觉率 | 0-1 |

### 3. 负载测试指标 (Load Test Metrics)

| 指标名 | 描述 | 单位 |
|--------|------|------|
| `total_requests` | 总请求数 | count |
| `successful_requests` | 成功请求数 | count |
| `failed_requests` | 失败请求数 | count |
| `success_rate` | 成功率 | % |
| `avg_latency_ms` | 平均延迟 | ms |
| `max_concurrent_users` | 最大并发用户数 | count |

## 🚀 快速开始

### 1. 运行综合指标测试

```bash
# 运行所有测试
python tests/run_all_metrics.py

# 跳过负载测试
python tests/run_all_metrics.py --skip-load-test

# 指定输出目录
python tests/run_all_metrics.py --output-dir ./my_reports
```

### 2. 在代码中使用指标收集

```python
from src.metrics import metrics_collector, performance_tracker

# 使用上下文管理器追踪延迟
with performance_tracker.track_latency("my_operation"):
    # 执行操作
    do_something()

# 手动记录指标
metrics_collector.record(
    metric_type="custom",
    metric_name="my_metric",
    value=123.45,
    unit="ms"
)

# 获取统计信息
stats = metrics_collector.get_stats("custom.my_metric")
print(stats)
# {'count': 10, 'mean': 120.5, 'min': 100.0, 'max': 150.0, 'latest': 123.45}
```

### 3. 使用质量评估器

```python
from src.metrics.quality import quality_evaluator

# 评估检索质量
retrieval_result = quality_evaluator.evaluate_retrieval(
    query="Python是什么？",
    retrieved_docs=["doc1.py", "doc2.py"],
    expected_docs=["doc1.py", "doc3.py"]
)

print(f"精确率: {retrieval_result.precision}")
print(f"召回率: {retrieval_result.recall}")
print(f"MRR: {retrieval_result.mrr}")

# 评估生成质量
generation_result = quality_evaluator.evaluate_generation(
    query="Python是什么？",
    response="Python是一种编程语言...",
    context="Python是由Guido van Rossum创建的编程语言...",
    ground_truth="Python是一门解释型编程语言"
)

print(f"忠实度: {generation_result.faithfulness_score}")
print(f"是否幻觉: {generation_result.hallucination_detected}")
```

### 4. 运行A/B测试

```python
from src.metrics.ab_testing import ab_testing

# 创建实验
ab_testing.create_experiment(
    experiment_id="model_comparison",
    name="GPT-3.5 vs GPT-4",
    description="对比不同模型的性能",
    variants=[
        {"name": "control", "config": {"model": "gpt-3.5-turbo"}, "weight": 0.5, "is_control": True},
        {"name": "treatment", "config": {"model": "gpt-4"}, "weight": 0.5}
    ],
    metrics=["latency_ms", "quality_score"]
)

# 启动实验
ab_testing.start_experiment("model_comparison")

# 为用户分配变体
variant = ab_testing.assign_variant("model_comparison", "user_123")
print(f"用户分配到: {variant.name}")

# 记录指标
ab_testing.record_metric(
    experiment_id="model_comparison",
    variant_name=variant.name,
    metric_name="latency_ms",
    value=150.5,
    user_id="user_123"
)

# 分析结果
analysis = ab_testing.analyze_experiment("model_comparison")
print(f"推荐: {analysis['recommendation']}")
print(f"原因: {analysis['recommendation_reason']}")
```

### 5. 运行负载测试

```python
from tests.load_test import LoadTester, LoadTestConfig, StressTester

# 配置负载测试
config = LoadTestConfig(
    concurrent_users=20,
    total_requests=200,
    ramp_up_seconds=10.0,
    think_time_seconds=1.0
)

# 定义测试函数
def test_query(request_id: int, user_id: int) -> str:
    # 执行实际查询
    return rag_system.query("测试问题")

# 运行测试
tester = LoadTester()
result = tester.run_test(test_query, config)

print(f"RPS: {result.requests_per_second}")
print(f"P95延迟: {result.p95_latency_ms}ms")
print(f"成功率: {result.successful_requests / result.total_requests * 100}%")

# 压力测试 - 找崩溃点
stress_tester = StressTester()
breaking_point = stress_tester.find_breaking_point(
    test_query,
    start_users=10,
    max_users=100,
    step=10
)
print(f"系统崩溃点: {breaking_point['breaking_point']} 并发用户")
```

### 6. 运行基准测试

```python
from tests.benchmark import BenchmarkSuite, BenchmarkConfig

# 创建基准测试套件
suite = BenchmarkSuite("RAG性能测试", "对比不同配置")

# 添加基线配置
suite.add_benchmark(
    BenchmarkConfig(
        name="baseline",
        description="基线配置",
        config={"chunk_size": 800, "top_k": 3}
    ),
    is_baseline=True
)

# 添加对比配置
suite.add_benchmark(
    BenchmarkConfig(
        name="large_chunks",
        description="大块切分",
        config={"chunk_size": 1200, "top_k": 3}
    )
)

# 定义测试函数
def test_func(config: dict) -> dict:
    # 使用配置执行测试
    latency = run_rag_query(config)
    return {"latency_ms": latency}

# 运行测试
results = suite.run_all(test_func)

# 对比结果
comparison = suite.compare_with_baseline()
print(f"最佳配置: {comparison.winner}")
print(comparison.summary)

# 导出报告
suite.export_report("benchmark_report.json")
```

## 📈 指标数据存储

所有指标数据存储在 `metrics/` 目录下：

```
metrics/
├── metrics.db           # 指标SQLite数据库
├── quality.db           # 质量评估数据库
├── ab_testing.db        # A/B测试数据库
└── reports/             # 测试报告目录
    ├── comprehensive_report_YYYYMMDD_HHMMSS.json
    └── latest_report.json
```

## 📝 测试报告格式

综合测试报告示例：

```json
{
  "generated_at": "2024-01-15T10:30:00",
  "total_execution_time_seconds": 45.2,
  "summary": {
    "performance": {
      "total_metric_types": 15
    },
    "quality": {
      "avg_precision": 0.85,
      "avg_recall": 0.78,
      "avg_faithfulness": 0.92
    },
    "load_test": {
      "rps": 25.5,
      "avg_latency_ms": 156.3
    },
    "benchmark": {
      "winner": "large_chunks"
    }
  },
  "results": {
    "performance": {...},
    "quality": {...},
    "load_test": {...},
    "benchmark": {...}
  }
}
```

## 🔧 集成到现有系统

### 1. 在 ChatService 中集成性能追踪

```python
from src.metrics.performance import performance_tracker

class ChatService:
    def chat(self, prompt: str, session_id: str, project_id: str):
        with performance_tracker.track_latency("chat_request"):
            # 原有逻辑
            response = self.agent_app.invoke(...)
        
        return response
```

### 2. 在检索器中集成质量评估

```python
from src.metrics.quality import quality_evaluator

class VectorRetriever:
    def query_with_eval(self, question: str, expected_docs: List[str] = None):
        results = self.query(question)
        
        if expected_docs:
            retrieved = [doc.metadata.get("source") for doc, _ in results]
            quality_evaluator.evaluate_retrieval(
                query=question,
                retrieved_docs=retrieved,
                expected_docs=expected_docs
            )
        
        return results
```

### 3. 在 Web API 中添加指标端点

```python
# 在 web_app.py 中添加
from src.metrics import metrics_collector, performance_tracker

@app.get("/api/metrics")
def get_metrics():
    return {
        "performance": performance_tracker.get_full_report(),
        "stats": metrics_collector.get_all_stats()
    }

@app.get("/api/metrics/report")
def get_report():
    return metrics_collector.export_report()
```

## ⚠️ 注意事项

1. **生产环境使用**：指标收集会带来少量性能开销，建议在生产环境中设置采样率

2. **数据清理**：定期清理旧的指标数据，避免数据库膨胀

3. **敏感信息**：确保指标中不包含用户隐私数据

4. **并发安全**：所有指标收集组件都是线程安全的

## 📚 参考资料

- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
- [RAGAS Framework](https://docs.ragas.io/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)
- [Load Testing Best Practices](https://locust.io/)

---

**文档版本**：1.0  
**最后更新**：2024年