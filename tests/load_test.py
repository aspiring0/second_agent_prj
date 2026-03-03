# tests/load_test.py
"""
高并发压力测试 - Load Testing
测试系统在高并发情况下的性能表现和瓶颈
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import concurrent.futures
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import queue
from pathlib import Path

from src.metrics.collector import metrics_collector
from src.metrics.performance import PerformanceTracker, performance_tracker
from src.utils.logger import setup_logger

logger = setup_logger("LOAD_TEST")


@dataclass
class LoadTestConfig:
    """负载测试配置"""
    concurrent_users: int = 10  # 并发用户数
    total_requests: int = 100  # 总请求数
    ramp_up_seconds: float = 10.0  # 用户逐步增加时间
    think_time_seconds: float = 1.0  # 用户思考时间
    timeout_seconds: float = 30.0  # 请求超时时间
    fail_fast: bool = False  # 遇到错误是否快速失败


@dataclass
class RequestResult:
    """请求结果"""
    request_id: int
    user_id: int
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Any] = None


@dataclass
class LoadTestResult:
    """负载测试结果"""
    config: LoadTestConfig
    start_time: float
    end_time: float
    total_duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    errors: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "config": {
                "concurrent_users": self.config.concurrent_users,
                "total_requests": self.config.total_requests,
                "ramp_up_seconds": self.config.ramp_up_seconds,
                "think_time_seconds": self.config.think_time_seconds,
                "timeout_seconds": self.config.timeout_seconds
            },
            "results": {
                "total_duration_seconds": round(self.total_duration_seconds, 2),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": round(self.successful_requests / self.total_requests * 100, 2) if self.total_requests > 0 else 0,
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "min_latency_ms": round(self.min_latency_ms, 2),
                "max_latency_ms": round(self.max_latency_ms, 2),
                "p50_latency_ms": round(self.p50_latency_ms, 2),
                "p95_latency_ms": round(self.p95_latency_ms, 2),
                "p99_latency_ms": round(self.p99_latency_ms, 2),
                "requests_per_second": round(self.requests_per_second, 2)
            },
            "errors": self.errors[:10]  # 只保留前10个错误
        }


class LoadTester:
    """负载测试器"""
    
    def __init__(self):
        self.results: List[RequestResult] = []
        self.results_lock = threading.Lock()
        self.request_counter = 0
        self.counter_lock = threading.Lock()
        self.stop_flag = threading.Event()
    
    def run_test(
        self,
        test_func: Callable[[int, int], Any],
        config: LoadTestConfig
    ) -> LoadTestResult:
        """
        运行负载测试
        
        Args:
            test_func: 测试函数，接收 (request_id, user_id)，返回响应数据或抛出异常
            config: 测试配置
        """
        self.results = []
        self.request_counter = 0
        self.stop_flag.clear()
        
        start_time = time.time()
        
        logger.info(f"🚀 开始负载测试:")
        logger.info(f"   - 并发用户: {config.concurrent_users}")
        logger.info(f"   - 总请求数: {config.total_requests}")
        logger.info(f"   - 爬升时间: {config.ramp_up_seconds}s")
        
        def user_worker(user_id: int):
            """模拟用户行为"""
            # 计算用户启动延迟（爬升）
            if config.ramp_up_seconds > 0:
                delay = (config.ramp_up_seconds / config.concurrent_users) * user_id
                time.sleep(delay)
            
            while not self.stop_flag.is_set():
                # 获取请求ID
                with self.counter_lock:
                    if self.request_counter >= config.total_requests:
                        break
                    self.request_counter += 1
                    request_id = self.request_counter
                
                # 执行请求
                request_start = time.time()
                success = True
                error_msg = None
                response_data = None
                
                try:
                    with performance_tracker.track_latency("load_test_request"):
                        response_data = test_func(request_id, user_id)
                
                except Exception as e:
                    success = False
                    error_msg = str(e)
                    
                    if config.fail_fast:
                        self.stop_flag.set()
                
                request_end = time.time()
                
                result = RequestResult(
                    request_id=request_id,
                    user_id=user_id,
                    start_time=request_start,
                    end_time=request_end,
                    duration_ms=(request_end - request_start) * 1000,
                    success=success,
                    error_message=error_msg,
                    response_data=response_data
                )
                
                with self.results_lock:
                    self.results.append(result)
                
                # 思考时间
                if config.think_time_seconds > 0:
                    time.sleep(config.think_time_seconds * (0.5 + random.random()))
        
        # 创建并启动线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = [
                executor.submit(user_worker, user_id)
                for user_id in range(config.concurrent_users)
            ]
            
            # 等待所有完成
            concurrent.futures.wait(futures)
        
        end_time = time.time()
        
        # 分析结果
        return self._analyze_results(config, start_time, end_time)
    
    def _analyze_results(
        self,
        config: LoadTestConfig,
        start_time: float,
        end_time: float
    ) -> LoadTestResult:
        """分析测试结果"""
        total_duration = end_time - start_time
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        latencies = sorted([r.duration_ms for r in self.results])
        
        def percentile(p: float) -> float:
            if not latencies:
                return 0
            idx = int(len(latencies) * p / 100)
            idx = min(idx, len(latencies) - 1)
            return latencies[idx]
        
        errors = [r.error_message for r in failed if r.error_message]
        
        result = LoadTestResult(
            config=config,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=total_duration,
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p50_latency_ms=percentile(50),
            p95_latency_ms=percentile(95),
            p99_latency_ms=percentile(99),
            requests_per_second=len(self.results) / total_duration if total_duration > 0 else 0,
            errors=errors
        )
        
        # 记录指标
        metrics_collector.record(
            metric_type="load_test",
            metric_name="requests_per_second",
            value=result.requests_per_second,
            unit="rps"
        )
        
        metrics_collector.record(
            metric_type="load_test",
            metric_name="success_rate",
            value=result.successful_requests / result.total_requests * 100 if result.total_requests > 0 else 0,
            unit="percent"
        )
        
        return result


class StressTester:
    """压力测试器 - 测试系统极限"""
    
    def __init__(self):
        self.load_tester = LoadTester()
    
    def find_breaking_point(
        self,
        test_func: Callable[[int, int], Any],
        start_users: int = 10,
        max_users: int = 100,
        step: int = 10,
        requests_per_level: int = 50,
        success_threshold: float = 95.0,
        latency_threshold_ms: float = 5000.0
    ) -> Dict[str, Any]:
        """
        找出系统崩溃点
        
        Args:
            test_func: 测试函数
            start_users: 起始并发用户数
            max_users: 最大并发用户数
            step: 每轮增加的用户数
            requests_per_level: 每轮请求数
            success_threshold: 成功率阈值（百分比）
            latency_threshold_ms: 延迟阈值（毫秒）
        
        Returns:
            测试结果，包含崩溃点信息
        """
        results = []
        breaking_point = None
        
        logger.info(f"🔍 开始压力测试，寻找崩溃点...")
        
        for users in range(start_users, max_users + 1, step):
            config = LoadTestConfig(
                concurrent_users=users,
                total_requests=requests_per_level,
                ramp_up_seconds=5.0,
                think_time_seconds=0.5
            )
            
            result = self.load_tester.run_test(test_func, config)
            results.append(result)
            
            success_rate = (result.successful_requests / result.total_requests * 100 
                          if result.total_requests > 0 else 0)
            
            logger.info(
                f"📊 并发 {users} 用户: "
                f"成功率={success_rate:.1f}%, "
                f"平均延迟={result.avg_latency_ms:.0f}ms, "
                f"P95={result.p95_latency_ms:.0f}ms, "
                f"RPS={result.requests_per_second:.1f}"
            )
            
            # 检查是否达到崩溃点
            if success_rate < success_threshold or result.p95_latency_ms > latency_threshold_ms:
                breaking_point = {
                    "concurrent_users": users,
                    "success_rate": success_rate,
                    "avg_latency_ms": result.avg_latency_ms,
                    "p95_latency_ms": result.p95_latency_ms,
                    "reason": "success_rate" if success_rate < success_threshold else "latency"
                }
                logger.warning(f"⚠️ 检测到崩溃点: {users} 并发用户")
                break
        
        return {
            "breaking_point": breaking_point,
            "all_results": [r.to_dict() for r in results],
            "max_sustainable_users": results[-2].config.concurrent_users if len(results) > 1 and breaking_point else None
        }
    
    def endurance_test(
        self,
        test_func: Callable[[int, int], Any],
        duration_seconds: int = 300,
        concurrent_users: int = 20,
        requests_per_user: int = 10
    ) -> Dict[str, Any]:
        """
        耐久性测试
        
        Args:
            test_func: 测试函数
            duration_seconds: 测试持续时间
            concurrent_users: 并发用户数
            requests_per_user: 每个用户的请求数
        """
        config = LoadTestConfig(
            concurrent_users=concurrent_users,
            total_requests=concurrent_users * requests_per_user,
            ramp_up_seconds=30.0,
            think_time_seconds=2.0
        )
        
        logger.info(f"⏱️ 开始耐久性测试，持续 {duration_seconds} 秒...")
        
        start_time = time.time()
        all_results = []
        
        while time.time() - start_time < duration_seconds:
            result = self.load_tester.run_test(test_func, config)
            all_results.append(result)
            
            elapsed = time.time() - start_time
            logger.info(
                f"⏱️ 已运行 {elapsed:.0f}s / {duration_seconds}s | "
                f"成功率={result.successful_requests / result.total_requests * 100:.1f}%"
            )
        
        # 汇总结果
        total_requests = sum(r.total_requests for r in all_results)
        total_success = sum(r.successful_requests for r in all_results)
        
        return {
            "duration_seconds": duration_seconds,
            "total_requests": total_requests,
            "successful_requests": total_success,
            "overall_success_rate": total_success / total_requests * 100 if total_requests > 0 else 0,
            "rounds_completed": len(all_results),
            "round_results": [r.to_dict() for r in all_results]
        }


def run_load_test_on_rag(
    query_func: Callable[[str], str],
    queries: List[str],
    config: Optional[LoadTestConfig] = None
) -> LoadTestResult:
    """
    对RAG系统进行负载测试
    
    Args:
        query_func: 查询函数
        queries: 测试查询列表
        config: 测试配置
    """
    if config is None:
        config = LoadTestConfig(
            concurrent_users=10,
            total_requests=100,
            ramp_up_seconds=5.0,
            think_time_seconds=1.0
        )
    
    def test_request(request_id: int, user_id: int) -> str:
        query = random.choice(queries)
        return query_func(query)
    
    tester = LoadTester()
    return tester.run_test(test_request, config)


def export_load_test_report(result: LoadTestResult, output_path: str):
    """导出负载测试报告"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "load_test_result": result.to_dict()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📄 负载测试报告已导出: {output_path}")


# 示例用法
if __name__ == "__main__":
    # 示例测试函数
    def sample_query(request_id: int, user_id: int) -> str:
        # 模拟查询延迟
        time.sleep(random.uniform(0.1, 0.5))
        
        # 模拟偶尔的失败
        if random.random() < 0.05:
            raise Exception("模拟的错误")
        
        return f"响应 {request_id}"
    
    # 运行测试
    config = LoadTestConfig(
        concurrent_users=5,
        total_requests=20,
        ramp_up_seconds=2.0,
        think_time_seconds=0.5
    )
    
    tester = LoadTester()
    result = tester.run_test(sample_query, config)
    
    print("\n" + "="*50)
    print("负载测试结果:")
    print("="*50)
    print(f"总请求数: {result.total_requests}")
    print(f"成功请求: {result.successful_requests}")
    print(f"失败请求: {result.failed_requests}")
    print(f"平均延迟: {result.avg_latency_ms:.2f}ms")
    print(f"P95延迟: {result.p95_latency_ms:.2f}ms")
    print(f"RPS: {result.requests_per_second:.2f}")