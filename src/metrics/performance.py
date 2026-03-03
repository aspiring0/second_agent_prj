# src/metrics/performance.py
"""
性能追踪器 - Performance Tracker
追踪响应延迟、吞吐量、并发等性能指标
"""
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import queue

from src.metrics.collector import metrics_collector
from src.utils.logger import setup_logger

logger = setup_logger("PERFORMANCE")


@dataclass
class LatencyRecord:
    """延迟记录"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class ThroughputWindow:
    """吞吐量时间窗口"""
    window_start: float
    window_end: float
    request_count: int
    success_count: int
    error_count: int
    
    @property
    def requests_per_second(self) -> float:
        duration = self.window_end - self.window_start
        if duration <= 0:
            return 0
        return self.request_count / duration


class PerformanceTracker:
    """性能追踪器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        
        # 延迟追踪
        self._latency_records: List[LatencyRecord] = []
        self._latency_lock = threading.Lock()
        
        # 吞吐量追踪
        self._request_queue: queue.Queue = queue.Queue()
        self._throughput_windows: List[ThroughputWindow] = []
        self._throughput_lock = threading.Lock()
        
        # 并发追踪
        self._concurrent_count = 0
        self._concurrent_max = 0
        self._concurrent_lock = threading.Lock()
        
        # P99/P95/P50 计算
        self._percentile_window: List[float] = []
        self._percentile_lock = threading.Lock()
        
        logger.info("✅ 性能追踪器初始化完成")
    
    @contextmanager
    def track_latency(self, operation: str):
        """追踪操作延迟的上下文管理器"""
        start_time = time.time()
        success = True
        error = None
        
        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            record = LatencyRecord(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error=error
            )
            
            with self._latency_lock:
                self._latency_records.append(record)
                # 保留最近1000条记录
                if len(self._latency_records) > 1000:
                    self._latency_records = self._latency_records[-1000:]
            
            # 记录到指标收集器
            metrics_collector.record(
                metric_type="performance",
                metric_name=f"{operation}_latency",
                value=duration_ms,
                unit="ms",
                tags={"success": str(success)},
                metadata={"error": error} if error else {}
            )
            
            # 更新百分位窗口
            with self._percentile_lock:
                self._percentile_window.append(duration_ms)
                if len(self._percentile_window) > 1000:
                    self._percentile_window = self._percentile_window[-1000:]
    
    def track_request(self, success: bool = True):
        """追踪请求"""
        timestamp = time.time()
        
        self._request_queue.put((timestamp, success))
        
        # 记录吞吐量指标
        metrics_collector.record(
            metric_type="performance",
            metric_name="request_count",
            value=1,
            unit="count",
            tags={"success": str(success)}
        )
    
    @contextmanager
    def track_concurrent(self):
        """追踪并发数的上下文管理器"""
        with self._concurrent_lock:
            self._concurrent_count += 1
            if self._concurrent_count > self._concurrent_max:
                self._concurrent_max = self._concurrent_count
        
        try:
            yield
        finally:
            with self._concurrent_lock:
                self._concurrent_count -= 1
            
            # 记录并发指标
            metrics_collector.record(
                metric_type="performance",
                metric_name="concurrent_requests",
                value=self._concurrent_count,
                unit="count"
            )
    
    def calculate_throughput(self, window_seconds: int = 60) -> Dict[str, float]:
        """计算吞吐量"""
        now = time.time()
        window_start = now - window_seconds
        
        request_count = 0
        success_count = 0
        error_count = 0
        
        # 清理过期请求并统计
        temp_queue = queue.Queue()
        
        while not self._request_queue.empty():
            try:
                timestamp, success = self._request_queue.get_nowait()
                if timestamp >= window_start:
                    temp_queue.put((timestamp, success))
                    request_count += 1
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
            except queue.Empty:
                break
        
        # 放回队列
        while not temp_queue.empty():
            self._request_queue.put(temp_queue.get())
        
        rps = request_count / window_seconds if window_seconds > 0 else 0
        
        return {
            "window_seconds": window_seconds,
            "total_requests": request_count,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "requests_per_second": rps,
            "success_rate": success_count / request_count if request_count > 0 else 0
        }
    
    def calculate_percentiles(self) -> Dict[str, float]:
        """计算延迟百分位数"""
        with self._percentile_lock:
            if not self._percentile_window:
                return {}
            
            sorted_latencies = sorted(self._percentile_window)
            count = len(sorted_latencies)
            
            def percentile(p: float) -> float:
                idx = int(count * p / 100)
                idx = min(idx, count - 1)
                return sorted_latencies[idx]
            
            return {
                "p50": percentile(50),
                "p75": percentile(75),
                "p90": percentile(90),
                "p95": percentile(95),
                "p99": percentile(99),
                "min": sorted_latencies[0],
                "max": sorted_latencies[-1],
                "avg": sum(sorted_latencies) / count,
                "sample_count": count
            }
    
    def get_latency_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """获取延迟统计"""
        with self._latency_lock:
            records = self._latency_records
            
            if operation:
                records = [r for r in records if r.operation == operation]
            
            if not records:
                return {}
            
            durations = [r.duration_ms for r in records]
            success_count = sum(1 for r in records if r.success)
            
            return {
                "operation": operation or "all",
                "total_requests": len(records),
                "successful_requests": success_count,
                "failed_requests": len(records) - success_count,
                "success_rate": success_count / len(records),
                "avg_latency_ms": sum(durations) / len(durations),
                "min_latency_ms": min(durations),
                "max_latency_ms": max(durations)
            }
    
    def get_concurrent_stats(self) -> Dict[str, int]:
        """获取并发统计"""
        with self._concurrent_lock:
            return {
                "current_concurrent": self._concurrent_count,
                "max_concurrent": self._concurrent_max
            }
    
    def get_full_report(self) -> Dict[str, Any]:
        """获取完整性能报告"""
        return {
            "timestamp": datetime.now().isoformat(),
            "latency_stats": self.get_latency_stats(),
            "latency_percentiles": self.calculate_percentiles(),
            "throughput": self.calculate_throughput(),
            "concurrent": self.get_concurrent_stats()
        }
    
    def reset(self):
        """重置统计"""
        with self._latency_lock:
            self._latency_records.clear()
        
        with self._throughput_lock:
            self._throughput_windows.clear()
        
        with self._concurrent_lock:
            self._concurrent_count = 0
            self._concurrent_max = 0
        
        with self._percentile_lock:
            self._percentile_window.clear()
        
        # 清空请求队列
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("🔄 性能追踪器已重置")


def track_performance(operation: str):
    """性能追踪装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = PerformanceTracker()
            
            with tracker.track_concurrent():
                tracker.track_request(success=False)  # 先记录请求
                
                with tracker.track_latency(operation) as tracker_ctx:
                    try:
                        result = func(*args, **kwargs)
                        tracker.track_request(success=True)
                        return result
                    except Exception as e:
                        tracker.track_request(success=False)
                        raise
        
        return wrapper
    return decorator


# 全局单例
performance_tracker = PerformanceTracker()