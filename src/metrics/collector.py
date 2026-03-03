# src/metrics/collector.py
"""
指标收集器 - Metrics Collector
集中管理所有指标数据的收集、存储和查询
"""
import json
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import sqlite3

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("METRICS")


@dataclass
class MetricRecord:
    """指标记录"""
    timestamp: str
    metric_type: str  # performance, quality, retrieval, etc.
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class PerformanceMetric:
    """性能指标"""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: str = ""
    
    def complete(self, success: bool = True, error: str = ""):
        """完成计时"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_message = error


class MetricsCollector:
    """指标收集器"""
    
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
        self.metrics_dir = settings.BASE_DIR / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        # 内存缓存
        self._metrics_buffer: List[MetricRecord] = []
        self._buffer_lock = threading.Lock()
        
        # 性能追踪
        self._active_operations: Dict[str, PerformanceMetric] = {}
        self._operations_lock = threading.Lock()
        
        # 统计聚合
        self._stats: Dict[str, List[float]] = defaultdict(list)
        self._stats_lock = threading.Lock()
        
        # 初始化数据库
        self._init_metrics_db()
        
        logger.info("✅ 指标收集器初始化完成")
    
    def _init_metrics_db(self):
        """初始化指标数据库"""
        db_path = self.metrics_dir / "metrics.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metric_type 
            ON metrics(metric_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metric_name 
            ON metrics(metric_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON metrics(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def record(
        self, 
        metric_type: str,
        metric_name: str, 
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """记录指标"""
        record = MetricRecord(
            timestamp=datetime.now().isoformat(),
            metric_type=metric_type,
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self._buffer_lock:
            self._metrics_buffer.append(record)
            
            # 定期持久化
            if len(self._metrics_buffer) >= 100:
                self._flush_buffer()
        
        with self._stats_lock:
            key = f"{metric_type}.{metric_name}"
            self._stats[key].append(value)
        
        logger.debug(f"📊 记录指标: {key}={value}{unit}")
    
    def start_operation(self, operation: str) -> str:
        """开始操作计时"""
        op_id = f"{operation}_{time.time_ns()}"
        
        with self._operations_lock:
            self._active_operations[op_id] = PerformanceMetric(
                operation=operation,
                start_time=time.time()
            )
        
        return op_id
    
    def end_operation(self, op_id: str, success: bool = True, error: str = "") -> Optional[float]:
        """结束操作计时"""
        with self._operations_lock:
            if op_id not in self._active_operations:
                logger.warning(f"未找到操作: {op_id}")
                return None
            
            metric = self._active_operations.pop(op_id)
            metric.complete(success, error)
            
            # 记录指标
            self.record(
                metric_type="performance",
                metric_name=f"{metric.operation}_latency",
                value=metric.duration_ms,
                unit="ms",
                tags={"success": str(success)},
                metadata={"error": error} if error else {}
            )
            
            return metric.duration_ms
    
    def _flush_buffer(self):
        """将缓冲区数据写入数据库"""
        if not self._metrics_buffer:
            return
            
        db_path = self.metrics_dir / "metrics.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        for record in self._metrics_buffer:
            cursor.execute("""
                INSERT INTO metrics 
                (timestamp, metric_type, metric_name, value, unit, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp,
                record.metric_type,
                record.metric_name,
                record.value,
                record.unit,
                json.dumps(record.tags),
                json.dumps(record.metadata)
            ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"💾 持久化 {len(self._metrics_buffer)} 条指标")
        self._metrics_buffer.clear()
    
    def get_stats(self, metric_key: str) -> Dict[str, float]:
        """获取统计信息"""
        with self._stats_lock:
            values = self._stats.get(metric_key, [])
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1]
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有统计信息"""
        with self._stats_lock:
            result = {}
            for key in self._stats:
                result[key] = self.get_stats(key)
            return result
    
    def query_metrics(
        self,
        metric_type: Optional[str] = None,
        metric_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """查询指标"""
        # 先刷新缓冲区
        with self._buffer_lock:
            self._flush_buffer()
        
        db_path = self.metrics_dir / "metrics.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type)
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        columns = ["id", "timestamp", "metric_type", "metric_name", "value", "unit", "tags", "metadata"]
        return [dict(zip(columns, row)) for row in rows]
    
    def export_report(self, output_path: Optional[str] = None) -> Dict:
        """导出报告"""
        # 刷新缓冲区
        with self._buffer_lock:
            self._flush_buffer()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_all_stats(),
            "summary": self._generate_summary()
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"📄 报告已导出: {output_path}")
        
        return report
    
    def _generate_summary(self) -> Dict:
        """生成摘要"""
        stats = self.get_all_stats()
        summary = {
            "total_metrics": sum(s.get("count", 0) for s in stats.values()),
            "metric_types": {}
        }
        
        for key, stat in stats.items():
            metric_type = key.split(".")[0]
            if metric_type not in summary["metric_types"]:
                summary["metric_types"][metric_type] = {"count": 0, "metrics": []}
            
            summary["metric_types"][metric_type]["count"] += stat.get("count", 0)
            summary["metric_types"][metric_type]["metrics"].append({
                "name": key,
                "mean": stat.get("mean"),
                "min": stat.get("min"),
                "max": stat.get("max")
            })
        
        return summary
    
    def reset(self):
        """重置统计"""
        with self._buffer_lock:
            self._metrics_buffer.clear()
        
        with self._stats_lock:
            self._stats.clear()
        
        with self._operations_lock:
            self._active_operations.clear()
        
        logger.info("🔄 指标收集器已重置")


# 全局单例
metrics_collector = MetricsCollector()