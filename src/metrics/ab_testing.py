# src/metrics/ab_testing.py
"""
A/B测试框架 - A/B Testing Framework
支持不同配置/模型之间的对比测试
"""
import json
import random
import hashlib
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sqlite3
from enum import Enum

from src.metrics.collector import metrics_collector
from src.utils.logger import setup_logger

logger = setup_logger("AB_TESTING")


class ExperimentStatus(Enum):
    """实验状态"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class Variant:
    """变体配置"""
    name: str
    config: Dict[str, Any]
    weight: float = 0.5  # 流量分配权重
    is_control: bool = False  # 是否为对照组


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    variant_name: str
    metric_name: str
    value: float
    sample_size: int
    timestamp: str


@dataclass
class Experiment:
    """实验定义"""
    id: str
    name: str
    description: str
    variants: List[Variant]
    metrics: List[str]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    traffic_allocation: float = 1.0  # 参与实验的流量比例


class ABTestingFramework:
    """A/B测试框架"""
    
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
        
        # 实验存储
        self._experiments: Dict[str, Experiment] = {}
        
        # 用户分组缓存
        self._user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {experiment_id: variant}
        
        # 结果存储
        self._results: List[ExperimentResult] = []
        
        # 初始化数据库
        self._init_ab_db()
        
        # 加载现有实验
        self._load_experiments()
        
        logger.info("✅ A/B测试框架初始化完成")
    
    def _init_ab_db(self):
        """初始化A/B测试数据库"""
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        db_path = metrics_dir / "ab_testing.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 实验表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                variants TEXT,
                metrics TEXT,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                traffic_allocation REAL,
                created_at TEXT
            )
        """)
        
        # 用户分组表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                assigned_at TEXT,
                UNIQUE(user_id, experiment_id)
            )
        """)
        
        # 实验结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL,
                sample_size INTEGER,
                user_id TEXT,
                recorded_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiment_results 
            ON experiment_results(experiment_id, variant_name)
        """)
        
        conn.commit()
        conn.close()
    
    def _load_experiments(self):
        """从数据库加载实验"""
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        db_path = metrics_dir / "ab_testing.db"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, description, variants, metrics, status, 
                   start_time, end_time, traffic_allocation
            FROM experiments
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            variants_data = json.loads(row[3]) if row[3] else []
            variants = [
                Variant(
                    name=v["name"],
                    config=v["config"],
                    weight=v.get("weight", 0.5),
                    is_control=v.get("is_control", False)
                )
                for v in variants_data
            ]
            
            experiment = Experiment(
                id=row[0],
                name=row[1],
                description=row[2] or "",
                variants=variants,
                metrics=json.loads(row[4]) if row[4] else [],
                status=ExperimentStatus(row[5]) if row[5] else ExperimentStatus.DRAFT,
                start_time=row[6],
                end_time=row[7],
                traffic_allocation=row[8] or 1.0
            )
            
            self._experiments[row[0]] = experiment
        
        logger.info(f"📂 加载了 {len(self._experiments)} 个实验")
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        metrics: List[str],
        traffic_allocation: float = 1.0
    ) -> Experiment:
        """
        创建实验
        
        Args:
            experiment_id: 实验ID
            name: 实验名称
            description: 实验描述
            variants: 变体列表，如 [
                {"name": "control", "config": {...}, "weight": 0.5, "is_control": True},
                {"name": "treatment", "config": {...}, "weight": 0.5}
            ]
            metrics: 关注的指标列表
            traffic_allocation: 流量分配比例
        """
        variant_objects = [
            Variant(
                name=v["name"],
                config=v.get("config", {}),
                weight=v.get("weight", 0.5),
                is_control=v.get("is_control", False)
            )
            for v in variants
        ]
        
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variant_objects,
            metrics=metrics,
            traffic_allocation=traffic_allocation
        )
        
        self._experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"🧪 创建实验: {name} ({experiment_id})")
        return experiment
    
    def _save_experiment(self, experiment: Experiment):
        """保存实验到数据库"""
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        db_path = metrics_dir / "ab_testing.db"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        variants_data = [
            {
                "name": v.name,
                "config": v.config,
                "weight": v.weight,
                "is_control": v.is_control
            }
            for v in experiment.variants
        ]
        
        cursor.execute("""
            INSERT OR REPLACE INTO experiments 
            (id, name, description, variants, metrics, status, 
             start_time, end_time, traffic_allocation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment.id,
            experiment.name,
            experiment.description,
            json.dumps(variants_data),
            json.dumps(experiment.metrics),
            experiment.status.value,
            experiment.start_time,
            experiment.end_time,
            experiment.traffic_allocation,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def start_experiment(self, experiment_id: str):
        """启动实验"""
        if experiment_id not in self._experiments:
            raise ValueError(f"实验不存在: {experiment_id}")
        
        experiment = self._experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now().isoformat()
        
        self._save_experiment(experiment)
        logger.info(f"▶️ 启动实验: {experiment.name}")
    
    def pause_experiment(self, experiment_id: str):
        """暂停实验"""
        if experiment_id not in self._experiments:
            raise ValueError(f"实验不存在: {experiment_id}")
        
        experiment = self._experiments[experiment_id]
        experiment.status = ExperimentStatus.PAUSED
        self._save_experiment(experiment)
        logger.info(f"⏸️ 暂停实验: {experiment.name}")
    
    def complete_experiment(self, experiment_id: str):
        """结束实验"""
        if experiment_id not in self._experiments:
            raise ValueError(f"实验不存在: {experiment_id}")
        
        experiment = self._experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.now().isoformat()
        
        self._save_experiment(experiment)
        logger.info(f"✅ 结束实验: {experiment.name}")
    
    def assign_variant(self, experiment_id: str, user_id: str) -> Optional[Variant]:
        """
        为用户分配变体
        
        Args:
            experiment_id: 实验ID
            user_id: 用户ID
        
        Returns:
            分配的变体，如果实验未运行则返回None
        """
        if experiment_id not in self._experiments:
            return None
        
        experiment = self._experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.RUNNING:
            # 返回对照组
            for v in experiment.variants:
                if v.is_control:
                    return v
            return None
        
        # 检查流量分配
        if random.random() > experiment.traffic_allocation:
            return None
        
        # 检查是否已分配
        if user_id in self._user_assignments:
            if experiment_id in self._user_assignments[user_id]:
                variant_name = self._user_assignments[user_id][experiment_id]
                for v in experiment.variants:
                    if v.name == variant_name:
                        return v
        
        # 基于用户ID的一致性哈希分配
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        ratio = (hash_value % 10000) / 10000
        
        # 根据权重分配变体
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.weight
            if ratio <= cumulative:
                # 记录分配
                if user_id not in self._user_assignments:
                    self._user_assignments[user_id] = {}
                self._user_assignments[user_id][experiment_id] = variant.name
                
                # 持久化
                self._save_assignment(user_id, experiment_id, variant.name)
                
                return variant
        
        return experiment.variants[0]
    
    def _save_assignment(self, user_id: str, experiment_id: str, variant_name: str):
        """保存用户分配"""
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        db_path = metrics_dir / "ab_testing.db"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO user_assignments 
            (user_id, experiment_id, variant_name, assigned_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, experiment_id, variant_name, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def record_metric(
        self,
        experiment_id: str,
        variant_name: str,
        metric_name: str,
        value: float,
        user_id: Optional[str] = None,
        sample_size: int = 1
    ):
        """记录实验指标"""
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_name=variant_name,
            metric_name=metric_name,
            value=value,
            sample_size=sample_size,
            timestamp=datetime.now().isoformat()
        )
        
        self._results.append(result)
        
        # 持久化
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        db_path = metrics_dir / "ab_testing.db"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO experiment_results 
            (experiment_id, variant_name, metric_name, value, sample_size, user_id, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (experiment_id, variant_name, metric_name, value, sample_size, user_id, result.timestamp))
        
        conn.commit()
        conn.close()
        
        # 记录到指标收集器
        metrics_collector.record(
            metric_type="ab_testing",
            metric_name=f"{experiment_id}.{variant_name}.{metric_name}",
            value=value,
            tags={"experiment": experiment_id, "variant": variant_name}
        )
    
    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        """获取实验结果"""
        from config.settings import settings
        metrics_dir = settings.BASE_DIR / "metrics"
        db_path = metrics_dir / "ab_testing.db"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT variant_name, metric_name, AVG(value) as avg_value, 
                   COUNT(*) as count, SUM(sample_size) as total_samples
            FROM experiment_results
            WHERE experiment_id = ?
            GROUP BY variant_name, metric_name
        """, (experiment_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = {}
        for row in rows:
            variant_name, metric_name, avg_value, count, total_samples = row
            
            if variant_name not in results:
                results[variant_name] = {"metrics": {}, "sample_count": 0}
            
            results[variant_name]["metrics"][metric_name] = {
                "avg_value": avg_value,
                "observations": count,
                "total_samples": total_samples
            }
            results[variant_name]["sample_count"] = max(
                results[variant_name]["sample_count"],
                total_samples or count
            )
        
        return results
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """分析实验结果（包含统计显著性检验）"""
        results = self.get_results(experiment_id)
        
        if not results or len(results) < 2:
            return {
                "experiment_id": experiment_id,
                "status": "insufficient_data",
                "message": "数据不足，无法进行分析"
            }
        
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return {"error": "实验不存在"}
        
        # 找到对照组和实验组
        control_name = None
        treatment_name = None
        
        for v in experiment.variants:
            if v.is_control:
                control_name = v.name
            else:
                treatment_name = v.name
        
        if not control_name or not treatment_name:
            return {"error": "未找到对照组或实验组"}
        
        analysis = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "control": control_name,
            "treatment": treatment_name,
            "metrics_comparison": {},
            "recommendation": None
        }
        
        # 比较每个指标
        control_metrics = results.get(control_name, {}).get("metrics", {})
        treatment_metrics = results.get(treatment_name, {}).get("metrics", {})
        
        for metric_name in experiment.metrics:
            control_val = control_metrics.get(metric_name, {}).get("avg_value", 0)
            treatment_val = treatment_metrics.get(metric_name, {}).get("avg_value", 0)
            
            if control_val > 0:
                change_pct = (treatment_val - control_val) / control_val * 100
            else:
                change_pct = 0
            
            # 简单的统计显著性判断（实际应使用t检验或卡方检验）
            control_samples = results.get(control_name, {}).get("sample_count", 0)
            treatment_samples = results.get(treatment_name, {}).get("sample_count", 0)
            
            # 简化：样本量大于100且变化大于5%认为有显著性
            is_significant = (
                control_samples >= 100 and 
                treatment_samples >= 100 and 
                abs(change_pct) >= 5
            )
            
            analysis["metrics_comparison"][metric_name] = {
                "control_value": control_val,
                "treatment_value": treatment_val,
                "change_percent": change_pct,
                "is_significant": is_significant,
                "control_samples": control_samples,
                "treatment_samples": treatment_samples
            }
        
        # 生成推荐
        positive_metrics = sum(
            1 for m in analysis["metrics_comparison"].values() 
            if m["change_percent"] > 0 and m["is_significant"]
        )
        negative_metrics = sum(
            1 for m in analysis["metrics_comparison"].values() 
            if m["change_percent"] < 0 and m["is_significant"]
        )
        
        if positive_metrics > negative_metrics:
            analysis["recommendation"] = "adopt_treatment"
            analysis["recommendation_reason"] = f"实验组在 {positive_metrics} 个关键指标上表现更好"
        elif negative_metrics > positive_metrics:
            analysis["recommendation"] = "keep_control"
            analysis["recommendation_reason"] = f"对照组在 {negative_metrics} 个关键指标上表现更好"
        else:
            analysis["recommendation"] = "continue_testing"
            analysis["recommendation_reason"] = "结果不明确，建议继续收集数据"
        
        return analysis
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """获取实验"""
        return self._experiments.get(experiment_id)
    
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Experiment]:
        """列出实验"""
        experiments = list(self._experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        return experiments
    
    def export_report(self, experiment_id: str, output_path: str) -> Dict:
        """导出实验报告"""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return {"error": "实验不存在"}
        
        report = {
            "experiment": {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "status": experiment.status.value,
                "start_time": experiment.start_time,
                "end_time": experiment.end_time
            },
            "variants": [
                {
                    "name": v.name,
                    "config": v.config,
                    "weight": v.weight,
                    "is_control": v.is_control
                }
                for v in experiment.variants
            ],
            "results": self.get_results(experiment_id),
            "analysis": self.analyze_experiment(experiment_id),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 实验报告已导出: {output_path}")
        return report


# 全局单例
ab_testing = ABTestingFramework()