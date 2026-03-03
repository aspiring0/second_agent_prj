# tests/test_file_type_splitting.py
"""
多文件类型切分测试 - File Type Splitting Test
测试不同文件类型（txt, md, py, java, c, js, go, rs）的切分效果
使用 tests/test_files/ 目录下的真实文件进行测试
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.etl import (
    CHINESE_SEPARATORS, 
    MARKDOWN_SEPARATORS,
    PYTHON_SEPARATORS,
    JAVA_SEPARATORS,
    JS_SEPARATORS,
    GO_SEPARATORS,
    C_SEPARATORS,
    RUST_SEPARATORS
)
from src.utils.logger import setup_logger

logger = setup_logger("FILE_TYPE_TEST")

# 测试文件目录
TEST_FILES_DIR = Path(__file__).parent / "test_files"


# ==================== 测试配置 ====================

@dataclass
class SplitTestResult:
    """切分测试结果"""
    file_type: str
    file_name: str
    original_length: int
    chunk_count: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    issues: List[str]
    quality_score: float  # 0-1


# 文件类型到分隔符的映射
FILE_TYPE_CONFIG = {
    ".py": {"separators": PYTHON_SEPARATORS, "name": "Python", "is_code": True},
    ".java": {"separators": JAVA_SEPARATORS, "name": "Java", "is_code": True},
    ".c": {"separators": C_SEPARATORS, "name": "C", "is_code": True},
    ".js": {"separators": JS_SEPARATORS, "name": "JavaScript", "is_code": True},
    ".go": {"separators": GO_SEPARATORS, "name": "Go", "is_code": True},
    ".rs": {"separators": RUST_SEPARATORS, "name": "Rust", "is_code": True},
    ".md": {"separators": MARKDOWN_SEPARATORS, "name": "Markdown", "is_code": False},
    ".txt": {"separators": CHINESE_SEPARATORS, "name": "Text", "is_code": False},
}


class FileTypeSplittingTester:
    """多文件类型切分测试器 - 使用真实文件"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.results: Dict[str, SplitTestResult] = {}
    
    def run_all_tests(self) -> Dict[str, SplitTestResult]:
        """运行所有文件类型的切分测试"""
        print("\n" + "="*70)
        print("🔬 多文件类型切分测试 (真实文件)")
        print(f"📁 测试文件目录: {TEST_FILES_DIR}")
        print("="*70)
        
        if not TEST_FILES_DIR.exists():
            print(f"❌ 测试文件目录不存在: {TEST_FILES_DIR}")
            return {}
        
        # 遍历测试文件目录
        test_files = list(TEST_FILES_DIR.glob("*"))
        test_files = [f for f in test_files if f.is_file() and not f.name.startswith(".")]
        
        if not test_files:
            print(f"❌ 未找到测试文件")
            return {}
        
        print(f"\n📋 发现 {len(test_files)} 个测试文件:")
        for f in test_files:
            print(f"   - {f.name}")
        
        # 逐个测试
        test_num = 1
        for file_path in sorted(test_files):
            suffix = file_path.suffix.lower()
            
            if suffix not in FILE_TYPE_CONFIG:
                print(f"\n⏭️ 跳过不支持的文件类型: {file_path.name}")
                continue
            
            config = FILE_TYPE_CONFIG[suffix]
            print(f"\n📄 测试 {test_num}: {config['name']}代码 (.{suffix})")
            print("-"*70)
            
            result = self._test_file(file_path, config)
            self.results[file_path.stem] = result
            test_num += 1
        
        # 打印汇总
        self._print_summary()
        
        return self.results
    
    def _test_file(self, file_path: Path, config: dict) -> SplitTestResult:
        """测试单个文件"""
        try:
            # 加载文档
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            
            if not docs:
                return SplitTestResult(
                    file_type=config['name'],
                    file_name=file_path.name,
                    original_length=0,
                    chunk_count=0,
                    avg_chunk_size=0,
                    min_chunk_size=0,
                    max_chunk_size=0,
                    issues=["文档加载失败"],
                    quality_score=0
                )
            
            original_content = docs[0].page_content
            original_length = len(original_content)
            
            # 切分
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=config['separators']
            )
            chunks = splitter.split_documents(docs)
            
            # 分析结果
            return self._analyze_chunks(
                file_type=config['name'],
                file_name=file_path.name,
                chunks=chunks,
                original=original_content,
                is_code=config['is_code']
            )
            
        except Exception as e:
            logger.error(f"测试文件失败 {file_path}: {e}")
            return SplitTestResult(
                file_type=config['name'],
                file_name=file_path.name,
                original_length=0,
                chunk_count=0,
                avg_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                issues=[f"测试失败: {str(e)}"],
                quality_score=0
            )
    
    def _analyze_chunks(self, file_type: str, file_name: str, chunks: List, original: str, is_code: bool = False) -> SplitTestResult:
        """分析切分结果"""
        issues = []
        
        if not chunks:
            return SplitTestResult(
                file_type=file_type,
                file_name=file_name,
                original_length=len(original),
                chunk_count=0,
                avg_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                issues=["切分结果为空"],
                quality_score=0
            )
        
        # 计算统计信息
        sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        # 检测问题
        # 1. 过小的片段（可能丢失上下文）
        tiny_chunks = sum(1 for s in sizes if s < 100)
        if tiny_chunks > 0:
            issues.append(f"存在 {tiny_chunks} 个过小片段(<100字符)")
        
        # 2. 过大的片段（可能超出上下文窗口）
        huge_chunks = sum(1 for s in sizes if s > self.chunk_size * 1.2)
        if huge_chunks > 0:
            issues.append(f"存在 {huge_chunks} 个过大片段(>{int(self.chunk_size * 1.2)}字符)")
        
        # 3. 代码特定检查
        if is_code:
            for i, chunk in enumerate(chunks):
                content = chunk.page_content
                
                # Python特定检查
                if file_type == "Python":
                    if 'def ' in content:
                        def_line = content.split('def ')[-1].split('\n')[0]
                        if '(' not in def_line:
                            issues.append(f"片段{i+1}函数定义不完整")
                
                # Java/JS/Go/C/Rust 需要{}
                elif file_type in ["Java", "JavaScript", "Go", "C", "Rust"]:
                    if 'class ' in content and '{' not in content:
                        issues.append(f"片段{i+1}类定义不完整（缺少'{{'）")
        
        # 4. 计算质量分数
        quality_score = 1.0
        quality_score -= 0.1 * tiny_chunks  # 小片段扣分
        quality_score -= 0.15 * huge_chunks  # 大片段扣分
        quality_score -= 0.05 * len([i for i in issues if '不完整' in i])  # 不完整扣分
        quality_score = max(0, min(1, quality_score))
        
        # 打印结果
        print(f"   文件名: {file_name}")
        print(f"   原始长度: {len(original)} 字符")
        print(f"   切分数量: {len(chunks)} 个片段")
        print(f"   平均大小: {avg_size:.1f} 字符")
        print(f"   大小范围: {min_size} ~ {max_size} 字符")
        print(f"   质量分数: {quality_score:.2f}")
        
        if issues:
            print(f"   ⚠️ 发现问题 ({len(issues)}个):")
            for issue in issues[:5]:
                print(f"      - {issue}")
            if len(issues) > 5:
                print(f"      ... 还有 {len(issues)-5} 个问题")
        else:
            print(f"   ✅ 未发现明显问题")
        
        # 显示前3个片段预览
        print(f"\n   📋 片段预览:")
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk.page_content[:120].replace('\n', ' ')
            print(f"      [{i+1}] {preview}...")
        
        return SplitTestResult(
            file_type=file_type,
            file_name=file_name,
            original_length=len(original),
            chunk_count=len(chunks),
            avg_chunk_size=round(avg_size, 1),
            min_chunk_size=min_size,
            max_chunk_size=max_size,
            issues=issues,
            quality_score=round(quality_score, 2)
        )
    
    def _print_summary(self):
        """打印汇总报告"""
        print("\n" + "="*70)
        print("📊 切分测试汇总")
        print("="*70)
        
        print(f"\n{'文件类型':<12} {'文件名':<15} {'片段数':<8} {'平均大小':<10} {'质量分':<8}")
        print("-"*70)
        
        for name, result in self.results.items():
            print(f"{result.file_type:<12} {result.file_name:<15} {result.chunk_count:<8} {result.avg_chunk_size:<10.1f} {result.quality_score:<8.2f}")
        
        if not self.results:
            return
        
        # 识别问题最多的类型
        worst = min(self.results.items(), key=lambda x: x[1].quality_score)
        best = max(self.results.items(), key=lambda x: x[1].quality_score)
        
        print(f"\n🏆 最佳切分: {best[1].file_name} (质量分: {best[1].quality_score:.2f})")
        print(f"⚠️ 需要优化: {worst[1].file_name} (质量分: {worst[1].quality_score:.2f})")
        
        # 生成优化建议
        all_issues = []
        for result in self.results.values():
            all_issues.extend(result.issues)
        
        if all_issues:
            print("\n💡 优化建议:")
            if any('过小' in i for i in all_issues):
                print("   1. 存在过小片段 → 考虑增大CHUNK_SIZE或减少CHUNK_OVERLAP")
            if any('过大' in i for i in all_issues):
                print("   2. 存在过大片段 → 优化分隔符顺序或减小CHUNK_SIZE")
            if any('不完整' in i for i in all_issues):
                print("   3. 存在语义断裂 → 针对该文件类型优化分隔符")
            if any('类定义' in i or '函数' in i for i in all_issues):
                print("   4. 代码结构断裂 → 使用AST解析进行智能切分")


def main():
    """主函数"""
    tester = FileTypeSplittingTester(chunk_size=800, chunk_overlap=100)
    results = tester.run_all_tests()
    
    # 保存报告
    report_path = Path("metrics/reports/file_splitting_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    report = {
        name: {
            "file_type": r.file_type,
            "file_name": r.file_name,
            "chunk_count": r.chunk_count,
            "avg_chunk_size": r.avg_chunk_size,
            "quality_score": r.quality_score,
            "issues": r.issues
        }
        for name, r in results.items()
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 报告已保存: {report_path}")
    
    return results


if __name__ == "__main__":
    main()