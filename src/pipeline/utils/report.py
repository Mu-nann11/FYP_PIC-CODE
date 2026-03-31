"""
流水线运行报告生成工具
生成汇总统计和详细日志报告
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline.config import BASE_DIR

DEFAULT_REPORT_DIR = BASE_DIR / "results" / "pipeline_reports"


@dataclass
class BlockResult:
    """单个 Block 的处理结果"""
    block: str
    dataset: str
    preprocess_status: str = "pending"
    preprocess_error: Optional[str] = None
    stitch_status: str = "pending"  # pending / skipped / success / error
    stitch_error: Optional[str] = None
    align_status: str = "pending"
    align_error: Optional[str] = None
    align_params: dict = field(default_factory=dict)
    segment_status: str = "pending"
    segment_error: Optional[str] = None
    grading_status: str = "pending"
    grading_error: Optional[str] = None
    vis_align_status: str = "pending"  # Alignment visualization status
    cell_count: int = 0
    nuclei_count: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output_features: Optional[Path] = None

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def overall_status(self) -> str:
        if self.preprocess_status == "error" or self.stitch_status == "error" or self.align_status == "error" or self.segment_status == "error" or self.grading_status == "error":
            return "error"
        if self.preprocess_status == "skipped" and self.stitch_status == "skipped" and self.align_status == "skipped" and self.segment_status == "skipped" and self.grading_status == "skipped":
            return "skipped_all"
        return "success"


@dataclass
class PipelineReport:
    """整条流水线的运行报告"""
    dataset: str
    blocks: list[BlockResult] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.blocks is None:
            self.blocks = []

    def add_block(self, block_result: BlockResult):
        self.blocks.append(block_result)

    def to_text(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append(f"TMA Pipeline Run Report")
        lines.append(f"Dataset: {self.dataset}")
        lines.append(f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # 汇总统计
        total = len(self.blocks)
        success = sum(1 for b in self.blocks if b.overall_status == "success")
        skipped = sum(1 for b in self.blocks if b.overall_status == "skipped_all")
        errors = sum(1 for b in self.blocks if b.overall_status == "error")
        total_cells = sum(b.cell_count for b in self.blocks)

        lines.append("Summary:")
        lines.append(f"  Total blocks : {total}")
        lines.append(f"  Success     : {success}")
        lines.append(f"  Skipped      : {skipped}")
        lines.append(f"  Error        : {errors}")
        lines.append(f"  Total cells  : {total_cells}")
        lines.append("")

        # 详细列表
        lines.append("Block Details:")
        lines.append("-" * 70)
        for b in self.blocks:
            status_icon = {
                "success": "[OK]",
                "error": "[ERR]",
                "skipped_all": "[SKIP]",
            }.get(b.overall_status, "[??]")

            lines.append(f"  {status_icon} {b.block}")
            lines.append(f"      Preproc : {b.preprocess_status}")
            if b.preprocess_error:
                lines.append(f"             Error: {b.preprocess_error[:60]}")
            lines.append(f"      Stitch  : {b.stitch_status}")
            if b.stitch_error:
                lines.append(f"             Error: {b.stitch_error[:60]}")
            lines.append(f"      Align   : {b.align_status}")
            if b.align_error:
                lines.append(f"             Error: {b.align_error[:60]}")
            if b.align_params:
                lines.append(f"             Params: angle={b.align_params.get('angle', 'N/A')}, "
                             f"shift=({b.align_params.get('dy', 'N/A')}, {b.align_params.get('dx', 'N/A')})")
            lines.append(f"      Segment : {b.segment_status}")
            if b.segment_error:
                lines.append(f"             Error: {b.segment_error[:60]}")
            else:
                lines.append(f"             Cells: {b.cell_count}")
            if b.duration_seconds > 0:
                lines.append(f"      Duration: {b.duration_seconds:.1f}s")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def save(self, report_dir: Optional[Path] = None) -> Path:
        if report_dir is None:
            report_dir = DEFAULT_REPORT_DIR
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"report_{self.dataset}_{timestamp}.txt"
        report_file.write_text(self.to_text(), encoding="utf-8")

        # 同时保存 JSON 格式便于程序读取
        json_file = report_dir / f"report_{self.dataset}_{timestamp}.json"
        _save_json(self, json_file)

        return report_file


def _save_json(report: PipelineReport, path: Path):
    """将报告保存为 JSON 格式"""
    import json

    data = {
        "dataset": report.dataset,
        "created_at": report.created_at.isoformat(),
        "summary": {
            "total": len(report.blocks),
            "success": sum(1 for b in report.blocks if b.overall_status == "success"),
            "skipped": sum(1 for b in report.blocks if b.overall_status == "skipped_all"),
            "errors": sum(1 for b in report.blocks if b.overall_status == "error"),
            "total_cells": sum(b.cell_count for b in report.blocks),
        },
        "blocks": [
            {
                "block": b.block,
                "preprocess_status": b.preprocess_status,
                "stitch_status": b.stitch_status,
                "align_status": b.align_status,
                "segment_status": b.segment_status,
                "cell_count": b.cell_count,
                "duration_s": b.duration_seconds,
            }
            for b in report.blocks
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_report(
    dataset: str,
    blocks: list[BlockResult],
    save: bool = True,
    verbose: bool = True,
) -> PipelineReport:
    """
    生成并（可选）保存流水线运行报告

    Args:
        dataset: 数据集名称
        blocks: BlockResult 列表
        save: 是否保存到文件
        verbose: 是否打印到控制台
    """
    report = PipelineReport(dataset=dataset, blocks=blocks)

    if verbose:
        print(report.to_text())

    if save:
        report_file = report.save()
        print(f"\nReport saved to: {report_file}")

    return report
