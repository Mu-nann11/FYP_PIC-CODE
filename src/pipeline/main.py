"""
TMA image analysis pipeline - main entry point.

Usage:
    python -m pipeline.main --dataset TMAd --blocks G2 A2 B8
    python -m pipeline.main --dataset TMAd --all-blocks
    python -m pipeline.main --dataset TMAd --blocks G2 --from-step 2   # skip stitching
    python -m pipeline.main --dataset TMAd --blocks G2 --from-step 3   # skip stitching + alignment
    python -m pipeline.main --dataset TMAd --blocks G2 --from-step 4   # only grading/subtyping
    python -m pipeline.main --dataset TMAd --blocks G2 --force          # force rerun
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from .config import (
        BASE_DIR,
        DATASETS,
        discover_blocks,
    )
    from .steps import (
        run_preprocess,
        check_preprocess_done,
        run_stitching,
        check_stitch_done,
        run_alignment,
        check_alignment_done,
        run_segmentation,
        check_segmentation_done,
        run_grading_and_subtyping,
        check_grading_done,
        run_visualization_alignment,
        run_visualization_summary,
    )
    from .utils.logging import setup_logging, get_logger
    from .utils.report import BlockResult, generate_report
except ImportError:
    # When run directly (python main.py or python -m pipeline.main), use absolute imports
    _parent = str(Path(__file__).resolve().parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    from config import (
        BASE_DIR,
        DATASETS,
        discover_blocks,
    )
    from steps import (
        run_stitching,
        check_stitch_done,
        run_alignment,
        check_alignment_done,
        run_segmentation,
        check_segmentation_done,
        run_grading_and_subtyping,
        check_grading_done,
        run_visualization_alignment,
        run_visualization_summary,
    )
    from utils.logging import setup_logging, get_logger
    from utils.report import BlockResult, generate_report


# =====================================================================
# Command-line arguments
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TMA image analysis pipeline: stitching -> alignment -> segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        default="TMAd",
        choices=list(DATASETS.keys()),
        help="Dataset name (default: TMAd)",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        help="Blocks to process, e.g. G2 A2 B8",
    )
    parser.add_argument(
        "--all-blocks",
        action="store_true",
        help="Automatically discover and process all blocks for this dataset",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6],
        default=1,
        help=(
            "Start execution from step: "
            "0=preprocessing (tile-to-channel), 1=stitching (default), 2=alignment (skip stitching), 3=segmentation (skip stitching and alignment), "
            "4=grading + subtype classification, 5=alignment visualization, 6=summary visualization"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run (ignore resume/checkpoint checks)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (log only to file, do not print to console)",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        default=True,
        help="Save run report (default: True)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log output directory (default: results/pipeline_logs)",
    )

    return parser


# =====================================================================
# Pipeline execution
# =====================================================================

def run_pipeline(
    dataset: str,
    blocks: list[str],
    from_step: int = 1,
    force: bool = False,
) -> list[BlockResult]:
    """
    Run the full pipeline for the specified blocks.

    Args:
        dataset: Dataset name.
        blocks: Block list.
        from_step: Starting step (1=stitching, 2=alignment, 3=segmentation).
        force: Whether to force a re-run.

    Returns:
        list[BlockResult]: Results for each block.
    """
    results: list[BlockResult] = []
    total = len(blocks)

    for idx, block in enumerate(blocks, 1):
        logger = get_logger("pipeline")
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing block {idx}/{total}: {block} ({dataset})")
        logger.info(f"{'='*60}")

        block_result = BlockResult(block=block, dataset=dataset)
        block_result.start_time = datetime.now()

        # ----- Step 0: Preprocessing -----
        if from_step <= 0:
            logger.info(f"[Step 0] Preprocessing ...")
            preprocess_res = run_preprocess(block, dataset, force=force)
            block_result.preprocess_status = preprocess_res["status"]
            block_result.preprocess_error = preprocess_res.get("error")
            if preprocess_res["status"] == "error":
                logger.error(f"[Step 0] Preprocessing failed: {preprocess_res.get('error')}")
                block_result.end_time = datetime.now()
                results.append(block_result)
                continue
        else:
            skipped = check_preprocess_done(block, dataset)
            block_result.preprocess_status = "skipped"
            if not skipped:
                logger.warning(f"[Step 0] Preprocessing output might not be ready, but skipping as requested")

        # ----- Step 1: Stitching -----
        if from_step <= 1:
            logger.info(f"[Step 1] Stitching ...")
            stitch_res = run_stitching(block, dataset, force=force)
            block_result.stitch_status = stitch_res["status"]
            block_result.stitch_error = stitch_res.get("error")
            if stitch_res["status"] == "error":
                logger.error(f"[Step 1] Stitching failed: {stitch_res.get('error')}")
                block_result.end_time = datetime.now()
                results.append(block_result)
                continue  # Skip the remaining steps if stitching fails
        else:
            skipped = check_stitch_done(block, dataset)
            block_result.stitch_status = "skipped"
            logger.info(f"[Step 1] Stitching skipped (--from-step=2)")
            if not skipped:
                logger.warning(f"[Step 1] Stitching output not found, but skipping as requested")

        # ----- Step 2: Alignment -----
        if from_step <= 2:
            logger.info(f"[Step 2] Alignment ...")
            align_res = run_alignment(block, dataset, force=force)
            block_result.align_status = align_res["status"]
            block_result.align_error = align_res.get("error")
            block_result.align_params = align_res.get("align_params", {})
            if align_res["status"] == "error":
                logger.error(f"[Step 2] Alignment failed: {align_res.get('error')}")
                block_result.end_time = datetime.now()
                results.append(block_result)
                continue  # Skip segmentation if alignment fails
        else:
            skipped = check_alignment_done(block, dataset)
            block_result.align_status = "skipped"
            logger.info(f"[Step 2] Alignment skipped (--from-step=3)")
            if not skipped:
                logger.warning(f"[Step 2] Alignment output not found, but skipping as requested")

        # ----- Step 3: Segmentation -----
        if from_step <= 3:
            logger.info(f"[Step 3] Segmentation ...")
            seg_res = run_segmentation(block, dataset, force=force)
            block_result.segment_status = seg_res["status"]
            block_result.segment_error = seg_res.get("error")
            block_result.cell_count = seg_res.get("cell_count", 0)
            block_result.output_features = seg_res.get("output_features")
            if seg_res["status"] == "error":
                logger.error(f"[Step 3] Segmentation failed: {seg_res.get('error')}")
        else:
            skipped = check_segmentation_done(block, dataset)
            block_result.segment_status = "skipped"
            logger.info(f"[Step 3] Segmentation skipped (--from-step>3)")
            if not skipped:
                logger.warning(f"[Step 3] Segmentation output not found, but skipping as requested")

        # ----- Step 4: Grading and Molecular Subtype Classification -----
        if from_step <= 4:
            logger.info(f"[Step 4] Grading and molecular subtype classification ...")
            grade_res = run_grading_and_subtyping(block, dataset, force=force)
            block_result.grading_status = grade_res["status"]
            block_result.grading_error = grade_res.get("error")
            if grade_res["status"] == "error":
                logger.error(f"[Step 4] Grading failed: {grade_res.get('error')}")
            else:
                logger.info(f"[Step 4] Grading completed successfully")
        else:
            skipped = check_grading_done(block, dataset)
            block_result.grading_status = "skipped"
            logger.info(f"[Step 4] Grading skipped (--from-step>4)")
            if not skipped:
                logger.warning(f"[Step 4] Grading output not found, but skipping as requested")

        # ----- Step 5: Alignment Visualization -----
        if from_step <= 5:
            logger.info(f"[Step 5] Alignment visualization ...")
            try:
                vis_res = run_visualization_alignment([block], dataset, force=force)
                if vis_res:
                    block_result.vis_align_status = "done"
                    logger.info(f"[Step 5] Alignment visualization completed")
                else:
                    block_result.vis_align_status = "error"
                    logger.warning(f"[Step 5] Alignment visualization partially failed")
            except Exception as e:
                block_result.vis_align_status = "error"
                logger.error(f"[Step 5] Alignment visualization error: {str(e)}")
        else:
            block_result.vis_align_status = "skipped"
            logger.info(f"[Step 5] Alignment visualization skipped (--from-step>5)")

        block_result.end_time = datetime.now()

        duration = block_result.duration_seconds
        overall = block_result.overall_status
        logger.info(
            f"[Done] {block}: preproc={block_result.preprocess_status}  "
            f"stitch={block_result.stitch_status}  "
            f"align={block_result.align_status}  segment={block_result.segment_status}  "
            f"grading={block_result.grading_status}  "
            f"cells={block_result.cell_count}  ({duration:.1f}s)  [{overall}]"
        )

        results.append(block_result)

    return results


# =====================================================================
# Main function
# =====================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    logger = setup_logging(
        log_dir=args.log_dir,
        quiet=args.quiet,
    )

    logger.info("=" * 60)
    logger.info("TMA Pipeline - unified stitching / alignment / segmentation pipeline")
    logger.info(f"Dataset   : {args.dataset}")
    logger.info(f"From step : {args.from_step}")
    logger.info(f"Force     : {args.force}")
    logger.info("=" * 60)

    # Determine which blocks to process
    if args.all_blocks:
        blocks = discover_blocks(args.dataset)
        logger.info(f"Auto-discovered {len(blocks)} blocks: {blocks}")
    elif args.blocks:
        blocks = args.blocks
        logger.info(f"Processing {len(blocks)} blocks: {blocks}")
    else:
        logger.error("Please specify --blocks or --all-blocks")
        parser.print_help()
        sys.exit(1)

    if not blocks:
        logger.error(f"No blocks found (dataset={args.dataset})")
        sys.exit(1)

    # Run the pipeline
    start_time = datetime.now()
    results = run_pipeline(
        dataset=args.dataset,
        blocks=blocks,
        from_step=args.from_step,
        force=args.force,
    )
    total_duration = (datetime.now() - start_time).total_seconds()

    # ----- Step 6: Summary Visualization (cross-block summary) -----
    if args.from_step <= 6:
        logger.info(f"\n{'='*60}")
        logger.info(f"[Step 6] Running summary visualization...")
        logger.info(f"{'='*60}")
        try:
            run_visualization_summary(blocks, BASE_DIR, force=args.force)
            logger.info(f"[Step 6] Summary visualization completed")
        except Exception as e:
            logger.error(f"[Step 6] Summary visualization error: {str(e)}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline finished in {total_duration:.1f}s")
    logger.info(f"{'='*60}")

    # Generate report
    if args.save_report:
        report = generate_report(
            dataset=args.dataset,
            blocks=results,
            save=True,
            verbose=not args.quiet,
        )
        logger.info(f"\nReport saved.")

    # Exit code
    errors = sum(1 for b in results if b.overall_status == "error")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
