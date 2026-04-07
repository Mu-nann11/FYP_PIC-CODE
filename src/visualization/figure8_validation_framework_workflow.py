"""
[Figure 8] Validation Framework Workflow Diagram
Centroid-based nearest-neighbour matching pipeline for evaluating
segmentation and grading performance against QuPath reference data.

Layout:  INPUT  -->  PROCESSING  -->  OUTPUT
         (left)       (centre)       (right)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.path import Path
import numpy as np
from pathlib import Path as PathLib

# =============================================================================
# Scientific Plotting rcParams (per SKILL.md)
# =============================================================================
import matplotlib
matplotlib.rcParams.update({
    'font.family':      'Arial',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        False,
})

# =============================================================================
# Color palette — IHC / colorblind-safe
# =============================================================================
C = {
    # Structural colours
    'input_bg':      '#F0F4F8',   # Input panel background
    'proc_bg':       '#E8F0FE',   # Processing panel background
    'output_bg':     '#E8F5E9',   # Output panel background
    'step_bg':       '#DDEEFF',   # Step box fill
    'step_border':   '#1565C0',   # Step box border / title colour
    'in_box':        '#C8E6C9',   # Input box fill
    'in_border':     '#2E7D32',   # Input box border
    'out_box':       '#C8E6C9',   # Output box fill
    'out_border':    '#2E7D32',   # Output box border
    'arrow':         '#37474F',
    'border':        '#263238',
    # Marker colours (for sub-labels)
    'her2':          '#E53935',
    'er':            '#43A047',
    'pr':            '#FB8C00',
    'ki67':          '#8E24AA',
    # Annotation
    'annot':         '#607D8B',
    'metric_pos':    '#2E7D32',
    'metric_err':    '#1565C0',
}


# =============================================================================
# Geometry helpers
# =============================================================================

def rounded_box(ax, cx, cy, w, h, text, facecolor, edgecolor='#263238',
                fontsize=9, textcolor='#1A1A1A', bold=False, radius=0.25,
                alpha=1.0, zorder=3, subtext=None):
    """Draw a rounded rectangle with centred text and optional sub-text."""
    lw = 1.8
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight=weight, zorder=zorder + 1,
            multialignment='center')
    if subtext:
        ax.text(cx, cy - h * 0.28, subtext, ha='center', va='center',
                fontsize=fontsize - 2, color=annot, zorder=zorder + 1,
                multialignment='center', style='italic')


def pill(ax, cx, cy, w, h, text, facecolor, edgecolor='#263238',
         fontsize=8, textcolor='#1A1A1A', radius=0.5):
    """Draw a pill-shaped label."""
    lw = 1.2
    pill_box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=3,
    )
    ax.add_patch(pill_box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', zorder=4)


def step_box(ax, cx, cy, w, h, title, body_lines, facecolor, title_color,
             edgecolor='#1565C0', fontsize=9, title_fs=9.5, radius=0.3):
    """Draw a titled step box with multi-line body."""
    lw = 1.8
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=3,
    )
    ax.add_patch(box)

    # Title bar
    title_h = 0.42
    title_box = FancyBboxPatch(
        (cx - w / 2, cy + h / 2 - title_h), w, title_h,
        boxstyle=f"round,pad=0,rounding_size=0.15",
        facecolor=title_color, edgecolor='none',
        linewidth=0, alpha=0.85, zorder=4,
    )
    ax.add_patch(title_box)

    ax.text(cx, cy + h / 2 - title_h / 2, title,
            ha='center', va='center', fontsize=title_fs,
            color='white', fontweight='bold', zorder=5)

    # Body text
    body_text = '\n'.join(body_lines)
    ax.text(cx, cy - 0.05, body_text,
            ha='center', va='center', fontsize=fontsize,
            color='#1A1A1A', zorder=4,
            multialignment='center')


def straight_arrow(ax, x0, y0, x1, y1, color='#37474F', lw=2.0,
                   label='', label_color='#37474F', label_offset=(0, 0.18),
                   fontsize=8, zorder=2):
    """Draw a straight arrow with an optional label above it."""
    ax.annotate('',
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    mutation_scale=14,
                ), zorder=zorder)
    if label:
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=fontsize, color=label_color, fontweight='bold',
                zorder=zorder + 1,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))


def curved_arrow(ax, x0, y0, x1, y1, color='#37474F', lw=1.6,
                 label='', label_color='#37474F', fontsize=8,
                 connectionstyle="arc3,rad=0.0", zorder=2):
    """Draw a curved arrow between two points."""
    ax.annotate('',
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    connectionstyle=connectionstyle,
                    mutation_scale=12,
                ), zorder=zorder)
    if label:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        ax.text(mx, my + 0.18, label, ha='center', va='center',
                fontsize=fontsize, color=label_color, fontweight='bold',
                zorder=zorder + 1,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))


# =============================================================================
# Build the diagram
# =============================================================================

annot = C['annot']


def create_validation_workflow():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------
    fig.suptitle('[Figure 8] Validation Framework Workflow',
                 fontsize=16, fontweight='bold', y=0.97, color='#1A1A1A')
    fig.text(0.5, 0.935,
             'Centroid-based Nearest-Neighbour Matching for Segmentation '
             '& Grading Evaluation Against QuPath Reference Data',
             ha='center', fontsize=10.5, color=C['annot'], style='italic')

    # -------------------------------------------------------------------------
    # Stage backgrounds
    # -------------------------------------------------------------------------
    # INPUT panel
    ax.add_patch(FancyBboxPatch((0.1, 0.5), 3.8, 9.0,
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                facecolor=C['input_bg'], edgecolor='#B0BEC5',
                                linewidth=1.5, alpha=0.5, zorder=0))
    ax.text(2.0, 9.6, 'INPUT', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#37474F', zorder=1)

    # PROCESSING panel
    ax.add_patch(FancyBboxPatch((4.4, 0.5), 8.6, 9.0,
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                facecolor=C['proc_bg'], edgecolor='#90CAF9',
                                linewidth=1.5, alpha=0.5, zorder=0))
    ax.text(8.7, 9.6, 'PROCESSING', ha='center', va='center',
            fontsize=11, fontweight='bold', color=C['step_border'], zorder=1)

    # OUTPUT panel
    ax.add_patch(FancyBboxPatch((13.5, 0.5), 5.3, 9.0,
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                facecolor=C['output_bg'], edgecolor='#A5D6A7',
                                linewidth=1.5, alpha=0.5, zorder=0))
    ax.text(16.15, 9.6, 'OUTPUT', ha='center', va='center',
            fontsize=11, fontweight='bold', color=C['metric_pos'], zorder=1)

    # -------------------------------------------------------------------------
    # INPUT boxes
    # -------------------------------------------------------------------------
    # --- Pipeline Results
    rounded_box(ax, 2.0, 7.8, 3.3, 1.1,
                'Pipeline Output\n(Grading Results)',
                facecolor='#DCEDC8', edgecolor=C['in_border'],
                textcolor='#1B5E20', bold=True, radius=0.3)
    pill(ax, 2.0, 7.0, 2.6, 0.38, 'Cell-level intensity table',
         '#E8F5E9', '#2E7D32', fontsize=7.5, textcolor='#2E7D32')
    pill(ax, 2.0, 6.4, 2.6, 0.38, 'Centroid coordinates (x, y)',
         '#E8F5E9', '#2E7D32', fontsize=7.5, textcolor='#2E7D32')
    pill(ax, 2.0, 5.8, 2.6, 0.38, 'Cellpose segmentation mask',
         '#E8F5E9', '#2E7D32', fontsize=7.5, textcolor='#2E7D32')

    # --- QuPath Reference
    rounded_box(ax, 2.0, 4.2, 3.3, 1.1,
                'QuPath Reference\n(Ground Truth)',
                facecolor='#DCEDC8', edgecolor=C['in_border'],
                textcolor='#1B5E20', bold=True, radius=0.3)
    pill(ax, 2.0, 3.4, 2.6, 0.38, 'Annotated cell objects',
         '#E8F5E9', '#2E7D32', fontsize=7.5, textcolor='#2E7D32')
    pill(ax, 2.0, 2.8, 2.6, 0.38, 'Reference centroid (x, y)',
         '#E8F5E9', '#2E7D32', fontsize=7.5, textcolor='#2E7D32')
    pill(ax, 2.0, 2.2, 2.6, 0.38, 'Manual intensity labels',
         '#E8F5E9', '#2E7D32', fontsize=7.5, textcolor='#2E7D32')

    # -------------------------------------------------------------------------
    # Arrows from INPUT to PROCESSING
    # -------------------------------------------------------------------------
    straight_arrow(ax, 3.65, 7.8, 4.85, 8.2,
                   color=C['step_border'], lw=2.0, label='Pipeline',
                   label_color=C['step_border'], label_offset=(0, 0.22), fontsize=8)
    straight_arrow(ax, 3.65, 4.2, 4.85, 5.8,
                   color=C['step_border'], lw=2.0, label='QuPath',
                   label_color=C['step_border'], label_offset=(0, 0.22), fontsize=8)

    # -------------------------------------------------------------------------
    # PROCESSING — three step boxes
    # -------------------------------------------------------------------------
    STEP_Y = 6.5        # vertical centre of step boxes
    STEP_W = 2.5
    STEP_H = 3.5

    # Step 1 — Centroid Extraction
    step1_x = 6.6
    step_box(ax, step1_x, STEP_Y, STEP_W, STEP_H,
             'Step 1\nCentroid Extraction',
             ['Extract (x, y) from',
              'pipeline masks &',
              'QuPath objects',
              '',
              '- DAPI nucleus centre',
              '- Cytoplasm centroid',
              '- Cell bounding box'],
             facecolor='#DCEEFB', title_color=C['step_border'],
             edgecolor=C['step_border'], radius=0.3)

    # Step 2 — KD-Tree Matching
    step2_x = 9.4
    step_box(ax, step2_x, STEP_Y, STEP_W, STEP_H,
             'Step 2\nKD-Tree Matching',
             ['scipy.spatial',
              '.cKDTree nearest',
              'neighbour search',
              '',
              'For each QuPath cell:',
              '-> find pipeline cell',
              '  with min. distance',
              '-> apply distance',
              '  threshold filter',
              '-> label as TP/FP/FN'],
             facecolor='#DCEEFB', title_color=C['step_border'],
             edgecolor=C['step_border'], radius=0.3)

    # Step 3 — Intensity Validation
    step3_x = 12.2
    step_box(ax, step3_x, STEP_Y, STEP_W, STEP_H,
             'Step 3\nIntensity Validation',
             ['Compare matched',
              'cell intensities',
              '',
              '- HER2 grade >= 1',
              '- ER   >= 1 (positive)',
              '- PR   >= 1 (positive)',
              '- Ki67 >= 1 (positive)',
              '',
              '-> Classify TP / FP'],
             facecolor='#DCEEFB', title_color=C['step_border'],
             edgecolor=C['step_border'], radius=0.3)

    # Arrows between steps
    straight_arrow(ax, step1_x + STEP_W / 2, STEP_Y,
                   step2_x - STEP_W / 2, STEP_Y,
                   color=C['step_border'], lw=2.2, label='',
                   label_offset=(0, 0.15), fontsize=8)

    straight_arrow(ax, step2_x + STEP_W / 2, STEP_Y,
                   step3_x - STEP_W / 2, STEP_Y,
                   color=C['step_border'], lw=2.2, label='',
                   label_offset=(0, 0.15), fontsize=8)

    # Vertical arrow from step outputs to OUTPUT
    straight_arrow(ax, step3_x + STEP_W / 2 + 0.15, STEP_Y - STEP_H / 2,
                   13.9, 4.5,
                   color=C['metric_pos'], lw=2.0, label='Matched pairs',
                   label_color=C['metric_pos'], label_offset=(0.3, 0),
                   fontsize=8)

    # -------------------------------------------------------------------------
    # OUTPUT — metrics panel
    # -------------------------------------------------------------------------

    # --- Primary metrics box
    rounded_box(ax, 16.2, 7.0, 4.5, 3.0,
                 '',
                 facecolor='#C8E6C9', edgecolor=C['out_border'],
                 radius=0.3, zorder=3)
    ax.text(16.2, 8.5, 'Classification Metrics',
            ha='center', va='center', fontsize=10,
            fontweight='bold', color='#1B5E20', zorder=4)

    metrics = [
        ('Precision', 'TP / (TP + FP)', '#2E7D32'),
        ('Recall',    'TP / (TP + FN)', '#1565C0'),
        ('F1-Score',  '2·P·R / (P + R)', '#E53935'),
        ('Accuracy', '(TP + TN) / All', '#F57F17'),
    ]
    for i, (name, formula, col) in enumerate(metrics):
        y_m = 7.9 - i * 0.68
        ax.text(14.05, y_m, name + ':',
                ha='left', va='center', fontsize=9.5,
                fontweight='bold', color=col, zorder=4)
        ax.text(17.8, y_m, formula,
                ha='right', va='center', fontsize=9,
                color='#37474F', zorder=4, style='italic')
        # Underline
        ax.plot([14.0, 18.0], [y_m - 0.22, y_m - 0.22],
                color='#C8E6C9', lw=1, zorder=3)

    # --- Secondary metrics box
    rounded_box(ax, 16.2, 3.0, 4.5, 2.2,
                '',
                facecolor='#DCEEFB', edgecolor=C['step_border'],
                radius=0.3, zorder=3)
    ax.text(16.2, 4.1, 'Spatial & Intensity Metrics',
            ha='center', va='center', fontsize=10,
            fontweight='bold', color=C['step_border'], zorder=4)

    sec_metrics = [
        ('Mean Position Error', 'μm (matched pairs)', '#37474F'),
        ('Intensity Pearson r', 'Pipeline vs QuPath', '#37474F'),
    ]
    for i, (name, unit, col) in enumerate(sec_metrics):
        y_m = 3.5 - i * 0.75
        ax.text(14.05, y_m, name + ':',
                ha='left', va='center', fontsize=9,
                fontweight='bold', color=col, zorder=4)
        ax.text(17.8, y_m, unit,
                ha='right', va='center', fontsize=8.5,
                color=C['annot'], zorder=4, style='italic')

    # -------------------------------------------------------------------------
    # Legend — bottom strip
    # -------------------------------------------------------------------------
    legend_y = 1.0

    ax.text(0.4, legend_y + 0.15, 'Marker colours:',
            fontsize=8.5, color='#37474F', fontweight='bold', va='center')

    markers = [
        (C['her2'], 'HER2'),
        (C['er'],   'ER'),
        (C['pr'],   'PR'),
        (C['ki67'], 'Ki67'),
    ]
    for i, (col, name) in enumerate(markers):
        bx = 2.8 + i * 1.6
        rect = FancyBboxPatch((bx, legend_y - 0.15), 0.45, 0.3,
                              boxstyle="round,pad=0,rounding_size=0.08",
                              facecolor=col, edgecolor='#263238', lw=1)
        ax.add_patch(rect)
        ax.text(bx + 0.55, legend_y, name,
                va='center', fontsize=8.5, color='#263238')

    # Arrow legend
    ax.annotate('', xy=(9.0, legend_y), xytext=(8.3, legend_y),
                arrowprops=dict(arrowstyle='->', color='#37474F', lw=2,
                                mutation_scale=12))
    ax.text(8.65, legend_y, 'Data flow',
            ha='center', va='center', fontsize=8, color='#37474F')

    # Distance threshold note
    ax.text(12.5, legend_y,
            'Distance threshold: 5 μm (physical scale from pixel size)',
            ha='center', va='center', fontsize=8,
            color=C['annot'], style='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FAFAFA',
                      edgecolor='#CFD8DC', alpha=0.9))

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    output_dir = PathLib(r'd:\Try_munan\FYP_LAST\Report\Report_figure')
    output_dir.mkdir(parents=True, exist_ok=True)

    out_png = output_dir / 'figure8.png'
    out_pdf = output_dir / 'figure8.pdf'

    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')

    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    return out_png, out_pdf


if __name__ == '__main__':
    create_validation_workflow()
