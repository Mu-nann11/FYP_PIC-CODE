"""
Figure 5: St. Gallen Molecular Subtype Classification Logic Diagram
Breast cancer molecular subtyping decision tree based on ER/PR/HER2/Ki67 expression
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
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
    'legend.fontsize':   9,
    'figure.dpi':       150,
    'savefig.dpi':       300,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        False,
})

# =============================================================================
# Color palette (IHC / colorblind-safe categorical)
# =============================================================================
COLORS = {
    'marker_bg':    '#F0F4F8',   # Light blue-grey for marker decision boxes
    'decision_bg':  '#E8F0FE',   # Decision node background
    'er':           '#43A047',   # ER  — green
    'pr':           '#FB8C00',   # PR  — orange
    'her2':         '#E53935',   # HER2 — red
    'ki67':         '#8E24AA',   # Ki67 — purple
    'lumA':         '#1565C0',   # Luminal A     — deep blue
    'lumB':         '#0288D1',   # Luminal B     — lighter blue
    'lumA_her2':    '#6A1B9A',   # Luminal A HER2+ — dark purple
    'lumB_her2':    '#9C27B0',   # Luminal B HER2+ — lighter purple
    'her2_enr':     '#C62828',   # HER2 Enriched  — dark red
    'tn':           '#546E7A',   # Triple Negative — slate grey
    'arrow':        '#37474F',   # Arrow colour
    'border':       '#263238',   # Box border
    'no':           '#EF9A9A',   # "No" / negative branch tint
    'yes':          '#A5D6A7',   # "Yes" / positive branch tint
    'neutral':      '#CFD8DC',   # Neutral / intermediate
}


# =============================================================================
# Node geometry helpers
# =============================================================================
def rounded_box(ax, cx, cy, w, h, text, facecolor, edgecolor='#263238',
                fontsize=9, textcolor='#1A1A1A', bold=False, radius=0.25,
                alpha=1.0, zorder=3):
    """Draw a rounded rectangle with centred text."""
    lw = 1.5
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight=weight, zorder=zorder+1,
            multialignment='center')


def diamond(ax, cx, cy, w, h, text, facecolor, edgecolor='#263238',
            fontsize=8.5, textcolor='#1A1A1A', alpha=1.0, zorder=3):
    """Draw a diamond (rhombus) decision node."""
    diamond_pts = np.array([
        [cx,     cy + h/2],
        [cx + w/2, cy    ],
        [cx,     cy - h/2],
        [cx - w/2, cy    ],
    ])
    from matplotlib.patches import Polygon
    poly = Polygon(diamond_pts, closed=True,
                   facecolor=facecolor, edgecolor=edgecolor,
                   linewidth=1.5, alpha=alpha, zorder=zorder)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', zorder=zorder+1,
            multialignment='center')


def arrow(ax, x0, y0, x1, y1, label='', color='#37474F', lw=1.6,
          label_color='#37474F', label_side='right', offset=0,
          fontsize=8, zorder=2):
    """Draw a straight arrow between two points with optional branch label."""
    ax.annotate('',
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    mutation_scale=12,
                ), zorder=zorder)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if label:
        dx = -0.08 if label_side == 'left' else 0.08
        if abs(x1 - x0) > abs(y1 - y0):
            # Horizontal-dominant: label above or below
            ly = my + 0.07 if label.lower() == 'yes' else my - 0.22
            lx = mx
            ha = 'center'
        else:
            # Vertical-dominant: label to the side
            ly = my
            lx = mx + 0.07 if label_side == 'right' else mx - 0.07
            ha = 'left' if label_side == 'right' else 'right'
        ax.text(lx, ly, label, ha=ha, va='center',
                fontsize=fontsize, color=label_color,
                fontweight='bold', zorder=zorder+1,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.8))


# =============================================================================
# Build the flowchart
# =============================================================================
def create_flowchart():
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 17)
    ax.axis('off')

    fig.suptitle('[Figure 5] St. Gallen Molecular Subtype Classification',
                 fontsize=16, fontweight='bold', y=0.97, color='#1A1A1A')
    fig.text(0.5, 0.945,
             'Decision tree based on ER/PR/HER2/Ki67 biomarker expression\n'
             '(St. Gallen International Expert Consensus, 2023)',
             ha='center', fontsize=10, color='#546E7A', style='italic')

    # -------------------------------------------------------------------------
    # Layout constants
    # -------------------------------------------------------------------------
    # Level y-coordinates
    Y_TOP    = 15.5   # Root — HER2?
    Y_L2     = 13.0   # HER2- branch: Hormone?
    Y_L2_H   = 13.0   # HER2+ branch: ER/PR?
    Y_L3     = 10.5   # Ki67?
    Y_L3_H   = 10.5   # Ki67? (for HER2+ hormone+ branch)
    Y_LEAF   = 7.5    # Final subtypes

    # Subtype x-coordinates (spaced across)
    X_HER2_POS   = 9.5   # HER2+ → right half
    X_HER2_NEG   = 3.0   # HER2- → left half
    X_H_NO_HORM  = 11.5  # HER2+ / no hormone → far right
    X_LUM_A_H2   = 8.5   # HER2+ hormone+ Ki67-  → LumA HER2+
    X_LUM_B_H2   = 10.5  # HER2+ hormone+ Ki67+  → LumB HER2+

    # -------------------------------------------------------------------------
    # 1. ROOT — HER2? decision
    # -------------------------------------------------------------------------
    diamond(ax, 6.5, Y_TOP, 2.2, 0.9,
            'HER2\nExpression?',
            facecolor=COLORS['decision_bg'],
            edgecolor=COLORS['her2'], textcolor=COLORS['her2'], fontsize=9.5)

    # -------------------------------------------------------------------------
    # 2a. HER2- branch → Hormone Receptor?
    # -------------------------------------------------------------------------
    arrow(ax, 5.4, Y_TOP - 0.45, 5.4, Y_L2 + 0.45,
          label='HER2−', color=COLORS['her2'], label_side='left',
          fontsize=8.5, lw=1.8)

    diamond(ax, 5.4, Y_L2, 2.6, 0.9,
            'ER+ or PR+\n(Hormone)?',
            facecolor=COLORS['decision_bg'],
            edgecolor=COLORS['border'], fontsize=9.5)

    # -------------------------------------------------------------------------
    # 2b. HER2+ branch → Hormone Receptor?
    # -------------------------------------------------------------------------
    arrow(ax, 7.6, Y_TOP - 0.45, X_HER2_POS, Y_L2_H,
          label='HER2+', color=COLORS['her2'], label_side='right',
          fontsize=8.5, lw=1.8)

    diamond(ax, X_HER2_POS, Y_L2_H, 2.6, 0.9,
            'ER+ or PR+\n(Hormone)?',
            facecolor=COLORS['decision_bg'],
            edgecolor=COLORS['border'], fontsize=9.5)

    # -------------------------------------------------------------------------
    # HER2- / Hormone NO → Triple Negative
    # -------------------------------------------------------------------------
    arrow(ax, 4.15, Y_L2 - 0.45, 1.5, Y_LEAF + 1.0,
          label='Hormone−', color='#78909C', label_side='left',
          fontsize=8, lw=1.6)
    rounded_box(ax, 1.5, Y_LEAF, 2.4, 1.1,
                'Triple Negative\n(TNBC)',
                facecolor='#ECEFF1', edgecolor=COLORS['tn'],
                textcolor=COLORS['tn'], fontsize=9, bold=True,
                radius=0.35)
    ax.text(1.5, Y_LEAF - 0.9, 'ER− / PR− / HER2−',
            ha='center', va='top', fontsize=7.5, color='#607D8B',
            style='italic')

    # -------------------------------------------------------------------------
    # HER2- / Hormone YES → Ki67?
    # -------------------------------------------------------------------------
    arrow(ax, 5.4 + 1.3, Y_L2 - 0.45, 5.4 + 1.3, Y_L3 + 0.45,
          label='Hormone+', color='#43A047', label_side='right',
          fontsize=8, lw=1.6)

    diamond(ax, 6.7, Y_L3, 2.2, 0.9,
            'Ki67\nHigh?',
            facecolor=COLORS['decision_bg'],
            edgecolor=COLORS['ki67'], textcolor=COLORS['ki67'], fontsize=9.5)

    # HER2- / Hormone+ / Ki67− → Luminal A
    arrow(ax, 5.6, Y_L3 - 0.45, 4.5, Y_LEAF + 0.5,
          label='Ki67−', color='#78909C', label_side='left',
          fontsize=8, lw=1.6)
    rounded_box(ax, 4.5, Y_LEAF, 2.4, 1.1,
                'Luminal A',
                facecolor='#E3F2FD', edgecolor=COLORS['lumA'],
                textcolor=COLORS['lumA'], fontsize=10, bold=True,
                radius=0.35)
    ax.text(4.5, Y_LEAF - 0.9, 'HER2− / Ki67−',
            ha='center', va='top', fontsize=7.5, color='#546E7A',
            style='italic')

    # HER2- / Hormone+ / Ki67+ → Luminal B
    arrow(ax, 7.8, Y_L3 - 0.45, 8.0, Y_LEAF + 0.5,
          label='Ki67+', color=COLORS['ki67'], label_side='right',
          fontsize=8, lw=1.6)
    rounded_box(ax, 8.0, Y_LEAF, 2.4, 1.1,
                'Luminal B',
                facecolor='#E1F5FE', edgecolor=COLORS['lumB'],
                textcolor=COLORS['lumB'], fontsize=10, bold=True,
                radius=0.35)
    ax.text(8.0, Y_LEAF - 0.9, 'HER2− / Ki67+',
            ha='center', va='top', fontsize=7.5, color='#546E7A',
            style='italic')

    # -------------------------------------------------------------------------
    # HER2+ / Hormone NO → HER2 Enriched
    # -------------------------------------------------------------------------
    arrow(ax, X_HER2_POS - 1.3, Y_L2_H - 0.45, X_H_NO_HORM, Y_LEAF + 1.0,
          label='Hormone−', color='#78909C', label_side='left',
          fontsize=8, lw=1.6)
    rounded_box(ax, X_H_NO_HORM, Y_LEAF, 2.8, 1.1,
                'HER2 Enriched',
                facecolor='#FFEBEE', edgecolor=COLORS['her2_enr'],
                textcolor=COLORS['her2_enr'], fontsize=10, bold=True,
                radius=0.35)
    ax.text(X_H_NO_HORM, Y_LEAF - 0.9, 'HER2+ / ER− / PR−',
            ha='center', va='top', fontsize=7.5, color='#C62828',
            style='italic')

    # -------------------------------------------------------------------------
    # HER2+ / Hormone YES → Ki67?
    # -------------------------------------------------------------------------
    arrow(ax, X_HER2_POS + 1.3, Y_L2_H - 0.45, X_LUM_B_H2, Y_L3_H,
          label='Hormone+', color='#43A047', label_side='right',
          fontsize=8, lw=1.6)

    diamond(ax, 9.5, Y_L3_H, 2.2, 0.9,
            'Ki67\nHigh?',
            facecolor=COLORS['decision_bg'],
            edgecolor=COLORS['ki67'], textcolor=COLORS['ki67'], fontsize=9.5)

    # HER2+ / Hormone+ / Ki67− → Luminal A (HER2+)
    arrow(ax, 8.4, Y_L3_H - 0.45, X_LUM_A_H2, Y_LEAF + 0.5,
          label='Ki67−', color='#78909C', label_side='left',
          fontsize=8, lw=1.6)
    rounded_box(ax, X_LUM_A_H2, Y_LEAF, 2.6, 1.1,
                'Luminal A\n(HER2+)',
                facecolor='#F3E5F5', edgecolor=COLORS['lumA_her2'],
                textcolor=COLORS['lumA_her2'], fontsize=10, bold=True,
                radius=0.35)
    ax.text(X_LUM_A_H2, Y_LEAF - 0.9, 'HER2+ / Hormone+ / Ki67−',
            ha='center', va='top', fontsize=7.5, color='#6A1B9A',
            style='italic')

    # HER2+ / Hormone+ / Ki67+ → Luminal B (HER2+)
    arrow(ax, 10.6, Y_L3_H - 0.45, X_LUM_B_H2, Y_LEAF + 0.5,
          label='Ki67+', color=COLORS['ki67'], label_side='right',
          fontsize=8, lw=1.6)
    rounded_box(ax, X_LUM_B_H2, Y_LEAF, 2.6, 1.1,
                'Luminal B\n(HER2+)',
                facecolor='#F3E5F5', edgecolor=COLORS['lumB_her2'],
                textcolor=COLORS['lumB_her2'], fontsize=10, bold=True,
                radius=0.35)
    ax.text(X_LUM_B_H2, Y_LEAF - 0.9, 'HER2+ / Hormone+ / Ki67+',
            ha='center', va='top', fontsize=7.5, color='#6A1B9A',
            style='italic')

    # -------------------------------------------------------------------------
    # 3. Legend (bottom)
    # -------------------------------------------------------------------------
    legend_x = 0.3
    legend_y = 2.8

    # Title
    ax.text(legend_x, legend_y + 0.4, 'Marker Legend',
            fontsize=9, fontweight='bold', color='#37474F', va='center')

    marker_items = [
        ('#E53935', 'HER2'),
        ('#43A047', 'ER'),
        ('#FB8C00', 'PR'),
        ('#8E24AA', 'Ki67'),
    ]
    for i, (c, name) in enumerate(marker_items):
        bx = legend_x + i * 2.0
        rect = FancyBboxPatch((bx, legend_y - 0.25), 0.5, 0.5,
                              boxstyle="round,pad=0,rounding_size=0.1",
                              facecolor=c, edgecolor='#263238', lw=1)
        ax.add_patch(rect)
        ax.text(bx + 0.65, legend_y, name,
                va='center', fontsize=8.5, color='#263238')

    # Subtype legend
    sub_legend_y = 1.8
    subtypes_legend = [
        ('#E3F2FD', COLORS['lumA'],       'Luminal A'),
        ('#E1F5FE', COLORS['lumB'],       'Luminal B'),
        ('#F3E5F5', COLORS['lumA_her2'],  'Luminal A (HER2+)'),
        ('#F3E5F5', COLORS['lumB_her2'],  'Luminal B (HER2+)'),
        ('#FFEBEE', COLORS['her2_enr'],   'HER2 Enriched'),
        ('#ECEFF1', COLORS['tn'],          'Triple Negative'),
    ]
    n_cols = 3
    for idx, (bg, fg, name) in enumerate(subtypes_legend):
        col = idx % n_cols
        row = idx // n_cols
        bx = legend_x + col * 3.8
        by = sub_legend_y - row * 0.7
        rect = FancyBboxPatch((bx, by - 0.2), 0.5, 0.4,
                              boxstyle="round,pad=0,rounding_size=0.1",
                              facecolor=bg, edgecolor=fg, lw=1.5)
        ax.add_patch(rect)
        ax.text(bx + 0.65, by, name,
                va='center', fontsize=8.5, color=fg, fontweight='bold')

    # Decision node legend
    ax.text(legend_x, sub_legend_y - 1.2, 'Decision node',
            fontsize=8.5, color='#37474F')
    diamond_pts = np.array([[legend_x + 0.6, sub_legend_y - 0.7],
                             [legend_x + 1.1, sub_legend_y - 1.0],
                             [legend_x + 0.6, sub_legend_y - 1.3],
                             [legend_x + 0.1, sub_legend_y - 1.0]])
    from matplotlib.patches import Polygon
    poly = Polygon(diamond_pts, closed=True,
                   facecolor=COLORS['decision_bg'], edgecolor='#263238', lw=1)
    ax.add_patch(poly)

    # Vertical annotation on the left side
    ax.annotate('', xy=(0.1, Y_TOP), xytext=(0.1, Y_LEAF - 0.5),
                arrowprops=dict(arrowstyle='<->', color='#90A4AE', lw=1.2))
    ax.text(-0.2, (Y_TOP + Y_LEAF) / 2,
            'Classification\nhierarchy',
            ha='right', va='center', fontsize=7.5,
            color='#78909C', style='italic', rotation=90)

    # -------------------------------------------------------------------------
    # Note box
    # -------------------------------------------------------------------------
    note_text = (
        'Note: "Hormone+" = ER≥1 or PR≥1.  '
        '"Ki67+" = Ki67≥1 (positive).  '
        'HER2+ = HER2 grade ≥1.  '
        'Grade 0 = negative.  '
        'Thresholds derived from Otsu analysis of co-registered imaging data.'
    )
    fig.text(0.5, 0.02, note_text,
             ha='center', fontsize=7.5, color='#607D8B',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA',
                       edgecolor='#CFD8DC', alpha=0.9))

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    output_dir = PathLib(r'd:\Try_munan\FYP_LAST\Report\Report_figure')
    output_dir.mkdir(parents=True, exist_ok=True)

    out_png = output_dir / 'figure5.png'
    out_pdf = output_dir / 'figure5.pdf'

    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')

    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    return out_png, out_pdf


if __name__ == '__main__':
    create_flowchart()
