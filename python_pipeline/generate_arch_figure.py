"""
Cleaner architecture diagram for the SOLIS paper.

Design principles:
  * Strict top-down vertical flow for the main pipeline.
  * IMM mixing box is CENTERED below its two parent models (not offset to the right).
  * Support modules are stacked in a clean right-hand column, vertically aligned
    with the main pipeline boxes they advise.
  * All main arrows are short vertical segments only — no diagonal lines, no
    crossings.
  * Advisory feedback uses dashed horizontal arrows from the right-hand column
    into the main pipeline (also no crossings).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

FIG_DIR = '/home/z/my-project/scripts/paper_figures'

BLACK = '#000000'
DARK  = '#222222'
GRAY  = '#666666'
LIGHT = '#DDDDDD'
PALE  = '#F2F2F2'
WHITE = '#FFFFFF'


def box(ax, x, y, w, h, title, sub='', fs=9, sub_fs=7,
        facecolor=WHITE, edgecolor=BLACK, lw=1.3):
    """Rounded rectangle with bold title and optional subtitle."""
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=2,
    )
    ax.add_patch(b)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, title, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=BLACK, zorder=3)
        ax.text(x + w / 2, y + h * 0.26, sub, ha='center', va='center',
                fontsize=sub_fs, color=GRAY, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=BLACK, zorder=3)


def varrow(ax, x, y_top, y_bot, color=BLACK, lw=1.3):
    """Vertical arrow pointing DOWN from (x, y_top) to (x, y_bot)."""
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=0, shrinkB=0), zorder=1)


def harrow(ax, x_left, y, x_right, y_right=None, color=GRAY, lw=1.0, dashed=True):
    """Horizontal arrow from (x_left, y) to (x_right, y_right).
    If y_right is None, defaults to y (pure horizontal).
    If y_right != y, draws an L-shaped elbow (horizontal then short vertical)."""
    linestyle = (0, (4, 3)) if dashed else 'solid'
    if y_right is None or y_right == y:
        ax.annotate('', xy=(x_right, y), xytext=(x_left, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    linestyle=linestyle,
                                    shrinkA=0, shrinkB=0), zorder=1)
    else:
        # Horizontal segment (no arrowhead)
        ax.plot([x_left, x_right], [y, y], color=color, lw=lw,
                linestyle=linestyle, zorder=1)
        # Short vertical with arrowhead
        ax.annotate('', xy=(x_right, y_right), xytext=(x_right, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    linestyle=linestyle,
                                    shrinkA=0, shrinkB=0), zorder=1)


def harrow_to_main(ax, x_left, y, x_right, y_right, color=GRAY, lw=1.0):
    """Horizontal then short vertical elbow arrow (for off-alignment advisory)."""
    ax.plot([x_left, x_right], [y, y], color=color, lw=lw,
            linestyle=(0, (4, 3)), zorder=1)
    ax.annotate('', xy=(x_right, y_right), xytext=(x_right, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=(0, (4, 3)),
                                shrinkA=0, shrinkB=0), zorder=1)


def elbow(ax, x_from, y_from, x_to, y_to, color=BLACK, lw=1.3):
    """Right-angle elbow: vertical down then horizontal, OR horizontal then vertical.
    Here we always go: down then horizontal-right (or horizontal-left)."""
    # Draw vertical segment
    ax.plot([x_from, x_from], [y_from, y_to], color=color, lw=lw, zorder=1)
    # Draw horizontal segment with arrowhead
    ax.annotate('', xy=(x_to, y_to), xytext=(x_from, y_to),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=0, shrinkB=0), zorder=1)


def create_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 8.0), dpi=200)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 8.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ---------- Geometry ----------
    # Main pipeline occupies x in [0.4, 6.6], centered at x_main = 3.5
    # Support column occupies x in [7.3, 10.1]
    x_main = 3.5          # center x of main pipeline
    main_left  = 0.6
    main_right = 6.4
    main_w = main_right - main_left     # 5.8
    sup_left  = 7.3
    sup_right = 10.1
    sup_w = sup_right - sup_left        # 2.8
    sup_center = (sup_left + sup_right) / 2

    # ---------- Title ----------
    ax.text(x_main, 7.75, 'SOLIS System Architecture',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x_main, 7.50,
            'Top-down pipeline flow with right-side support modules '
            '(dashed = advisory feedback)',
            ha='center', va='center', fontsize=8, color=GRAY, style='italic')

    # ---------- Lane 1: INPUT ----------
    y_in_top = 7.10
    in_h = 0.75
    in_w = 2.2
    sxr_x = x_main - 0.5 - in_w
    hxr_x = x_main + 0.5
    sxr_c = sxr_x + in_w / 2    # center of SoLEXS box
    hxr_c = hxr_x + in_w / 2    # center of HEL1OS box
    box(ax, sxr_x, y_in_top - in_h, in_w, in_h,
        'SoLEXS', 'SXR: 2--22 keV', fs=9)
    box(ax, hxr_x, y_in_top - in_h, in_w, in_h,
        'HEL1OS', 'HXR: 10--150 keV', fs=9)
    ax.text(0.20, y_in_top - in_h / 2, 'INPUT', ha='center', va='center',
            rotation=90, fontsize=8, color=GRAY, fontweight='bold')

    # ---------- Lane 2: PREPROCESSING ----------
    y_pp_top = 6.10
    pp_h = 0.75
    pp_x = main_left + 0.4
    pp_w = main_w - 0.8
    pp_c = pp_x + pp_w / 2
    box(ax, pp_x, y_pp_top - pp_h, pp_w, pp_h,
        'Preprocessing Layer',
        'Hampel  |  Saturation  |  Lag re-estimation  |  QPP (CWT)',
        fs=9)
    ax.text(0.20, y_pp_top - pp_h / 2, 'PREPROCESS', ha='center', va='center',
            rotation=90, fontsize=8, color=GRAY, fontweight='bold')

    # ---------- Lane 3: 5D UKF ----------
    y_ukf_top = 5.10
    ukf_h = 0.75
    box(ax, pp_x, y_ukf_top - ukf_h, pp_w, ukf_h,
        '5D Unscented Kalman Filter (UKF)',
        '11 sigma points  |  Joseph-form covariance',
        fs=9)
    ax.text(0.20, y_ukf_top - ukf_h / 2, 'ESTIMATE', ha='center', va='center',
            rotation=90, fontsize=8, color=GRAY, fontweight='bold')

    # ---------- Lane 4: IMM (two parallel models + mixing) ----------
    y_imm_top = 4.10
    imm_h = 0.80
    m_w = 1.9
    m_gap = 0.6
    q_x = x_main - m_gap / 2 - m_w
    p_x = x_main + m_gap / 2
    q_c = q_x + m_w / 2
    p_c = p_x + m_w / 2
    box(ax, q_x, y_imm_top - imm_h, m_w, imm_h,
        'Quiet Sun', r'$\beta = 0.7$' + '\nstickier decay', fs=8, sub_fs=7)
    box(ax, p_x, y_imm_top - imm_h, m_w, imm_h,
        'Pre-Flare', r'$\beta = 0.3$' + '\nfast SXR rise', fs=8, sub_fs=7)
    ax.text(0.20, y_imm_top - imm_h / 2, 'IMM', ha='center', va='center',
            rotation=90, fontsize=8, color=GRAY, fontweight='bold')

    # IMM Mixing box (centered below the two model boxes)
    y_mix_top = 3.00
    mix_h = 0.70
    mix_w = 3.0
    mix_x = x_main - mix_w / 2
    mix_c = x_main
    box(ax, mix_x, y_mix_top - mix_h, mix_w, mix_h,
        'IMM Mixing',
        r'$\mu_{\mathrm{flare}} = P(\mathrm{Pre\text{-}Flare}\,|\,\mathbf{z})$',
        fs=9, facecolor=LIGHT)

    # ---------- Lane 5: Physics + Stats (two boxes, centered) ----------
    y_phys_top = 1.95
    phys_h = 0.75
    n_w = 2.4
    n_gap = 0.6
    n_x = x_main - n_gap / 2 - n_w
    m_x = x_main + n_gap / 2
    n_c = n_x + n_w / 2
    m_c = m_x + n_w / 2
    box(ax, n_x, y_phys_top - phys_h, n_w, phys_h,
        'Neupert Residual',
        r'$D = \mathrm{SXR} - \mathrm{SXR}_{bg} - \alpha I$',
        fs=9, sub_fs=8)
    box(ax, m_x, y_phys_top - phys_h, n_w, phys_h,
        'Mahalanobis Gate',
        r'$d^2 = \mathbf{r}^T \mathbf{S}^{-1} \mathbf{r}$  (3$\sigma$)',
        fs=9, sub_fs=8)
    ax.text(0.20, y_phys_top - phys_h / 2, 'PHYSICS', ha='center', va='center',
            rotation=90, fontsize=8, color=GRAY, fontweight='bold')

    # ---------- Lane 6: Decision ----------
    y_dec_top = 0.95
    dec_h = 0.75
    box(ax, pp_x, y_dec_top - dec_h, pp_w, dec_h,
        'Adaptive Confirmation  +  Alert',
        r'$N \in \{2,3,4,6\}$ steps  |  LED + Web + JSON  |  NASA DONKI cross-check',
        fs=9, sub_fs=7, facecolor=LIGHT)
    ax.text(0.20, y_dec_top - dec_h / 2, 'DECIDE', ha='center', va='center',
            rotation=90, fontsize=8, color=GRAY, fontweight='bold')

    # ---------- ALERT output ----------
    ax.text(x_main, 0.05, 'ALERT', ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.30', facecolor=WHITE,
                      edgecolor=BLACK, linewidth=1.4))

    # ---------- Support column (right) ----------
    ax.text(sup_center, 7.50, 'Support Modules',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=DARK)

    # Align support modules vertically with the main boxes they advise
    # RLS alpha estimator: aligns with UKF (y_ukf_top - ukf_h/2)
    rls_y = y_ukf_top - ukf_h / 2
    rls_h = 0.90
    box(ax, sup_left, rls_y - rls_h / 2, sup_w, rls_h,
        r'RLS $\alpha$ Estimator',
        'Two-timescale\nBayesian prior', fs=9, sub_fs=7,
        facecolor=PALE)

    # CBR layer: aligns with Mixing/Decision boundary (between y_mix_top and y_dec_top)
    cbr_y = (y_mix_top - mix_h + y_dec_top) / 2
    cbr_h = 1.20
    box(ax, sup_left, cbr_y - cbr_h / 2, sup_w, cbr_h,
        'CBR Experience Layer',
        '15 cases\n14-D signature\nGaussian-kernel vote',
        fs=9, sub_fs=7, facecolor=PALE)

    # Defensive layers: align with Preprocessing
    dfn_y = y_pp_top - pp_h / 2
    dfn_h = 0.90
    box(ax, sup_left, dfn_y - dfn_h / 2, sup_w, dfn_h,
        'Defensive Layers',
        r'$\chi^2$ health monitor' + '\nPredict-only fallback',
        fs=9, sub_fs=7, facecolor=PALE)

    # ---------- Main pipeline arrows (strictly vertical) ----------
    # SoLEXS / HEL1OS -> Preprocess: both converge to pp_c
    # Use two short verticals + a horizontal joiner to avoid diagonal lines
    pp_top = y_pp_top           # top of preprocessing box
    in_bot = y_in_top - in_h    # bottom of input boxes
    join_y = (pp_top + in_bot) / 2  # midline for horizontal joiner
    # SoLEXS down to join
    ax.plot([sxr_c, sxr_c], [in_bot, join_y], color=BLACK, lw=1.3, zorder=1)
    # HEL1OS down to join
    ax.plot([hxr_c, hxr_c], [in_bot, join_y], color=BLACK, lw=1.3, zorder=1)
    # Horizontal joiner
    ax.plot([sxr_c, hxr_c], [join_y, join_y], color=BLACK, lw=1.3, zorder=1)
    # Down from joiner center to preprocessing top
    varrow(ax, pp_c, join_y, pp_top)

    # Preprocess -> UKF
    varrow(ax, pp_c, y_pp_top - pp_h, y_ukf_top)

    # UKF -> Quiet Sun and Pre-Flare: split using a T-junction
    ukf_bot = y_ukf_top - ukf_h
    imm_top = y_imm_top
    split_y = (ukf_bot + imm_top) / 2
    # Down from UKF center to split
    ax.plot([pp_c, pp_c], [ukf_bot, split_y], color=BLACK, lw=1.3, zorder=1)
    # Horizontal split
    ax.plot([q_c, p_c], [split_y, split_y], color=BLACK, lw=1.3, zorder=1)
    # Down from q_c and p_c to model box tops
    varrow(ax, q_c, split_y, imm_top)
    varrow(ax, p_c, split_y, imm_top)

    # Quiet Sun / Pre-Flare -> IMM Mixing: converge with another T-junction
    imm_bot = y_imm_top - imm_h
    mix_top = y_mix_top
    join_y2 = (imm_bot + mix_top) / 2
    ax.plot([q_c, q_c], [imm_bot, join_y2], color=BLACK, lw=1.3, zorder=1)
    ax.plot([p_c, p_c], [imm_bot, join_y2], color=BLACK, lw=1.3, zorder=1)
    ax.plot([q_c, p_c], [join_y2, join_y2], color=BLACK, lw=1.3, zorder=1)
    varrow(ax, mix_c, join_y2, mix_top)

    # IMM Mixing -> Neupert and Mahalanobis: split
    mix_bot = y_mix_top - mix_h
    phys_top = y_phys_top
    split_y2 = (mix_bot + phys_top) / 2
    ax.plot([mix_c, mix_c], [mix_bot, split_y2], color=BLACK, lw=1.3, zorder=1)
    ax.plot([n_c, m_c], [split_y2, split_y2], color=BLACK, lw=1.3, zorder=1)
    varrow(ax, n_c, split_y2, phys_top)
    varrow(ax, m_c, split_y2, phys_top)

    # Neupert / Mahalanobis -> Decision: converge
    phys_bot = y_phys_top - phys_h
    dec_top = y_dec_top
    join_y3 = (phys_bot + dec_top) / 2
    ax.plot([n_c, n_c], [phys_bot, join_y3], color=BLACK, lw=1.3, zorder=1)
    ax.plot([m_c, m_c], [phys_bot, join_y3], color=BLACK, lw=1.3, zorder=1)
    ax.plot([n_c, m_c], [join_y3, join_y3], color=BLACK, lw=1.3, zorder=1)
    varrow(ax, pp_c, join_y3, dec_top)

    # Decision -> ALERT
    varrow(ax, pp_c, y_dec_top - dec_h, 0.30, color=DARK)

    # ---------- Support module arrows (horizontal dashed) ----------
    # RLS alpha -> UKF (horizontal arrow from left edge of RLS to right edge of UKF)
    harrow(ax, sup_left, rls_y, main_right, rls_y, color=GRAY, lw=1.0, dashed=True)
    # CBR -> IMM Mixing (horizontal arrow)
    harrow(ax, sup_left, cbr_y, mix_x + mix_w, cbr_y,
           color=GRAY, lw=1.0, dashed=True)
    # CBR -> Decision (horizontal arrow, lower)
    harrow(ax, sup_left, cbr_y - cbr_h / 2 + 0.05, main_right,
           y_dec_top - dec_h / 2,
           color=GRAY, lw=1.0, dashed=True)
    # Defensive -> Preprocessing (horizontal arrow)
    harrow(ax, sup_left, dfn_y, main_right, dfn_y,
           color=GRAY, lw=1.0, dashed=True)

    # ---------- Legend ----------
    leg_y = 7.20
    ax.plot([7.3, 7.7], [leg_y, leg_y], color=BLACK, lw=1.3)
    ax.text(7.75, leg_y, 'Main data flow', fontsize=7, va='center')
    ax.plot([8.7, 9.1], [leg_y, leg_y], color=GRAY, lw=1.0,
            linestyle=(0, (4, 3)))
    ax.text(9.15, leg_y, 'Advisory feedback', fontsize=7, va='center',
            color=GRAY)

    plt.tight_layout(pad=0.3)
    plt.savefig(f'{FIG_DIR}/fig1_architecture.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print('OK Figure 1: clean architecture diagram regenerated')


if __name__ == '__main__':
    create_architecture()
