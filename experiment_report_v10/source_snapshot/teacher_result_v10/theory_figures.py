"""Draw explanatory figures from theory, separately from measured result charts."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.text import Text
import numpy as np

BLUE = '#3569a5'
ORANGE = '#c47a32'
GREEN = '#487354'
INK = '#27313c'


def box(ax, x, y, w, h, label, fill='#edf2f7', size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.006',
                              facecolor=fill, edgecolor='#8290a0', linewidth=1))
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=size, color=INK)


def arrow(ax, start, end, color='#6c7887', lw=1.1):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '->', 'lw': lw, 'color': color})


def save(fig, folder, name):
    fig.canvas.draw()
    for label in fig.findobj(Text):
        label.set_fontsize(max(12, label.get_fontsize()*1.25))
    fig.savefig(folder/(name+'.png'), dpi=200, bbox_inches='tight', pad_inches=.08)
    fig.savefig(folder/(name+'.pdf'), bbox_inches='tight', pad_inches=.08,
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def build_theory_figures(folder):
    folder = Path(folder)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    xs = [.07, .34, .62, .90]
    groups = [(['R', 'G', 'B'], [.78, .60, .42]),
              (['h1[0]', 'h1[H−1]'], [.78, .42]),
              (['h2[0]', 'h2[H−1]'], [.78, .42]),
              (['R′', 'G′', 'B′'], [.78, .60, .42])]
    for i in range(3):
        for y1 in groups[i][1]:
            for y2 in groups[i+1][1]:
                arrow(ax, (xs[i]+.045, y1), (xs[i+1]-.045, y2), '#b9c6d4', .75)
    for i, (labels, ys) in enumerate(groups):
        ax.text(xs[i], .94, ['入力3個', '隠れ層H個', '隠れ層H個', '出力3個'][i],
                ha='center', va='center', fontsize=12)
        for label, y in zip(labels, ys):
            ax.add_patch(Ellipse((xs[i], y), .09, .115, facecolor='#edf2f7', edgecolor=BLUE))
            ax.text(xs[i], y, label, ha='center', va='center', fontsize=9)
        if i in [1, 2]:
            ax.text(xs[i], .60, r'$\vdots$', ha='center', va='center', fontsize=19)
    ax.plot([.01, .99], [.33, .33], color='#c8cfd7', lw=.8)
    ax.text(.01, .30, '隠れ層の計算単位1個', fontsize=11, va='top', weight='bold')
    for label, weight, y in [('R', 'wR', .22), ('G', 'wG', .14), ('B', 'wB', .06)]:
        ax.text(.035, y, label, ha='center', va='center', fontsize=11)
        box(ax, .11, y-.032, .14, .064, '× '+weight, size=10)
        arrow(ax, (.055, y), (.10, y))
        arrow(ax, (.26, y), (.39, .14))
    box(ax, .40, .095, .11, .09, '合計', size=11)
    box(ax, .58, .095, .11, .09, '+ b', '#fff0df', 11)
    box(ax, .77, .095, .15, .09, 'ReLU', '#edf4eb', 11)
    arrow(ax, (.52, .14), (.57, .14)); arrow(ax, (.70, .14), (.76, .14))
    arrow(ax, (.93, .14), (.99, .14))
    ax.text(.77, .025, '負なら0　正ならそのまま', ha='center', va='center', fontsize=10)
    fig.subplots_adjust(left=.03, right=.97, top=.98, bottom=.03)
    save(fig, folder, 'theory_network')

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    nodes = [(.015, '学習入力\n135行'), (.27, 'モデル\n重みW・バイアスb'),
             (.525, '予測と正解から\n誤差＋L1を計算'), (.78, '更新方向を求め\nAdamで更新')]
    for x, label in nodes:
        box(ax, x, .70, .20, .14, label, '#edf2f7', 10.5)
    for x in [.22, .475, .73]:
        arrow(ax, (x, .77), (x+.045, .77))
    ax.plot([.88, .88, .37], [.85, .94, .94], color=BLUE, lw=1.1)
    arrow(ax, (.37, .94), (.37, .85), BLUE)
    ax.text(.64, .965, '更新したW・bで繰り返す', ha='center', va='center', fontsize=10, color=BLUE)
    box(ax, .50, .49, .25, .11, '正解RGB', '#edf4eb', 11)
    arrow(ax, (.625, .61), (.625, .69), GREEN)
    ax.text(.02, .37, '色の誤差：予測と正解のずれ', fontsize=11, va='center')
    ax.text(.02, .27, 'L1：重みの絶対値の合計', fontsize=11, va='center')
    ax.text(.02, .17, '両方を小さくするように学習', fontsize=11, va='center', color=BLUE)
    curve = fig.add_axes([.58, .13, .36, .27])
    e = np.linspace(-.15, .15, 501); beta = .05
    loss = np.where(np.abs(e)<beta, e**2/(2*beta), np.abs(e)-beta/2)
    curve.plot(e, loss, color=BLUE, lw=2)
    for threshold in [-beta, beta]:
        curve.axvline(threshold, color=ORANGE, ls='--', lw=1)
    curve.set(xlabel='予測−正解 e', ylabel='損失', title='Smooth L1の形（β＝0.05）')
    curve.tick_params(labelsize=9); curve.grid(alpha=.2)
    curve.xaxis.label.set_size(10); curve.yaxis.label.set_size(10); curve.title.set_size(10)
    fig.subplots_adjust(left=.03, right=.97, top=.98, bottom=.03)
    save(fig, folder, 'theory_learning')

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    before = [[.03, .5, -.02], [-.2, .01, .1]]
    after = [[0, .5, 0], [-.2, 0, .1]]
    for start, matrix, title in [(.055, before, '枝刈り前'), (.42, after, '枝刈り後')]:
        ax.text(start+.12, .93, title, ha='center', va='center', fontsize=12)
        for c in range(3):
            ax.text(start+c*.08+.04, .82, f'列{c}', ha='center', fontsize=9)
        for r in range(2):
            y = .60-r*.21
            ax.text(start-.018, y+.075, f'行{r}', ha='right', va='center', fontsize=9)
            for c in range(3):
                value = matrix[r][c]
                fill = '#fff0df' if matrix is before and abs(value)<.05 else '#edf2f7' if value else '#f3f4f5'
                ax.add_patch(Rectangle((start+c*.08, y), .08, .15, facecolor=fill, edgecolor='#a3aeb9'))
                ax.text(start+c*.08+.04, y+.075, f'{value:g}', ha='center', va='center', fontsize=11,
                        color=INK if value else '#9aa0a7')
    arrow(ax, (.31, .57), (.40, .57), ORANGE)
    ax.text(.355, .76, '|w| < 0.05', ha='center', fontsize=9, color=ORANGE)
    arrow(ax, (.675, .57), (.73, .57), BLUE)
    ax.text(.704, .76, 'CSR', ha='center', fontsize=10, color=BLUE)
    ax.text(.75, .72, 'values = [0.5, −0.2, 0.1]', fontsize=10, va='center')
    ax.text(.75, .56, 'col     = [1, 0, 2]', fontsize=10, va='center')
    ax.text(.75, .40, 'row    = [0, 1, 3]', fontsize=10, va='center')
    ax.text(.75, .20, '行0：valuesの0番\n行1：valuesの1〜2番', fontsize=9.5, va='center')
    ax.text(.35, .13, '説明用の2行3列行列', ha='center', fontsize=10)
    fig.subplots_adjust(left=.03, right=.99, top=.97, bottom=.04)
    save(fig, folder, 'theory_pruning_csr')
