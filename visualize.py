"""
visualize.py  ──  数据可视化模块
─────────────────────────────────────────
生成 6 张高质量分析图表，适合答辩展示
依赖: matplotlib, seaborn, pandas, numpy
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from matplotlib.font_manager import FontProperties

# ── 中文字体 ────────────────────────────────────────────────
FONT_PATH = '/System/Library/Fonts/STHeiti Medium.ttc'
ZH_FONT = FontProperties(fname=FONT_PATH)
ZH_FONT_SM = FontProperties(fname=FONT_PATH, size=10)
ZH_FONT_LG = FontProperties(fname=FONT_PATH, size=14)
ZH_FONT_TITLE = FontProperties(fname=FONT_PATH, size=16)

plt.rcParams['font.family'] = 'sans-serif'

# ── 配色方案 ────────────────────────────────────────────────
C_STROKE = '#DC3545'    # 中风红
C_NOSTROKE = '#457B9D'  # 未中风蓝
C_PALETTE = ['#457B9D', '#DC3545', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']

OUTPUT_DIR = 'plots'


def load_data(csv_path='中风预测_预处理后.csv'):
    """加载预处理后数据，还原数值类型。"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['是否中风'] = df['是否中风'].map({'是': 1, '否': 0})
    return df


# ═══════════════════════════════════════════════════════════════
# 图 1：年龄分布 × 中风
# ═══════════════════════════════════════════════════════════════
def plot_age_distribution(df, save_path='plots/01_年龄分布.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 直方图
    ax = axes[0]
    for label, val, color in [('未中风', 0, C_NOSTROKE), ('中风', 1, C_STROKE)]:
        subset = df[df['是否中风'] == val]['年龄']
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor='white')
    ax.set_xlabel('年龄（岁）', fontproperties=ZH_FONT_SM)
    ax.set_ylabel('人数', fontproperties=ZH_FONT_SM)
    ax.set_title('年龄分布直方图', fontproperties=ZH_FONT_LG, fontweight='bold')
    ax.legend(prop=ZH_FONT_SM)

    # 右: KDE
    ax = axes[1]
    for label, val, color in [('未中风', 0, C_NOSTROKE), ('中风', 1, C_STROKE)]:
        subset = df[df['是否中风'] == val]['年龄']
        subset.plot.kde(ax=ax, label=label, color=color, linewidth=2)
        ax.axvline(subset.median(), color=color, linestyle='--', alpha=0.5)
    ax.set_xlabel('年龄（岁）', fontproperties=ZH_FONT_SM)
    ax.set_title('年龄分布密度曲线', fontproperties=ZH_FONT_LG, fontweight='bold')
    ax.legend(prop=ZH_FONT_SM)

    fig.suptitle('图1：年龄分布与中风关系', fontproperties=ZH_FONT_TITLE, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')


# ═══════════════════════════════════════════════════════════════
# 图 2：关键分类变量中风率对比
# ═══════════════════════════════════════════════════════════════
def plot_category_rates(df, save_path='plots/02_分类变量中风率.png'):
    from task3 import add_age_group, add_bmi_group

    df_e = add_age_group(df)
    df_e = add_bmi_group(df_e)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    categories = [
        ('年龄段', ['未成年(<18)', '青年(18~40)', '中年(40~60)', '老年(≥60)'],
         axes[0, 0], '年龄组中风率'),
        ('心脏病', [0, 1], axes[0, 1], '心脏病与中风率'),
        ('高血压', [0, 1], axes[1, 0], '高血压与中风率'),
        ('吸烟', ['不吸', '吸烟', '已戒'], axes[1, 1], '吸烟状态与中风率'),
    ]

    name_map = {0: '无', 1: '有'}
    for col, order, ax, title in categories:
        rates = []
        labels_disp = []
        for i, val in enumerate(order):
            subset = df_e[df_e[col] == val]
            rate = subset['是否中风'].mean() * 100
            rates.append(rate)
            labels_disp.append(name_map.get(val, str(val)))

        colors = sns.color_palette('RdBu_r', len(order))
        bars = ax.bar(range(len(order)), rates, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels_disp, fontproperties=ZH_FONT_SM)
        ax.set_ylabel('中风率 (%)', fontproperties=ZH_FONT_SM)
        ax.set_title(title, fontproperties=ZH_FONT_LG, fontweight='bold')

        for i, (bar, rate) in enumerate(zip(bars, rates)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', fontsize=9, fontproperties=ZH_FONT_SM)

    fig.suptitle('图2：关键分类变量中风率对比', fontproperties=ZH_FONT_TITLE, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')


# ═══════════════════════════════════════════════════════════════
# 图 3：χ² 统计量排序
# ═══════════════════════════════════════════════════════════════
def plot_chi2_ranking(df, save_path='plots/03_卡方值排序.png'):
    from scipy.stats import chi2_contingency

    chi2_cols = ['年龄段', '心脏病', '高血压', '曾婚', '职业', '吸烟', '居住类型', '性别',
                 '血糖分组', 'BMI分组']
    results = {}
    for col in chi2_cols:
        if col not in df.columns:
            # 需要添加分组列
            continue
        crosstab = pd.crosstab(df[col], df['是否中风'])
        chi2, p, _, _ = chi2_contingency(crosstab)
        results[col] = chi2

    # 排序
    sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    labels = []
    values = []
    for k, v in sorted_items:
        if v > 3.84:  # 临界值 (p<0.05, df=1)
            labels.append(f'{k} ★')
        else:
            labels.append(f'{k}')
        values.append(v)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#DC3545' if v > 30 else '#457B9D' if v > 10 else '#A8DADC' for v in values]
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontproperties=ZH_FONT_SM)
    ax.set_xlabel('χ² 值', fontproperties=ZH_FONT_SM)
    ax.set_title('图3：各变量 χ² 统计量排序（★ = p<0.05）', fontproperties=ZH_FONT_LG, fontweight='bold')
    ax.axvline(3.84, color='gray', linestyle='--', alpha=0.5, label='p=0.05 临界值')
    ax.legend(prop=ZH_FONT_SM)

    for i, v in enumerate(values):
        ax.text(v + 1, i, f'{v:.1f}', va='center', fontsize=9, fontproperties=ZH_FONT_SM)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')


# ═══════════════════════════════════════════════════════════════
# 图 4：数值变量箱线图
# ═══════════════════════════════════════════════════════════════
def plot_boxplots(df, save_path='plots/04_数值变量箱线图.png'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (col, title) in enumerate([('年龄', '年龄'), ('血糖', '血糖水平'), ('BMI', 'BMI')]):
        ax = axes[i]
        bp = ax.boxplot(
            [df[df['是否中风'] == 0][col], df[df['是否中风'] == 1][col]],
            labels=['未中风', '中风'],
            patch_artist=True,
            widths=0.5,
        )
        bp['boxes'][0].set_facecolor(C_NOSTROKE)
        bp['boxes'][1].set_facecolor(C_STROKE)
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax.set_title(title, fontproperties=ZH_FONT_LG, fontweight='bold')
        ax.set_ylabel(title, fontproperties=ZH_FONT_SM)
        for label in ax.get_xticklabels():
            label.set_fontproperties(ZH_FONT_SM)

        # 添加均值标注
        for j, val in enumerate([0, 1]):
            m = df[df['是否中风'] == val][col].mean()
            ax.text(j + 1, m, f'{m:.1f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    fig.suptitle('图4：数值变量箱线图（中风 vs 未中风）', fontproperties=ZH_FONT_TITLE,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')


# ═══════════════════════════════════════════════════════════════
# 图 5：相关性热力图
# ═══════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df, save_path='plots/05_相关性热力图.png'):
    corr_cols = ['年龄', '高血压', '心脏病', '血糖', 'BMI', '是否中风']
    corr_df = df[corr_cols].copy()

    # 计算相关系数矩阵
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8},
                ax=ax)
    ax.set_title('图5：关键变量相关系数矩阵', fontproperties=ZH_FONT_TITLE,
                 fontweight='bold', pad=15)

    # 设置中文标签
    labels_cn = corr_cols
    ax.set_xticklabels(labels_cn, fontproperties=ZH_FONT_SM, rotation=30, ha='right')
    ax.set_yticklabels(labels_cn, fontproperties=ZH_FONT_SM, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')


# ═══════════════════════════════════════════════════════════════
# 图 6：中风率随年龄变化趋势
# ═══════════════════════════════════════════════════════════════
def plot_age_trend(df, save_path='plots/06_年龄趋势.png'):
    # 按 5 岁分组计算中风率
    bins = np.arange(0, 85, 5)
    df_tmp = df.copy()
    df_tmp['年龄组'] = pd.cut(df_tmp['年龄'], bins=bins, right=False)
    age_rate = df_tmp.groupby('年龄组', observed=True).agg(
        中风率=('是否中风', 'mean'),
        样本数=('是否中风', 'count')
    ).reset_index()
    age_rate['年龄中点'] = age_rate['年龄组'].apply(lambda x: x.mid)
    age_rate['中风率_pct'] = age_rate['中风率'] * 100

    fig, ax = plt.subplots(figsize=(12, 5))

    # 柱状图 + 趋势线
    bars = ax.bar(age_rate['年龄中点'], age_rate['中风率_pct'],
                   width=4, alpha=0.6, color=C_STROKE, edgecolor='white',
                   label='各年龄段中风率')
    ax.plot(age_rate['年龄中点'], age_rate['中风率_pct'],
            'o-', color='#1D3557', linewidth=2, markersize=6, label='趋势线')

    ax.set_xlabel('年龄（岁）', fontproperties=ZH_FONT_SM)
    ax.set_ylabel('中风率 (%)', fontproperties=ZH_FONT_SM)
    ax.set_title('图6：中风率随年龄变化趋势（5岁一组）', fontproperties=ZH_FONT_LG, fontweight='bold')
    ax.legend(prop=ZH_FONT_SM)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════
def generate_all(csv_path='中风预测_预处理后.csv'):
    """生成全部 6 张图表。"""
    print('\n' + '═' * 50)
    print('  数据可视化 — 生成图表')
    print('═' * 50)

    df = load_data(csv_path)

    # 为部分图表添加分组列
    from task3 import add_age_group, add_glucose_group, add_bmi_group
    df_e = add_age_group(df)
    df_e = add_glucose_group(df_e)
    df_e = add_bmi_group(df_e)

    plot_age_distribution(df)
    plot_category_rates(df)
    plot_chi2_ranking(df_e)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_age_trend(df)

    print(f'\n  ✅ 全部图表已保存至 {OUTPUT_DIR}/ 目录')


if __name__ == '__main__':
    generate_all()
