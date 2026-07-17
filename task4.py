"""
task4.py  ──  任务4：统计假设检验
─────────────────────────────────────────
① 卡方检验：分类变量 vs 是否中风（年龄组/心脏病/吸烟/高血压/性别/曾婚/职业/居住类型）
② Mann-Whitney U 检验：数值变量 vs 是否中风（年龄/血糖/BMI）
③ 输出统计量和 p 值
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal


def chi2_test(df: pd.DataFrame, col: str, target: str = '是否中风') -> dict:
    """对分类列做卡方独立性检验。"""
    crosstab = pd.crosstab(df[col], df[target])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    # 检查是否有期望频数 < 5 的格子
    low_exp = (expected < 5).sum()
    warning = ''
    if low_exp > 0:
        warning = f'（注意：{low_exp} 个格子期望频数<5，结果仅供参考）'
    return {'检验': f'{col} × {target}',
            '方法': '卡方检验',
            'χ²': f'{chi2:.2f}',
            'p值': f'{p:.4f}',
            'p值_浮点': p,
            '自由度': dof,
            '警告': warning}


def mw_test(df: pd.DataFrame, col: str, target: str = '是否中风') -> dict:
    """对数值列 vs 二分类目标做 Mann-Whitney U 检验。"""
    group0 = df.loc[df[target] == 0, col].dropna()
    group1 = df.loc[df[target] == 1, col].dropna()
    stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
    mean0 = group0.mean()
    mean1 = group1.mean()
    return {'检验': f'{col} × {target}',
            '方法': 'Mann-Whitney U',
            'U统计量': f'{stat:.0f}',
            'p值': f'{p:.6f}',
            'p值_浮点': p,
            '未中风均值': f'{mean0:.2f}',
            '中风均值': f'{mean1:.2f}'}


def format_p(p_val: float) -> str:
    """格式化 p 值为可读形式。"""
    if p_val < 0.001:
        return 'p < 0.001 ***'
    elif p_val < 0.01:
        return f'p = {p_val:.4f} **'
    elif p_val < 0.05:
        return f'p = {p_val:.4f} *'
    else:
        return f'p = {p_val:.4f} (不显著)'


def run(df: pd.DataFrame) -> dict:
    """执行全部统计检验，返回结果字典。"""
    print('\n' + '═' * 50)
    print('  任务4：统计假设检验')
    print('═' * 50)

    results = []

    # ── 卡方检验 ──────────────────────────────────────────
    print('\n【卡方检验结果】')
    print('─' * 50)

    chi2_cols = ['年龄段', '心脏病', '高血压', '性别', '曾婚', '职业', '居住类型', '吸烟']
    for col in chi2_cols:
        if col in df.columns:
            r = chi2_test(df, col)
            results.append(r)
            sig = format_p(r['p值_浮点'])
            print(f'  {r["检验"]}:  χ²={r["χ²"]},  {sig}  {r["警告"]}')

    # ── Mann-Whitney U ────────────────────────────────────
    print('\n【Mann-Whitney U 检验结果】')
    print('─' * 50)

    mw_cols = ['年龄', '血糖', 'BMI']
    for col in mw_cols:
        if col in df.columns:
            r = mw_test(df, col)
            results.append(r)
            sig = format_p(r['p值_浮点'])
            print(f'  {r["检验"]}:  {sig}')
            print(f'    未中风组均值={r["未中风均值"]}, 中风组均值={r["中风均值"]}')

    # ── 汇总表 ─────────────────────────────────────────────
    print('\n【统计检验汇总】')
    print('─' * 50)
    tbl = pd.DataFrame(results)
    for col in tbl.columns:
        if col.endswith('_浮点'):
            tbl = tbl.drop(columns=[col])
    tbl = tbl.drop(columns=['自由度', '警告'], errors='ignore')
    print(tbl.to_string(index=False))

    return {'stat_results': results, 'stat_table': tbl}
