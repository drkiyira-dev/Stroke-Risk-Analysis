"""
task3.py  ──  任务3：数据探索 —— 猜想验证
─────────────────────────────────────────────────────────────
猜想1：年龄越大，中风比例越高？      → pd.cut 年龄分组 + groupby
猜想2：有心脏病的人，中风率更高？    → .loc[] 筛选 + groupby + mean
猜想3：吸烟的人中风率更高？          → .loc[] 过滤"未知" + groupby + describe
附加探索：血糖高低分组、BMI 胖瘦分组 各组中风率

必须覆盖知识点：
  .loc[]  /  .groupby()  /  .describe() / .mean() / .count()
  数值运算（年龄分组、血糖高低分组、BMI 胖瘦分组）
"""

import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 数值分组辅助函数
# ═══════════════════════════════════════════════════════════════

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """年龄切为4段：未成年 / 青年 / 中年 / 老年"""
    bins   = [0,  18,  40,  60, 120]
    labels = ['未成年(<18)', '青年(18~40)', '中年(40~60)', '老年(≥60)']
    df = df.copy()
    df['年龄段'] = pd.cut(df['年龄'], bins=bins, labels=labels, right=False)
    return df


def add_glucose_group(df: pd.DataFrame) -> pd.DataFrame:
    """血糖高低分组：以 125 mg/dL 为界（接近糖尿病诊断参考线）"""
    df = df.copy()
    df['血糖分组'] = (df['血糖'] >= 125).map({True: '偏高(≥125)',
                                              False: '正常(<125)'})
    return df


def add_bmi_group(df: pd.DataFrame) -> pd.DataFrame:
    """BMI 分组（WHO 标准）"""
    bins   = [0, 18.5, 25.0, 30.0, float('inf')]
    labels = ['偏瘦(<18.5)', '正常(18.5~<25)', '超重(25~<30)', '肥胖(≥30)']
    df = df.copy()
    df['BMI分组'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=False)
    return df


# ═══════════════════════════════════════════════════════════════
# 猜想验证函数
# ═══════════════════════════════════════════════════════════════

def hypothesis_1_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    猜想1：年龄越大，中风比例越高？
    知识点：pd.cut 数值分组 + .groupby() + .mean() + .count()
    """
    print('\n' + '─' * 50)
    print('  猜想1：年龄越大，中风比例越高？')
    print('─' * 50)

    result = (
        df.groupby('年龄段', observed=True)['是否中风']
        .agg(样本数='count', 中风人数='sum', 中风率='mean')
        .reset_index()
    )
    result['中风率'] = result['中风率'].map('{:.2%}'.format)
    print(result.to_string(index=False))

    # 额外用 .loc[] 精确查看老年组数据（与分组 bins=[60,120) 保持一致，含60岁）
    elder_mask    = df['年龄'] >= 60
    elder_stroke  = df.loc[elder_mask, '是否中风'].mean()
    elder_count   = df.loc[elder_mask, '是否中风'].count()
    print(f'\n  → 用 .loc[] 验证：老年(≥60岁) 中风率 = {elder_stroke:.2%}，共 {elder_count} 人')

    return result


def hypothesis_2_heart_disease(df: pd.DataFrame) -> pd.DataFrame:
    """
    猜想2：有心脏病的人，中风率更高？
    知识点：.loc[] 条件筛选 + .groupby() + .mean() / .count()
    """
    print('\n' + '─' * 50)
    print('  猜想2：有心脏病的人，中风率更高？')
    print('─' * 50)

    result = (
        df.groupby('心脏病')['是否中风']
        .agg(样本数='count', 中风人数='sum', 中风率='mean')
        .reset_index()
    )
    result['心脏病'] = result['心脏病'].map({1: '有心脏病', 0: '无心脏病'})
    result['中风率'] = result['中风率'].map('{:.2%}'.format)
    print(result.to_string(index=False))

    rate_yes  = df.loc[df['心脏病'] == 1, '是否中风'].mean()
    count_yes = df.loc[df['心脏病'] == 1, '是否中风'].count()
    rate_no   = df.loc[df['心脏病'] == 0, '是否中风'].mean()
    count_no  = df.loc[df['心脏病'] == 0, '是否中风'].count()
    print(f'\n  → 有心脏病：中风率 {rate_yes:.2%}（n = {count_yes}）')
    print(f'  → 无心脏病：中风率 {rate_no:.2%}（n = {count_no}）')
    print(f'  → 有心脏病者中风率是无心脏病者的 {rate_yes/rate_no:.1f} 倍')

    return result


def hypothesis_3_smoking(df: pd.DataFrame) -> pd.DataFrame:
    """
    猜想3：吸烟的人中风率更高？
    知识点：.loc[] 过滤"未知" + .groupby() + .describe() + .mean()
    """
    print('\n' + '─' * 50)
    print('  猜想3：吸烟的人中风率更高？')
    print('─' * 50)

    unknown_n = (df['吸烟'] == '未知').sum()
    df_smoke  = df.loc[df['吸烟'] != '未知'].copy()
    print(f'  注：已用 .loc[] 过滤掉 {unknown_n} 条"未知"记录，剩余 {len(df_smoke)} 条')

    print('\n  吸烟列描述统计（.describe()）：')
    print(df_smoke['吸烟'].describe().to_string())

    result = (
        df_smoke.groupby('吸烟')['是否中风']
        .agg(样本数='count', 中风人数='sum', 中风率='mean')
        .reset_index()
    )
    result['中风率'] = result['中风率'].map('{:.2%}'.format)
    print(f'\n  吸烟分组 vs 中风率：')
    print(result.to_string(index=False))

    smokes   = df_smoke.loc[df_smoke['吸烟'] == '吸烟', '是否中风'].mean()
    quit_smk = df_smoke.loc[df_smoke['吸烟'] == '已戒', '是否中风'].mean()
    no_smk   = df_smoke.loc[df_smoke['吸烟'] == '不吸', '是否中风'].mean()
    print(f'\n  → 当前吸烟：{smokes:.2%}  已戒烟：{quit_smk:.2%}  从不吸烟：{no_smk:.2%}')
    print(f'  → 已戒烟者({quit_smk:.2%})高于当前吸烟者({smokes:.2%})，')
    print(f'    可能原因：已戒者年龄偏大，年龄是更强的混淆变量')

    # ── 被过滤的"未知"组分析 ──
    df_unknown = df.loc[df['吸烟'] == '未知']
    unknown_stroke = df_unknown['是否中风'].mean()
    unknown_cnt = len(df_unknown)
    unknown_stroke_n = int(df_unknown['是否中风'].sum())
    print(f'\n  → 被过滤的"未知"组（{unknown_cnt} 人，占全部 {unknown_cnt/len(df)*100:.1f}%）：')
    print(f'    中风率 {unknown_stroke:.2%}（{unknown_stroke_n} 人），平均年龄 {df_unknown["年龄"].mean():.1f} 岁')

    # ── 年龄分层 × 吸烟交叉验证 ──
    print(f'\n  年龄分层 × 吸烟 交叉分析（控制年龄混淆）：')
    smoke_age_mean = (
        df_smoke.groupby('吸烟')['年龄']
        .agg(平均年龄='mean')
        .reset_index()
    )
    smoke_age_mean['平均年龄'] = smoke_age_mean['平均年龄'].round(1)
    print(smoke_age_mean.to_string(index=False))

    smoke_age_cross = (
        df_smoke.groupby(['年龄段', '吸烟'], observed=True)['是否中风']
        .agg(样本数='count', 中风率='mean')
        .reset_index()
    )
    smoke_age_cross['中风率'] = smoke_age_cross['中风率'].map('{:.2%}'.format)
    print('\n  各年龄段 × 吸烟状态 中风率：')
    print(smoke_age_cross.to_string(index=False))
    print('  → 年龄分层后可见：在同一年龄段内，吸烟/已戒的中风率差异远小于年龄差异')

    return result


def extra_exploration(df: pd.DataFrame) -> dict:
    """
    附加探索：血糖高低分组、BMI 胖瘦分组 各组中风率
    知识点：数值运算（血糖高低分组 / BMI 胖瘦分组）+ .groupby() + .count() / .mean()
    """
    print('\n' + '─' * 50)
    print('  附加探索：血糖分组 / BMI分组 与中风率')
    print('─' * 50)

    glu_tbl = (
        df.groupby('血糖分组')['是否中风']
        .agg(样本数='count', 中风人数='sum', 中风率='mean')
        .reset_index()
    )
    glu_tbl['中风率'] = glu_tbl['中风率'].map('{:.2%}'.format)
    print('\n  血糖高低分组 vs 中风率：')
    print(glu_tbl.to_string(index=False))

    bmi_order = ['偏瘦(<18.5)', '正常(18.5~<25)', '超重(25~<30)', '肥胖(≥30)']
    bmi_tbl = (
        df.groupby('BMI分组')['是否中风']
        .agg(样本数='count', 中风人数='sum', 中风率='mean')
        .reindex(bmi_order)
        .reset_index()
    )
    bmi_tbl['中风率'] = bmi_tbl['中风率'].map('{:.2%}'.format)
    print('\n  BMI 胖瘦分组 vs 中风率：')
    print(bmi_tbl.to_string(index=False))

    return {'glucose_table': glu_tbl, 'bmi_table': bmi_tbl}


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总统计表：关键数值列描述统计
    知识点：.describe()
    """
    print('\n' + '─' * 50)
    print('  汇总统计表（关键数值列）')
    print('─' * 50)
    cols    = ['年龄', '血糖', 'BMI', '是否中风']
    summary = df[cols].describe().round(2)
    print(summary.to_string())
    return summary


# ═══════════════════════════════════════════════════════════════
# 任务3 主入口
# ═══════════════════════════════════════════════════════════════

def run(df: pd.DataFrame) -> dict:
    print('\n' + '═' * 50)
    print('  任务3：数据探索 —— 猜想验证')
    print('═' * 50)

    # 添加分组列（年龄 / 血糖 / BMI 三种数值分组）
    df = add_age_group(df)
    df = add_glucose_group(df)
    df = add_bmi_group(df)

    # 三个猜想
    r1 = hypothesis_1_age(df)
    r2 = hypothesis_2_heart_disease(df)
    r3 = hypothesis_3_smoking(df)

    # 附加探索（血糖分组 / BMI分组）
    extra = extra_exploration(df)

    # 汇总统计
    s = summary_table(df)

    return {
        'age_result':     r1,
        'heart_result':   r2,
        'smoking_result': r3,
        'extra':          extra,
        'summary':        s,
        'df_enriched':    df,
    }
