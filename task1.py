"""
task1.py  ──  任务1：数据读取与基础体检
─────────────────────────────────────────
输出：行列数 / 各列数据类型 / 缺失值数量 / 前5行预览
"""

import pandas as pd


# ──────────────────────────────────────────────────────────────
# 1-1  读取数据
# ──────────────────────────────────────────────────────────────
def load_data(filepath: str = '中风预测.csv') -> pd.DataFrame:
    """读取 CSV 文件，返回原始 DataFrame。"""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


# ──────────────────────────────────────────────────────────────
# 1-2  基础体检
# ──────────────────────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> dict:
    """打印数据行列数、数据类型、缺失值统计，返回体检结果字典。"""

    sep = '─' * 50

    print('\n' + '═' * 50)
    print('  任务1：数据读取与基础体检')
    print('═' * 50)

    # ① 行列数
    print(f'\n【数据维度】\n  {df.shape[0]} 行  ×  {df.shape[1]} 列')

    # ② 各列数据类型
    print(f'\n【各列数据类型】\n{sep}')
    dtype_df = pd.DataFrame({
        '列名':     df.columns,
        '数据类型': df.dtypes.values,
        '非空数量': df.count().values,
    })
    print(dtype_df.to_string(index=False))

    # ③ 缺失值统计
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f'\n【缺失值统计】\n{sep}')
    if missing_cols.empty:
        print('  无缺失值 ✅')
    else:
        miss_df = pd.DataFrame({
            '列名':   missing_cols.index,
            '缺失数': missing_cols.values,
            '缺失率': (missing_cols.values / len(df) * 100).round(2),
        })
        miss_df['缺失率'] = miss_df['缺失率'].map('{:.2f}%'.format)
        print(miss_df.to_string(index=False))

    # ④ 前5行
    print(f'\n【前5行预览】\n{sep}')
    print(df.head().to_string())

    # ⑤ 问题列讨论（小组讨论结论：哪些列有问题？怎么处理？）
    n = len(df)
    n_bmi     = df['bmi'].isna().sum()
    n_unknown = (df['smoking_status'] == 'Unknown').sum()
    n_other   = (df['gender'] == 'Other').sum()
    print(f'\n【问题列讨论】\n{sep}')
    print(f'  ① bmi            缺失 {n_bmi} 个（{n_bmi/n:.2%}）'
          f'→ 比例不大，用中位数填充（任务2处理）')
    print(f'  ② id             仅为流水号，无分析价值 → 直接删除（任务2处理）')
    print(f'  ③ smoking_status 有 {n_unknown} 条 Unknown（{n_unknown/n:.2%}）'
          f'→ 占比过高不能删行，分析吸烟时单独过滤（任务3处理）')
    print(f'  ④ gender         有 {n_other} 条 Other → 样本极少，保留不影响统计')

    return {
        'shape':   df.shape,
        'dtypes':  df.dtypes,
        'missing': missing,
    }


# ──────────────────────────────────────────────────────────────
# 任务1 主入口
# ──────────────────────────────────────────────────────────────
def run(filepath: str = '中风预测.csv') -> pd.DataFrame:
    df = load_data(filepath)
    inspect_data(df)
    return df
