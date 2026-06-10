"""
task2.py  ──  任务2：数据预处理
─────────────────────────────────────────
① 删除无用列 id
② 填充 bmi 缺失值（中位数）
③ 列名改为中文
④ 分类值改为中文（性别/职业/婚姻/居住类型/吸烟）
⑤ 保存预处理后 CSV（是否中风: 0→否 / 1→是）
"""

import pandas as pd


# ──────────────────────────────────────────────────────────────
# 映射表（统一管理，方便修改）
# ──────────────────────────────────────────────────────────────
COLUMNS_MAP: dict = {
    'gender':            '性别',
    'age':               '年龄',
    'hypertension':      '高血压',
    'heart_disease':     '心脏病',
    'ever_married':      '婚姻',
    'work_type':         '职业',
    'Residence_type':    '居住类型',
    'avg_glucose_level': '血糖',
    'bmi':               'BMI',
    'smoking_status':    '吸烟',
    'stroke':            '是否中风',
}

VALUE_MAPS: dict = {
    '性别':     {'Male': '男', 'Female': '女', 'Other': '其他'},
    '职业':     {'Private': '私企', 'Self-employed': '自雇',
                 'Govt_job': '公职', 'children': '儿童',
                 'Never_worked': '未工作'},
    '婚姻':     {'Yes': '已婚', 'No': '未婚'},
    '居住类型': {'Urban': '城镇', 'Rural': '农村'},
    '吸烟':     {'never smoked': '不吸', 'formerly smoked': '已戒',
                 'smokes': '吸烟', 'Unknown': '未知'},
}


# ──────────────────────────────────────────────────────────────
# 各步骤函数
# ──────────────────────────────────────────────────────────────
def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    """删除无意义的 id 列。"""
    df = df.drop(columns=['id'])
    print('  ✅ 已删除无用列 id')
    return df


def fill_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """① BMI 缺失值用中位数填充。"""
    median_val = df['bmi'].median()
    n_missing  = df['bmi'].isna().sum()
    df = df.copy()
    df['bmi'] = df['bmi'].fillna(median_val)
    print(f'  ✅ BMI 缺失值已填充 {n_missing} 个（中位数 = {median_val:.2f}）')
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """③ 英文列名 → 中文列名（共11列）。"""
    df = df.rename(columns=COLUMNS_MAP)
    print('  ✅ 列名已改为中文')
    return df


def map_category_values(df: pd.DataFrame) -> pd.DataFrame:
    """③ 各分类列英文值 → 中文值。"""
    for col, mapping in VALUE_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    print('  ✅ 分类列已改为中文（性别 / 职业 / 婚姻 / 居住类型 / 吸烟）')
    return df


def save_csv(df: pd.DataFrame, path: str = '中风预测_预处理后.csv') -> None:
    """④ 保存预处理后数据（是否中风: 0→否 / 1→是，仅 CSV 展示用）。"""
    df_save = df.copy()
    df_save['是否中风'] = df_save['是否中风'].map({1: '是', 0: '否'})
    df_save.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  📁 预处理后数据已保存：{path}')


# ──────────────────────────────────────────────────────────────
# 任务2 主入口
# ──────────────────────────────────────────────────────────────
def run(df: pd.DataFrame,
        output_path: str = '中风预测_预处理后.csv') -> pd.DataFrame:

    print('\n' + '═' * 50)
    print('  任务2：数据预处理')
    print('═' * 50)

    df = drop_id(df)
    df = fill_bmi(df)
    df = rename_columns(df)
    df = map_category_values(df)
    save_csv(df, output_path)

    print(f'\n  预处理完成：{df.shape[0]} 行 × {df.shape[1]} 列，无缺失值 ✅')
    return df
