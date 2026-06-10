"""
main.py  ──  中风风险数据探索大挑战  主程序
────────────────────────────────────────────────
运行方式：python main.py
依赖文件：中风预测.csv（放在同目录下）
输出文件：中风预测_预处理后.csv
────────────────────────────────────────────────
模块结构：
  task1.py  →  数据读取与基础体检
  task2.py  →  数据预处理
  task3.py  →  数据探索（猜想验证）
  main.py   →  主入口（调度 + 报告汇总）
"""

import sys
import pandas as pd

import task1
import task2
import task3

# ─────────────── 配置区 ───────────────
INPUT_CSV  = '中风预测.csv'
OUTPUT_CSV = '中风预测_预处理后.csv'
# ──────────────────────────────────────


def print_report(results: dict, df_clean: pd.DataFrame) -> None:
    """打印任务4分析报告摘要。"""
    df_e: pd.DataFrame = results['df_enriched']

    print('\n' + '═' * 50)
    print('  任务4：分析报告摘要')
    print('═' * 50)

    # ─── 基础体检摘要 ─────────────────────────────────
    print('\n【一、基础体检】')
    # 用 df_clean（预处理后11列）报告字段数，排除 task3 临时加的分组列
    print(f'  数据集共 {df_clean.shape[0]} 条记录，{df_clean.shape[1]} 个字段')
    print(f'  中风患者：{df_e["是否中风"].sum()} 人，整体中风率：'
          f'{df_e["是否中风"].mean():.2%}')

    # ─── 三条结论 ─────────────────────────────────────
    print('\n【二、三条核心结论（含数据支撑）】')

    # 结论1：年龄
    age_result = results['age_result']
    oldest_row = age_result.iloc[-1]
    youngest_rate = age_result.iloc[0]['中风率']
    print(f'\n  结论1 ✦ 年龄是中风的强相关因素')
    print(f'    老年(≥60岁) 中风率：{oldest_row["中风率"]}，')
    print(f'    远高于未成年(<18岁)的 {youngest_rate}。')

    # 结论2：心脏病
    hd_result  = results['heart_result']
    hd_yes_row = hd_result[hd_result['心脏病'] == '有心脏病'].iloc[0]
    hd_no_row  = hd_result[hd_result['心脏病'] == '无心脏病'].iloc[0]
    print(f'\n  结论2 ✦ 心脏病显著提升中风风险')
    print(f'    有心脏病者中风率 {hd_yes_row["中风率"]}，'
          f'无心脏病者 {hd_no_row["中风率"]}。')

    # 结论3：吸烟
    sm_result = results['smoking_result']
    smk_row   = sm_result[sm_result['吸烟'] == '吸烟'].iloc[0]
    no_row    = sm_result[sm_result['吸烟'] == '不吸'].iloc[0]
    quit_row  = sm_result[sm_result['吸烟'] == '已戒'].iloc[0]
    print(f'\n  结论3 ✦ 吸烟与中风率的关系（过滤"未知"后）')
    print(f'    已戒烟者中风率 {quit_row["中风率"]} 反高于当前吸烟者 {smk_row["中风率"]}，')
    print(f'    从不吸烟者最低 {no_row["中风率"]}，提示年龄是更强的混淆变量。')

    # ─── 汇总统计表 ─────────────────────────────────
    print('\n【三、汇总统计表】')
    print(results['summary'].to_string())

    print('\n' + '═' * 50)
    print('  ✅ 程序运行完毕')
    print('═' * 50)


def main() -> None:
    print('╔' + '═' * 48 + '╗')
    print('║    🧠  中风风险数据探索大挑战  ·  开始运行    ║')
    print('╚' + '═' * 48 + '╝')

    # ── 任务1：读取 & 体检 ──────────────────────────
    try:
        df_raw = task1.run(INPUT_CSV)
    except FileNotFoundError:
        print(f'\n❌ 找不到文件：{INPUT_CSV}')
        print('   请将数据文件放在同目录下，或修改 main.py 中的 INPUT_CSV 路径。')
        sys.exit(1)

    # ── 任务2：预处理 ────────────────────────────────
    df_clean = task2.run(df_raw, OUTPUT_CSV)

    # ── 任务3：探索 ──────────────────────────────────
    results = task3.run(df_clean)

    # ── 任务4：报告摘要 ──────────────────────────────
    print_report(results, df_clean)


if __name__ == '__main__':
    main()
