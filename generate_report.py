"""
generate_report.py  ──  生成 PDF 分析报告（fpdf2 + WenQuanYi 中文字体）
运行前请确保已执行 main.py（生成 中风预测_预处理后.csv）
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from fpdf import FPDF, XPos, YPos

# ── 字体路径 ──────────────────────────────────────────────
FONT_TTF = '/System/Library/Fonts/STHeiti Medium.ttc'

# ── 配色 ──────────────────────────────────────────────────
C_RED     = (220, 53,  69)
C_NAVY    = (29,  53,  87)
C_BLUE    = (69, 123, 157)
C_LIGHT   = (234,240,251)
C_WHITE   = (255,255,255)
C_GRAY    = (100,100,100)
C_LINE    = (187,187,187)
C_BGPAGE  = (250,250,252)


# ══════════════════════════════════════════════════════════
class StrokeReport(FPDF):
    """定制 PDF 报告类"""

    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.add_font('WQY', '', FONT_TTF)
        self.set_margins(22, 22, 22)
        self.set_auto_page_break(True, margin=22)

    # ── 页眉 / 页脚 ────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('WQY', size=8)
        self.set_text_color(*C_GRAY)
        self.cell(0, 8, '中风风险数据探索分析报告', align='L')
        self.cell(0, 8, f'第 {self.page_no()} 页', align='R',
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*C_LINE)
        self.line(22, self.get_y(), 188, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('WQY', size=8)
        self.set_text_color(*C_GRAY)
        self.cell(0, 8, '本报告由 Pandas 数据分析程序自动生成', align='C')

    # ── 辅助方法 ───────────────────────────────────────────
    def h1(self, text: str):
        self.ln(4)
        self.set_fill_color(*C_RED)
        self.rect(22, self.get_y(), 4, 6, 'F')
        self.set_x(28)
        self.set_font('WQY', size=13)
        self.set_text_color(*C_NAVY)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def h2(self, text: str):
        self.set_font('WQY', size=11)
        self.set_text_color(*C_BLUE)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text: str, indent: int = 0):
        self.set_font('WQY', size=10)
        self.set_text_color(50, 50, 50)
        if indent:
            self.set_x(22 + indent)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def note(self, text: str):
        self.set_font('WQY', size=9)
        self.set_text_color(*C_GRAY)
        self.set_x(28)
        self.multi_cell(0, 5.5, text)

    def hline(self, color=C_LINE, thickness=0.4):
        self.set_draw_color(*color)
        self.set_line_width(thickness)
        self.line(22, self.get_y(), 188, self.get_y())
        self.set_line_width(0.2)
        self.ln(3)

    def draw_table(self, df: pd.DataFrame,
                   header_bg=C_NAVY, alt_bg=C_LIGHT):
        """通用表格渲染"""
        cols    = list(df.columns)
        n_cols  = len(cols)
        usable  = 166  # mm
        col_w   = usable / n_cols

        # 表头
        self.set_fill_color(*header_bg)
        self.set_text_color(*C_WHITE)
        self.set_font('WQY', size=9)
        for c in cols:
            self.cell(col_w, 7, str(c), border=1, align='C', fill=True)
        self.ln()

        # 数据行
        self.set_text_color(40, 40, 40)
        self.set_font('WQY', size=9)
        for i, row in df.iterrows():
            fill = (i % 2 == 1)
            if fill:
                self.set_fill_color(*alt_bg)
            for j, val in enumerate(row):
                align = 'L' if j == 0 else 'C'
                self.cell(col_w, 6.5, str(val), border=1,
                          align=align, fill=fill)
            self.ln()
        self.ln(3)


# ══════════════════════════════════════════════════════════
def load_data(csv_path='中风预测_预处理后.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # CSV 中 是否中风 已汉化为 是/否，还原为数值便于统计
    df['是否中风'] = df['是否中风'].map({'是': 1, '否': 0})

    bins   = [0, 18, 40, 60, 120]
    labels = ['未成年(<18)', '青年(18~40)', '中年(40~60)', '老年(≥60)']
    df['年龄段'] = pd.cut(df['年龄'], bins=bins, labels=labels, right=False)

    age_tbl = (df.groupby('年龄段', observed=True)['是否中风']
               .agg(样本数='count', 中风人数='sum', 中风率='mean')
               .reset_index())
    age_tbl['中风率'] = age_tbl['中风率'].map('{:.2%}'.format)

    hd_tbl = (df.groupby('心脏病')['是否中风']
               .agg(样本数='count', 中风人数='sum', 中风率='mean')
               .reset_index())
    hd_tbl['心脏病'] = hd_tbl['心脏病'].map({1: '有心脏病', 0: '无心脏病'})
    hd_tbl['中风率'] = hd_tbl['中风率'].map('{:.2%}'.format)

    df_smoke = df.loc[df['吸烟'] != '未知']
    sm_tbl   = (df_smoke.groupby('吸烟')['是否中风']
                .agg(样本数='count', 中风人数='sum', 中风率='mean')
                .reset_index())
    sm_tbl['中风率'] = sm_tbl['中风率'].map('{:.2%}'.format)

    summary = df[['年龄', '血糖', 'BMI', '是否中风']].describe().round(2).reset_index()
    summary.rename(columns={'index': '统计项'}, inplace=True)

    info = dict(
        total=len(df), cols=df.shape[1] - 1,  # 排除临时分组列 年龄段
        stroke_n=int(df['是否中风'].sum()),
        stroke_rate=df['是否中风'].mean(),
        bmi_median=round(df['BMI'].median(), 2),
        unknown_smoke=int((df['吸烟'] == '未知').sum()),
    )
    return info, age_tbl, hd_tbl, sm_tbl, summary


# ══════════════════════════════════════════════════════════
def generate(output='中风风险分析报告.pdf', csv='中风预测_预处理后.csv'):
    info, age_tbl, hd_tbl, sm_tbl, summary = load_data(csv)

    pdf = StrokeReport()
    pdf.add_page()

    # ── 封面 ──────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_font('WQY', size=22)
    pdf.set_text_color(*C_NAVY)
    pdf.cell(0, 12, '中风风险数据探索分析报告', align='C',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font('WQY', size=11)
    pdf.set_text_color(*C_GRAY)
    pdf.cell(0, 8, 'Pandas 数据分析小组作业', align='C',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.hline(C_RED, 1.5)

    # ── 一、基础体检 ──────────────────────────────────────
    pdf.h1('一、数据基础体检')
    pdf.body(
        f'数据集共 {info["total"]} 条记录，预处理后保留 {info["cols"]} 个字段。'
        f'其中中风患者共 {info["stroke_n"]} 人，整体中风率 {info["stroke_rate"]:.2%}。'
    )
    pdf.h2('预处理步骤：')
    for item in [
        f'BMI 缺失值以中位数 {info["bmi_median"]} 填充',
        '删除无意义列 id',
        '所有列名及分类取值统一汉化（性别 / 职业 / 婚姻 / 居住类型 / 吸烟）',
    ]:
        pdf.note(f'· {item}')
    pdf.ln(3)

    # ── 二、三条结论 ──────────────────────────────────────
    pdf.h1('二、三条核心结论（含数据支撑）')

    # 结论1
    pdf.h2('结论 1  ·  年龄越大，中风比例越高')
    elder_rate = age_tbl.iloc[-1]['中风率']
    young_rate = age_tbl.iloc[0]['中风率']
    pdf.body(
        f'按年龄段分组，老年(>60岁)中风率高达 {elder_rate}，'
        f'未成年(<18岁)仅为 {young_rate}，呈现明显正相关。'
    )
    pdf.draw_table(age_tbl)

    # 结论2
    pdf.h2('结论 2  ·  心脏病显著提升中风风险')
    hd_y = hd_tbl[hd_tbl['心脏病'] == '有心脏病'].iloc[0]
    hd_n = hd_tbl[hd_tbl['心脏病'] == '无心脏病'].iloc[0]
    r_y  = float(hd_y['中风率'].strip('%'))
    r_n  = float(hd_n['中风率'].strip('%'))
    pdf.body(
        f'有心脏病者中风率 {hd_y["中风率"]}，无心脏病者 {hd_n["中风率"]}，'
        f'前者约为后者的 {r_y/r_n:.1f} 倍，提示心血管系统的整体风险联动。'
    )
    pdf.draw_table(hd_tbl, header_bg=C_BLUE)

    # 结论3
    pdf.h2('结论 3  ·  吸烟与中风率的关系')
    smk_row  = sm_tbl[sm_tbl['吸烟'] == '吸烟'].iloc[0]
    no_row   = sm_tbl[sm_tbl['吸烟'] == '不吸'].iloc[0]
    quit_row = sm_tbl[sm_tbl['吸烟'] == '已戒'].iloc[0]
    pdf.body(
        f'过滤掉 {info["unknown_smoke"]} 条"未知"记录后，'
        f'已戒烟者中风率 {quit_row["中风率"]} 反高于当前吸烟者 {smk_row["中风率"]}，'
        f'从不吸烟者最低 {no_row["中风率"]}。'
        f'已戒烟率偏高可能与年龄混淆有关——中老年人更倾向于已戒烟，而年龄本身是更强的中风风险因素。'
    )
    pdf.draw_table(sm_tbl, header_bg=(80, 140, 100),
                   alt_bg=(236, 250, 240))

    # ── 三、汇总统计表 ────────────────────────────────────
    pdf.ln(2)
    pdf.hline()
    pdf.h1('三、关键变量汇总统计表')
    pdf.body('以下为年龄、血糖、BMI、是否中风四列的描述统计（.describe()）：')
    pdf.draw_table(summary, header_bg=(80,120,160),
                   alt_bg=(240,246,255))

    # ── 尾注 ──────────────────────────────────────────────
    pdf.ln(2)
    pdf.hline()
    pdf.note('注：本报告基于中风预测数据集（5110 条记录）分析生成。')

    pdf.output(output)
    print(f'✅ PDF 报告已生成：{output}')


if __name__ == '__main__':
    generate()
