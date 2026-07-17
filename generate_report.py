"""
generate_report.py  ──  生成 PDF 分析报告（增强版：含图表 + 统计检验）
─────────────────────────────────────────────────────────────
运行前请确保已执行 main.py（生成 中风预测_预处理后.csv）
运行前请确保已执行 visualize.py（生成 plots/ 目录下的图表）
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from fpdf import FPDF, XPos, YPos

import task3
import task4

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
C_GREEN   = (46, 139, 87)


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
        self.cell(0, 8, '本报告由 Python 数据分析程序自动生成  |  中风预测数据集 5110 人', align='C')

    # ── 辅助方法 ───────────────────────────────────────────
    def h1(self, text: str):
        self.ln(4)
        self.set_fill_color(*C_RED)
        self.rect(22, self.get_y(), 4, 7, 'F')
        self.set_x(28)
        self.set_font('WQY', size=13)
        self.set_text_color(*C_NAVY)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def h2(self, text: str):
        self.set_font('WQY', size=11)
        self.set_text_color(*C_BLUE)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text: str, indent: int = 0):
        self.set_font('WQY', size=10)
        self.set_text_color(50, 50, 50)
        if indent:
            self.set_x(22 + indent)
        self.multi_cell(0, 6, text)
        self.ln(0.5)

    def note(self, text: str):
        self.set_font('WQY', size=9)
        self.set_text_color(*C_GRAY)
        self.set_x(28)
        self.multi_cell(0, 5.5, text)

    def highlight(self, text: str):
        """高亮的重要结论文本"""
        self.set_font('WQY', size=10)
        self.set_text_color(*C_RED)
        self.set_x(28)
        self.multi_cell(0, 6, text)

    def hline(self, color=C_LINE, thickness=0.4):
        self.set_draw_color(*color)
        self.set_line_width(thickness)
        self.line(22, self.get_y(), 188, self.get_y())
        self.set_line_width(0.2)
        self.ln(3)

    def draw_table(self, df: pd.DataFrame,
                   header_bg=C_NAVY, alt_bg=C_LIGHT, col_widths=None):
        """通用表格渲染，支持自定义列宽"""
        cols    = list(df.columns)
        n_cols  = len(cols)
        usable  = 166  # mm
        if col_widths is None:
            col_w = [usable / n_cols] * n_cols
        else:
            total_w = sum(col_widths)
            col_w = [usable * w / total_w for w in col_widths]

        # 表头
        self.set_fill_color(*header_bg)
        self.set_text_color(*C_WHITE)
        self.set_font('WQY', size=9)
        for i, c in enumerate(cols):
            self.cell(col_w[i], 7, str(c), border=1, align='C', fill=True)
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
                self.cell(col_w[j], 6.5, str(val), border=1,
                          align=align, fill=fill)
            self.ln()
        self.ln(3)

    def embed_chart(self, img_path: str, caption: str, w: float = 160):
        """嵌入图表，自动处理页面分页和说明"""
        self.ln(2)
        self.h2(caption)
        self.image(img_path, x=24, w=w)
        self.ln(2)


# ══════════════════════════════════════════════════════════
def load_data(csv_path='中风预测_预处理后.csv'):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['是否中风'] = df['是否中风'].map({'是': 1, '否': 0})
    df = task3.add_age_group(df)

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
        total=len(df), cols=df.shape[1] - 1,
        stroke_n=int(df['是否中风'].sum()),
        stroke_rate=df['是否中风'].mean(),
        bmi_median=round(df['BMI'].median(), 2),
        unknown_smoke=int((df['吸烟'] == '未知').sum()),
    )

    # 加入 task3 的更多分组列，供 task4 使用
    df = task3.add_glucose_group(df)
    df = task3.add_bmi_group(df)

    # 调用 task4 进行统计检验
    stat = task4.run(df)

    return info, age_tbl, hd_tbl, sm_tbl, summary, stat


# ══════════════════════════════════════════════════════════
def generate(output='中风风险分析报告.pdf', csv='中风预测_预处理后.csv'):
    info, age_tbl, hd_tbl, sm_tbl, summary, stat = load_data(csv)
    stat_tbl = stat['stat_table']

    pdf = StrokeReport()
    pdf.add_page()

    # ── 封面 ──────────────────────────────────────────────
    pdf.ln(20)
    pdf.set_font('WQY', size=24)
    pdf.set_text_color(*C_NAVY)
    pdf.cell(0, 14, '中风风险数据探索分析报告', align='C',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font('WQY', size=12)
    pdf.set_text_color(*C_GRAY)
    pdf.cell(0, 8, '基于 5110 条医疗记录的数据分析与统计检验', align='C',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.hline(C_RED, 2)
    pdf.ln(6)

    # 封面摘要
    pdf.h2('报告摘要')
    pdf.body(
        f'本报告对包含 {info["total"]} 条记录的医疗数据集进行了系统分析。'
        f'经过数据清洗（填充 {info["bmi_median"]} 为 BMI 中位数，删除 id 列，统一汉化分类取值），'
        f'得到 {info["cols"]} 个有效分析字段。'
    )
    pdf.body(
        f'数据集共 {info["stroke_n"]} 名中风患者，整体中风率 {info["stroke_rate"]:.2%}。'
        f'通过卡方检验和 Mann-Whitney U 检验，识别出年龄、心脏病、高血压、血糖为中风的最显著风险因素。'
    )

    pdf.ln(3)
    pdf.hline()
    pdf.ln(1)

    # ── 一、数据基础体检 ──────────────────────────────────
    pdf.h1('一、数据基础体检')

    pdf.body(
        f'数据集共 {info["total"]} 条记录，预处理后保留 {info["cols"]} 个字段。'
        f'中风患者共 {info["stroke_n"]} 人，整体中风率 {info["stroke_rate"]:.2%}。'
        f'吸烟状态中"未知"占 {info["unknown_smoke"]} 条（{info["unknown_smoke"]/info["total"]:.1%}），'
        f'分析吸烟时单独过滤处理。'
    )
    pdf.h2('预处理步骤：')
    for item in [
        f'删除无意义列 id',
        f'BMI 缺失值（201 个）以中位数 {info["bmi_median"]} 填充',
        '所有列名及分类取值统一汉化（性别 / 职业 / 曾婚 / 居住类型 / 吸烟）',
        '年龄分组（未成年/青年/中年/老年）、血糖分组（正常/偏高）、BMI 分组（偏瘦/正常/超重/肥胖）',
    ]:
        pdf.note(f'· {item}')
    pdf.ln(3)

    # ── 二、三条核心结论 ──────────────────────────────────
    pdf.h1('二、三条核心结论（含统计检验）')

    # 结论1
    pdf.h2('结论 1 · 年龄越大，中风比例越高')
    elder_rate = age_tbl.iloc[-1]['中风率']
    young_rate = age_tbl.iloc[0]['中风率']
    pdf.body(
        f'按年龄段分组，老年(≥60岁)中风率高达 {elder_rate}，'
        f'未成年(<18岁)仅为 {young_rate}，呈现极显著正相关。'
    )
    pdf.draw_table(age_tbl)
    pdf.highlight(
        f'▶ 卡方检验：χ² = 302.23, p < 0.001，年龄与中风存在极显著的统计学关联。'
        f'未中风组平均年龄 42.0 岁，中风组 67.7 岁（MW-U 检验 p < 0.001）。'
    )
    pdf.ln(2)

    # 结论2
    pdf.h2('结论 2 · 心脏病显著提升中风风险')
    hd_y = hd_tbl[hd_tbl['心脏病'] == '有心脏病'].iloc[0]
    hd_n = hd_tbl[hd_tbl['心脏病'] == '无心脏病'].iloc[0]
    r_y  = float(hd_y['中风率'].strip('%'))
    r_n  = float(hd_n['中风率'].strip('%'))
    pdf.body(
        f'有心脏病者中风率 {hd_y["中风率"]}，无心脏病者 {hd_n["中风率"]}，'
        f'前者约为后者的 {r_y/r_n:.1f} 倍，提示心血管系统的整体风险联动。'
    )
    pdf.draw_table(hd_tbl, header_bg=C_BLUE)
    pdf.highlight(
        f'▶ 卡方检验：χ² = 90.26, p < 0.001，心脏病与中风存在极显著的统计学关联。'
    )
    pdf.ln(2)

    # 结论3
    pdf.h2('结论 3 · 吸烟与中风率的关系')
    smk_row  = sm_tbl[sm_tbl['吸烟'] == '吸烟'].iloc[0]
    no_row   = sm_tbl[sm_tbl['吸烟'] == '不吸'].iloc[0]
    quit_row = sm_tbl[sm_tbl['吸烟'] == '已戒'].iloc[0]
    pdf.body(
        f'过滤掉 {info["unknown_smoke"]} 条"未知"记录后（占 30.2%，平均年龄 30.2 岁，中风率 3.04%），'
        f'已戒烟者中风率 {quit_row["中风率"]} 反高于当前吸烟者 {smk_row["中风率"]}，'
        f'从不吸烟者最低 {no_row["中风率"]}。'
        f'已戒烟组平均年龄 54.9 岁，远高于不吸组（46.7 岁）和当前吸烟组（47.1 岁），'
        f'年龄分层后同年龄段内三组差异大幅缩小，年龄是主要混淆变量。'
    )
    pdf.draw_table(sm_tbl, header_bg=(80, 140, 100),
                    alt_bg=(236, 250, 240))
    pdf.highlight(
        f'▶ 卡方检验：χ² = 29.15, p < 0.001，但年龄分层后效应减弱。'
    )
    pdf.ln(2)

    # ── 三、统计检验 ──────────────────────────────────────
    pdf.h1('三、统计检验结果汇总')

    pdf.body('对全部可用变量做了卡方检验（分类变量）和 Mann-Whitney U 检验（数值变量）：')
    pdf.ln(1)

    # 重排列表格便于展示
    display_cols = ['检验', '方法', 'χ²', 'p值', 'U统计量', '未中风均值', '中风均值']
    disp_tbl = stat_tbl[display_cols].fillna('-')
    pdf.draw_table(disp_tbl, header_bg=(60, 80, 100), alt_bg=(240, 244, 250),
                   col_widths=[3.0, 2.5, 1.5, 1.5, 2.0, 1.8, 1.8])

    pdf.body('关键发现：')
    findings = [
        '年龄 (χ²=302.2)、心脏病 (χ²=90.3)、高血压 (χ²=81.6) 是最显著的中风风险因素',
        '性别 (χ²=0.47, p=0.790) 和居住类型 (χ²=1.08, p=0.298) 无显著相关',
        '曾婚 (χ²=58.9) 和职业类型 (χ²=49.2) 也与中风显著相关，部分受年龄混淆',
        '血糖 (MW-U p<0.001) 和 BMI (MW-U p<0.001) 在中风/未中风组间差异显著',
    ]
    for f in findings:
        pdf.note(f'· {f}')
    pdf.ln(3)

    # ── 四、关键图表 ──────────────────────────────────────
    pdf.h1('四、关键分析图表')

    pdf.embed_chart('plots/01_年龄分布.png', '图 4-1：年龄分布与中风')
    pdf.embed_chart('plots/02_分类变量中风率.png', '图 4-2：关键分类变量中风率对比')
    pdf.embed_chart('plots/03_卡方值排序.png', '图 4-3：各变量卡方统计量排序')
    pdf.embed_chart('plots/04_数值变量箱线图.png', '图 4-4：数值变量箱线图（中风 vs 未中风）')
    pdf.embed_chart('plots/05_相关性热力图.png', '图 4-5：关键变量相关系数矩阵')
    pdf.embed_chart('plots/06_年龄趋势.png', '图 4-6：中风率随年龄变化趋势')

    # ── 五、汇总统计表 ────────────────────────────────────
    pdf.h1('五、关键变量汇总统计表')
    pdf.body('以下为年龄、血糖、BMI、是否中风四列的描述统计（.describe()）：')
    pdf.draw_table(summary, header_bg=(80, 120, 160),
                   alt_bg=(240, 246, 255),
                   col_widths=[1.8, 1.4, 1.4, 1.4, 1.4])

    # ── 尾注 ──────────────────────────────────────────────
    pdf.ln(2)
    pdf.hline()
    pdf.note('注：本报告基于中风预测数据集（5110 条记录）分析生成。统计检验采用 scipy.stats，'
             '图表使用 matplotlib + seaborn 绘制，机器学习模型基于 scikit-learn。')
    pdf.note('风险因素排序：年龄 > 高血压 > 心脏病 > 血糖 > BMI > 曾婚 > 职业 > 吸烟。'
             '性别和居住类型无显著统计学关联。')

    # ── 六、机器学习建模 ──────────────────────────────────
    pdf.add_page()
    pdf.h1('六、机器学习预测建模')

    pdf.body(
        '为验证风险因素的实际预测能力，构建了两种分类模型：逻辑回归和随机森林。'
        '采用 70/30 分层划分训练测试集，使用 class_weight="balanced" 处理类别不平衡问题。'
    )

    pdf.h2('模型评估')
    pdf.body(
        f'逻辑回归  AUC = 0.84（CV: 0.84±0.01），召回率 78.7%——更擅长识别中风高风险人群。'
        f'随机森林  AUC = 0.80（CV: 0.80±0.01），准确率 86.2%——整体分类更准确但召回偏低。'
        f'随机森林特征重要性显示：年龄（44.2%）、血糖（15.9%）、BMI（15.3%）为 Top 3 预测因子，'
        f'与统计检验结论一致。'
    )

    pdf.embed_chart('plots/07_ROC曲线.png', '图 6-1：ROC 曲线对比')
    pdf.embed_chart('plots/08_混淆矩阵.png', '图 6-2：混淆矩阵对比')
    pdf.embed_chart('plots/09_特征重要性.png', '图 6-3：随机森林特征重要性（Top 15）')

    # ── 七、完整特征分析 ──────────────────────────────────
    pdf.add_page()
    pdf.h1('七、完整特征分析')

    pdf.h2('高血压 —— 被低估的风险因素')
    pdf.body(
        '数据中 498 人（9.7%）患有高血压，其中风率 13.86%，为无高血压者（3.92%）的 3.5 倍。'
        '高血压组平均年龄 62.4 岁，无高血压组 41.2 岁，提示高血压与年龄高度共线。'
        '卡方检验 χ²=81.6（p<0.001），关联极为显著，但在年龄分层后效应有所减弱。'
    )

    pdf.h2('血糖 —— 第二重要的数值预测因子')
    pdf.body(
        '以 125 mg/dL 为界，偏高组（1000 人）中风率 10.00%，正常组（4110 人）仅 3.63%。'
        'Mann-Whitney U 检验 p<0.001。血糖偏高者平均年龄也更高（57.0 vs 39.9 岁），'
        '但与年龄相关性弱于高血压（r=0.23）。'
    )

    pdf.h2('曾婚状态 —— 年龄代理变量')
    pdf.body(
        '曾婚组中风率 6.5%，未曾婚组 1.8%（χ²=58.9, p<0.001），但曾婚组平均年龄 52.2 岁'
        '而未曾婚组仅 26.2 岁。年龄分层后两组差异大幅缩小，曾婚实为年龄的代理变量。'
    )

    pdf.h2('性别和居住类型 —— 无显著关联')
    pdf.body(
        '性别（p=0.790）和居住类型（p=0.298）与中风风险无显著统计学关联，'
        '可在预测建模中作为弱特征或直接排除。'
    )

    # ── 八、结论与建议 ──────────────────────────────────
    pdf.add_page()
    pdf.h1('八、结论与建议')
    pdf.ln(2)

    pdf.h2('核心结论')
    conclusions = [
        '1. 年龄是最强的中风预测因子：老年组（≥60岁）中风率 13.15%，是未成年组的 57 倍。'
        '随机森林模型中年龄贡献了 44.2% 的特征重要性。',
        '2. 心血管健康三联征（高血压+心脏病+高血糖）显著增加风险：'
        '三项指标的 χ² 值分别为 81.6、90.3 和统计显著。',
        '3. 吸烟的独立效应有限：已戒烟者中风率（7.9%）高但平均年龄大（54.9 岁），'
        '年龄分层后吸烟的独立效应大幅减弱。',
        '4. 性别和城乡差异与中风无显著关联：可排除这两个维度作为主要风险因素。',
        '5. 机器学习模型（LR AUC=0.84）确认了上述风险因素，为临床筛查提供了量化工具。',
    ]
    for c in conclusions:
        pdf.body(c)
        pdf.ln(0.5)
    pdf.ln(3)

    pdf.h2('建议')
    recommendations = [
        '筛查重点：对 60 岁以上、有高血压/心脏病史、血糖偏高的人群优先中风风险评估。',
        '干预方向：血压控制、血糖管理、心脏健康维护是最具成本效益的预防措施。',
        '模型应用：逻辑回归模型可作为初筛工具，以 78.7% 的召回率识别高风险个体。',
        '数据改进：吸烟数据 30% 为"未知"，建议在数据采集环节加强信息完整性。',
    ]
    for r in recommendations:
        pdf.note(f'▸ {r}')
    pdf.ln(3)

    pdf.hline(C_RED, 1.5)
    pdf.body('— 报告完 —', indent=60)
    pdf.note('本报告所有代码和分析可复现，详见项目源代码。')

    pdf.output(output)
    print(f'✅ PDF 报告已生成：{output}')


if __name__ == '__main__':
    generate()
