"""
modeling.py  ──  机器学习预测模型
─────────────────────────────────────────
① 数据准备：特征编码 + 标准化 + 不平衡处理
② 逻辑回归（LR）
③ 随机森林（RF）
④ 模型评估对比：准确率/精确率/召回率/F1/AUC
⑤ 特征重要性分析
⑥ ROC 曲线 + 混淆矩阵可视化
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)

FONT_PATH = '/System/Library/Fonts/STHeiti Medium.ttc'
ZH_FONT = FontProperties(fname=FONT_PATH)
ZH_FONT_SM = FontProperties(fname=FONT_PATH, size=10)
ZH_FONT_LG = FontProperties(fname=FONT_PATH, size=14)

C_RED = '#DC3545'
C_BLUE = '#457B9D'
C_GREEN = '#2A9D8F'


def load_and_prepare(csv_path='中风预测_预处理后.csv'):
    """加载预处理后数据，分离特征和标签。"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['是否中风'] = df['是否中风'].map({'是': 1, '否': 0})

    # 定义列类型
    numeric_cols = ['年龄', '血糖', 'BMI']
    binary_cols = ['高血压', '心脏病']
    categorical_cols = ['性别', '曾婚', '职业', '居住类型', '吸烟']

    feature_cols = numeric_cols + binary_cols + categorical_cols
    X = df[feature_cols].copy()
    y = df['是否中风'].values

    # 构建预处理流水线
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
        ('bin', 'passthrough', binary_cols),
    ])

    return X, y, preprocessor, numeric_cols, binary_cols, categorical_cols


def train_models(X, y, preprocessor):
    """训练逻辑回归和随机森林模型。"""
    # 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f'\n  训练集: {len(X_train)} 条 (中风 {y_train.sum()} 人, {y_train.mean():.2%})')
    print(f'  测试集: {len(X_test)} 条 (中风 {y_test.sum()} 人, {y_test.mean():.2%})')

    # ── 逻辑回归 ──────────────────────────────────────────
    lr_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)

    # ── 随机森林 ──────────────────────────────────────────
    rf_pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                       class_weight='balanced',
                                       random_state=42, n_jobs=-1)),
    ])
    rf_pipe.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, lr_pipe, rf_pipe


def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):
    """评估单个模型。"""
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # 交叉验证 AUC
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    metrics = {
        '模型': name,
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred, zero_division=0),
        '召回率': recall_score(y_test, y_pred, zero_division=0),
        'F1分数': f1_score(y_test, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_test, y_prob),
        'CV AUC均值': cv_mean,
        'CV AUC std': cv_std,
    }

    print(f'\n  【{name}】')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'    {k}: {v:.4f}')
        else:
            print(f'    {k}: {v}')

    print(f'\n  分类报告:')
    print(classification_report(y_test, y_pred, target_names=['未中风', '中风'], zero_division=0))

    return metrics, y_pred, y_prob


def plot_roc_curves(results_list, save_path='plots/07_ROC曲线.png'):
    """绘制多模型 ROC 曲线对比。"""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = [C_RED, C_BLUE]
    for i, (name, y_test, y_prob, auc) in enumerate(results_list):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2.5,
                label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
    ax.set_xlabel('假阳性率 (FPR)', fontproperties=ZH_FONT_SM)
    ax.set_ylabel('真阳性率 (TPR)', fontproperties=ZH_FONT_SM)
    ax.set_title('图 7：ROC 曲线对比', fontproperties=ZH_FONT_LG, fontweight='bold')
    ax.legend(prop=ZH_FONT_SM, loc='lower right')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')


def plot_confusion_matrices(results, save_path='plots/08_混淆矩阵.png'):
    """绘制混淆矩阵。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (name, y_test, y_pred) in enumerate(results):
        cm = confusion_matrix(y_test, y_pred)
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['未中风', '中风'],
                    yticklabels=['未中风', '中风'],
                    ax=ax, cbar=False)
        ax.set_title(f'{name}', fontproperties=ZH_FONT_LG, fontweight='bold')
        ax.set_xlabel('预测值', fontproperties=ZH_FONT_SM)
        ax.set_ylabel('真实值', fontproperties=ZH_FONT_SM)
        for label in ax.get_xticklabels():
            label.set_fontproperties(ZH_FONT_SM)
        for label in ax.get_yticklabels():
            label.set_fontproperties(ZH_FONT_SM)

    fig.suptitle('图 8：混淆矩阵对比', fontproperties=ZH_FONT_LG, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')


def plot_feature_importance(rf_pipe, numeric_cols, binary_cols, categorical_cols,
                            preprocessor, save_path='plots/09_特征重要性.png'):
    """提取并可视化随机森林特征重要性。"""
    # 获取变换后的特征名
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()

    all_features = numeric_cols + cat_names + binary_cols
    importances = rf_pipe.named_steps['clf'].feature_importances_

    # 排序
    feat_imp = pd.DataFrame({'特征': all_features, '重要性': importances})
    feat_imp = feat_imp.sort_values('重要性', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [C_RED if v > 0.05 else C_BLUE for v in feat_imp['重要性']]
    ax.barh(range(len(feat_imp)), feat_imp['重要性'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp['特征'], fontproperties=ZH_FONT_SM)
    ax.set_xlabel('特征重要性', fontproperties=ZH_FONT_SM)
    ax.set_title('图 9：随机森林特征重要性（Top 15）', fontproperties=ZH_FONT_LG, fontweight='bold')

    for i, v in enumerate(feat_imp['重要性']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {save_path}')

    return feat_imp


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════
def run(csv_path='中风预测_预处理后.csv'):
    """执行完整建模流程。"""
    print('\n' + '═' * 50)
    print('  机器学习预测建模')
    print('═' * 50)

    # 数据准备
    X, y, preprocessor, numeric_cols, binary_cols, categorical_cols = load_and_prepare(csv_path)
    print(f'\n  特征维度: {X.shape[1]}，标签类别比例: {1-y.mean():.1%} / {y.mean():.1%}')

    # 训练
    X_train, X_test, y_train, y_test, lr_pipe, rf_pipe = train_models(X, y, preprocessor)

    # 评估
    print('\n' + '─' * 50)
    print('  模型评估')
    print('─' * 50)

    lr_metrics, lr_pred, lr_prob = evaluate_model('逻辑回归', lr_pipe, X_train, X_test, y_train, y_test)
    rf_metrics, rf_pred, rf_prob = evaluate_model('随机森林', rf_pipe, X_train, X_test, y_train, y_test)

    # 模型对比表
    print('\n' + '─' * 50)
    print('  模型对比汇总')
    print('─' * 50)
    comparison = pd.DataFrame([lr_metrics, rf_metrics])
    comparison = comparison.set_index('模型')
    print(comparison.to_string(float_format=lambda x: f'{x:.4f}'))

    # ROC 曲线
    lr_auc = roc_auc_score(y_test, lr_prob)
    rf_auc = roc_auc_score(y_test, rf_prob)
    roc_results = [
        ('逻辑回归', y_test, lr_prob, lr_auc),
        ('随机森林', y_test, rf_prob, rf_auc),
    ]

    print('\n' + '─' * 50)
    print('  生成模型图表')
    print('─' * 50)
    plot_roc_curves(roc_results)
    plot_confusion_matrices([
        ('逻辑回归', y_test, lr_pred),
        ('随机森林', y_test, rf_pred),
    ])

    # 特征重要性
    feat_imp = plot_feature_importance(rf_pipe, numeric_cols, binary_cols,
                                       categorical_cols, preprocessor)

    print(f'\n  Top 5 风险特征:')
    top5 = feat_imp.sort_values('重要性', ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f'    · {row["特征"]}: {row["重要性"]:.4f}')

    print(f'\n  ✅ 建模完成')
    return {'lr': lr_metrics, 'rf': rf_metrics, 'comparison': comparison,
            'feature_importance': feat_imp}


if __name__ == '__main__':
    run()
