# ============================================================================
# NEC手术预测 - 保存LR模型和Scaler
# 运行一次即可，生成 lr_model.pkl 和 scaler.pkl
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pickle
import os

# ── 路径配置 ──────────────────────────────────────────────────────────────────
TRAIN_PATH = r"C:\Users\23102\Desktop\NEC\NEC_train_.csv"
VAL_PATH   = r"C:\Users\23102\Desktop\NEC\NEC_validation_2025.csv"
OUT_DIR    = r"C:\Users\23102\Desktop\NEC"   # pkl文件保存位置

FEATURES = [
    'xray_fixed_loops',
    'fibrinogen_gL_24h',
    'bw_catLBW',
    'glucose_mmolL_24h',
    'na_24h',
    'albumin_24h',
    'neut_percent_24h',
    'us_complex_ascites',
    'crp_mgL_24h'
]

TARGET = 'surgery_within_72h'

# ── 读取数据 ──────────────────────────────────────────────────────────────────
print("【Step 1】读取数据...")
train_df = pd.read_csv(TRAIN_PATH, encoding='utf-8')
val_df   = pd.read_csv(VAL_PATH,   encoding='utf-8')

# 创建 bw_catLBW（如果原始列是 bw_cat）
for df in (train_df, val_df):
    if 'bw_catLBW' not in df.columns and 'bw_cat' in df.columns:
        df['bw_catLBW'] = (df['bw_cat'] == 'LBW').astype(int)

# 检查特征是否齐全
available = [f for f in FEATURES if f in train_df.columns]
missing   = [f for f in FEATURES if f not in train_df.columns]
if missing:
    print(f"  ⚠️  训练集中缺少以下列，请检查列名：{missing}")
    raise SystemExit

print(f"  训练集 n={len(train_df)}, 验证集 n={len(val_df)}, 特征数={len(available)}")

# ── 准备特征矩阵 ──────────────────────────────────────────────────────────────
X_train = train_df[available].copy()
y_train = train_df[TARGET].values
X_val   = val_df[available].copy()
y_val   = val_df[TARGET].values

# 用训练集中位数填补缺失值（同时记录中位数供app使用）
medians = {}
for col in available:
    m = X_train[col].median()
    medians[col] = m
    X_train[col].fillna(m, inplace=True)
    X_val[col].fillna(m, inplace=True)

# ── 标准化 ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_va_sc = scaler.transform(X_val)

# ── 训练模型 ──────────────────────────────────────────────────────────────────
print("【Step 2】训练 Logistic Regression (C=0.1)...")
model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
model.fit(X_tr_sc, y_train)

# ── 验证 AUC ──────────────────────────────────────────────────────────────────
auc_train = roc_auc_score(y_train, model.predict_proba(X_tr_sc)[:, 1])
auc_val   = roc_auc_score(y_val,   model.predict_proba(X_va_sc)[:, 1])
print(f"  训练集 AUC = {auc_train:.3f}  （参考值：0.887）")
print(f"  验证集 AUC = {auc_val:.3f}  （参考值：0.816）")

if abs(auc_val - 0.816) > 0.01:
    print("  ⚠️  验证集AUC与历史结果偏差较大，请检查数据路径或列名")
else:
    print("  ✅ AUC核对通过")

# ── 保存 pkl ──────────────────────────────────────────────────────────────────
print("【Step 3】保存模型文件...")

model_path  = os.path.join(OUT_DIR, 'lr_model.pkl')
scaler_path = os.path.join(OUT_DIR, 'scaler.pkl')
median_path = os.path.join(OUT_DIR, 'medians.pkl')

with open(model_path,  'wb') as f: pickle.dump(model,   f)
with open(scaler_path, 'wb') as f: pickle.dump(scaler,  f)
with open(median_path, 'wb') as f: pickle.dump(medians, f)

print(f"  ✅ lr_model.pkl  → {model_path}")
print(f"  ✅ scaler.pkl    → {scaler_path}")
print(f"  ✅ medians.pkl   → {median_path}")

print("\n完成！将以上三个pkl文件和app.py一起推送到GitHub即可部署。")
