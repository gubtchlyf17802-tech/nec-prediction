# 🏥 NEC手术风险预测系统

基于机器学习的新生儿坏死性小肠结肠炎(NEC)手术风险预测Web应用

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](你的应用链接)

## 📋 项目简介

本研究开发了一个基于XGBoost的机器学习模型，用于预测NEC患者在确诊后72小时内需要手术干预的风险。该Web应用可帮助临床医生进行早期风险评估和决策支持。

### 主要特点

- ✅ **高性能**: 验证集AUC达0.866
- 🎯 **可解释性强**: 基于10个临床可获得的预测特征
- 🚀 **实时预测**: 输入临床数据即可获得风险评估
- 📊 **可视化分析**: 提供特征贡献度和风险分层
- 💡 **临床建议**: 根据预测结果生成个性化治疗建议

## 🔬 模型性能

| 指标 | 数值 |
|------|------|
| AUC | 0.866 |
| 敏感度 | 78.4% |
| 特异度 | 68.0% |
| 准确度 | 76.1% |
| Brier Score | 0.192 |

## 📊 数据集

- **训练集**: 356例患者 (2022-2024)
- **验证集**: 113例患者 (2025)
- **验证方法**: 时间分层验证

## 🎯 预测特征

### 炎症指标
1. CRP (C反应蛋白)
2. IL-6 (白介素-6)
3. 纤维蛋白原

### 代谢指标
4. 血糖
5. 碳酸氢根 (HCO₃)
6. 肌酐

### 血液学指标
7. 血红蛋白
8. 血小板

### 其他
9. X线固定肠襻
10. 出生体重分类

## 🚀 快速开始

### 在线访问

直接访问我们的Web应用：[NEC手术风险预测系统](你的应用链接)

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/你的用户名/nec-prediction.git
cd nec-prediction

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run nec_prediction_app.py
```

## 📁 项目结构

```
nec-prediction/
├── nec_prediction_app.py     # Streamlit Web应用
├── xgboost_model.pkl          # 训练好的XGBoost模型
├── scaler.pkl                 # 数据标准化器
├── label_encoders.pkl         # 分类变量编码器
├── feature_cols.pkl           # 特征列表
├── requirements.txt           # Python依赖包
└── README.md                  # 项目说明
```

## 💻 技术栈

- **前端**: Streamlit
- **机器学习**: XGBoost, scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn

## 🔐 使用声明

⚠️ **重要提示**：
- 本工具仅供临床辅助决策参考
- 最终诊疗方案应由医生根据完整临床信息综合判断
- 仅供医疗专业人员使用

## 📝 引用

如果您在研究中使用了本工具，请引用：

```
[待发表论文引用格式]
```

## 👥 作者

- 研究团队：[您的医院/机构]
- 联系方式：[您的邮箱]

## 📄 许可证

本项目采用 MIT 许可证

## 🙏 致谢

感谢所有参与数据收集和研究的医护人员。

---

**免责声明**: 本预测模型仅供科研和教育用途，不应作为临床诊断或治疗的唯一依据。
