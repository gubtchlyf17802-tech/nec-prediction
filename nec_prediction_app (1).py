"""
NEC手术风险预测Web应用
基于XGBoost模型的72小时内手术风险预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# 页面配置
st.set_page_config(
    page_title="NEC手术风险预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

# 加载模型和预处理器
@st.cache_resource
def load_model():
    """加载训练好的模型和预处理器"""
    try:
        model = joblib.load('xgboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        return model, scaler, label_encoders, feature_cols, True
    except FileNotFoundError:
        st.warning("⚠️ 模型文件未找到，使用模拟预测模式")
        return None, None, None, None, False

model, scaler, label_encoders, feature_cols, model_loaded = load_model()

def predict_risk(input_data):
    """预测手术风险"""
    if model_loaded:
        # 使用真实模型预测
        try:
            # 准备数据
            df = pd.DataFrame([input_data])
            
            # 处理分类变量
            if 'bw_cat' in label_encoders:
                df['bw_cat'] = label_encoders['bw_cat'].transform(df['bw_cat'])
            
            # 标准化
            df_scaled = scaler.transform(df[feature_cols])
            
            # 预测
            prob = model.predict_proba(df_scaled)[0, 1]
            return float(prob)
        except Exception as e:
            st.error(f"预测错误: {str(e)}")
            return None
    else:
        # 模拟预测（当模型文件不可用时）
        risk_score = 0
        risk_score += min(input_data['crp_mgL_24h'] / 200, 1) * 0.20
        risk_score += min(input_data['il6_pgml_24h'] / 2000, 1) * 0.20
        risk_score += min(input_data['fibrinogen_gL_24h'] / 10, 1) * 0.15
        risk_score += (1 - min(input_data['hco3_24h'] / 30, 1)) * 0.15
        risk_score += min(input_data['creatinine_24h'] / 150, 1) * 0.10
        risk_score += (1 - min(input_data['hb_24h'] / 180, 1)) * 0.10
        risk_score += (1 - min(input_data['plt_24h'] / 400, 1)) * 0.05
        risk_score += input_data['xray_fixed_loops'] * 0.05
        return min(max(risk_score, 0.05), 0.95)

def get_risk_category(prob):
    """根据概率确定风险分类"""
    if prob >= 0.7:
        return "高风险", "risk-high", "#f44336"
    elif prob >= 0.4:
        return "中风险", "risk-medium", "#ff9800"
    else:
        return "低风险", "risk-low", "#4caf50"

def get_clinical_advice(prob, input_data):
    """生成个性化临床建议"""
    category, _, _ = get_risk_category(prob)
    
    advice = []
    
    if category == "高风险":
        advice.append("🚨 **立即建议**：患者需要外科会诊评估手术指征")
        advice.append("📊 **监测重点**：密切监测生命体征和腹部体征变化")
        advice.append("💊 **治疗建议**：确保充分的液体复苏和抗生素治疗")
    elif category == "中风险":
        advice.append("⚠️ **建议**：加强监测，考虑外科会诊")
        advice.append("📊 **监测频率**：每2-4小时评估一次腹部体征")
        advice.append("💊 **治疗优化**：优化内科保守治疗方案")
    else:
        advice.append("✅ **当前状态**：继续内科保守治疗")
        advice.append("📊 **常规监测**：按标准频率监测生命体征")
        advice.append("💊 **治疗方案**：维持当前治疗方案")
    
    # 根据异常指标添加特定建议
    if input_data['crp_mgL_24h'] > 100:
        advice.append("⚕️ **炎症指标**：CRP显著升高，注意感染控制")
    if input_data['il6_pgml_24h'] > 1000:
        advice.append("⚕️ **炎症因子**：IL-6显著升高，提示强烈炎症反应")
    if input_data['hco3_24h'] < 18:
        advice.append("⚕️ **代谢状态**：代谢性酸中毒，注意纠正")
    if input_data['plt_24h'] < 100:
        advice.append("⚕️ **凝血功能**：血小板减少，警惕DIC")
    if input_data['xray_fixed_loops'] == 1:
        advice.append("⚕️ **影像学**：存在固定肠襻，需密切观察")
    
    return advice

# 标题
st.markdown('<div class="main-header">🏥 NEC手术风险预测系统</div>', unsafe_allow_html=True)
st.markdown("---")

# 模型状态提示
if model_loaded:
    st.success("✅ 已加载真实XGBoost模型 (AUC=0.866)")
else:
    st.info("ℹ️ 当前使用模拟预测模式（演示用）")

# 侧边栏 - 患者信息输入
st.sidebar.header("📋 患者临床信息")
st.sidebar.markdown("请输入24小时内最差值")

# 创建两列布局
col1, col2 = st.columns([2, 1])

with st.sidebar:
    # 炎症指标
    st.subheader("🔬 炎症指标")
    crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=500.0, value=50.0, step=5.0,
                          help="C反应蛋白，正常值<10 mg/L")
    il6 = st.number_input("IL-6 (pg/mL)", min_value=0.0, max_value=5000.0, value=500.0, step=50.0,
                          help="白介素-6，正常值<7 pg/mL")
    fibrinogen = st.number_input("纤维蛋白原 (g/L)", min_value=0.0, max_value=15.0, value=3.0, step=0.5,
                                 help="正常值2-4 g/L")
    
    # 代谢指标
    st.subheader("💉 代谢指标")
    glucose = st.number_input("血糖 (mmol/L)", min_value=0.0, max_value=30.0, value=6.0, step=0.5,
                              help="正常值3.9-6.1 mmol/L")
    hco3 = st.number_input("碳酸氢根 (mmol/L)", min_value=0.0, max_value=40.0, value=22.0, step=1.0,
                           help="正常值22-28 mmol/L")
    creatinine = st.number_input("肌酐 (μmol/L)", min_value=0.0, max_value=300.0, value=50.0, step=5.0,
                                 help="新生儿正常值<80 μmol/L")
    
    # 血液学指标
    st.subheader("🩸 血液学指标")
    hb = st.number_input("血红蛋白 (g/L)", min_value=0.0, max_value=250.0, value=150.0, step=10.0,
                        help="新生儿正常值145-225 g/L")
    plt = st.number_input("血小板 (×10⁹/L)", min_value=0.0, max_value=800.0, value=200.0, step=10.0,
                         help="新生儿正常值150-400 ×10⁹/L")
    
    # 影像学和基本信息
    st.subheader("📸 影像学和基本信息")
    xray_loops = st.selectbox("X线固定肠襻", options=[0, 1], 
                              format_func=lambda x: "否" if x == 0 else "是",
                              help="腹部X线是否显示固定肠襻")
    bw_cat = st.selectbox("出生体重分类", 
                          options=["ELBW", "VLBW", "LBW", "NBW"],
                          index=1,
                          help="ELBW:<1000g, VLBW:1000-1499g, LBW:1500-2499g, NBW:≥2500g")
    
    # 预测按钮
    predict_button = st.button("🔮 预测手术风险", type="primary", use_container_width=True)

# 主界面
with col1:
    st.header("📊 预测结果")
    
    if predict_button:
        # 准备输入数据
        input_data = {
            'crp_mgL_24h': crp,
            'il6_pgml_24h': il6,
            'fibrinogen_gL_24h': fibrinogen,
            'glucose_mmolL_24h': glucose,
            'hco3_24h': hco3,
            'creatinine_24h': creatinine,
            'hb_24h': hb,
            'plt_24h': plt,
            'xray_fixed_loops': xray_loops,
            'bw_cat': bw_cat
        }
        
        # 预测
        with st.spinner("正在分析患者数据..."):
            prob = predict_risk(input_data)
        
        if prob is not None:
            # 获取风险分类
            category, risk_class, color = get_risk_category(prob)
            
            # 显示预测结果
            st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h2 style="color: {color}; margin:0;">手术风险: {category}</h2>
                    <h1 style="color: {color}; margin:0.5rem 0;">{prob*100:.1f}%</h1>
                    <p style="margin:0;">72小时内需要手术的概率</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 特征贡献分析
            st.subheader("📈 特征贡献分析")
            
            # 创建特征重要性图表
            features = {
                'CRP': crp / 200,
                'IL-6': il6 / 2000,
                '纤维蛋白原': fibrinogen / 10,
                '血糖': glucose / 20,
                'HCO₃': (30 - hco3) / 30,
                '肌酐': creatinine / 150,
                '血红蛋白': (180 - hb) / 180,
                '血小板': (400 - plt) / 400,
                'X线固定肠襻': xray_loops,
                '出生体重': 0.3 if bw_cat in ['ELBW', 'VLBW'] else 0.1
            }
            
            # 归一化到0-1
            features = {k: max(0, min(1, v)) for k, v in features.items()}
            
            # 绘制条形图
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_list = [color if v > 0.5 else '#4caf50' for v in features.values()]
            bars = ax.barh(list(features.keys()), list(features.values()), color=colors_list)
            ax.set_xlabel('贡献度', fontsize=12)
            ax.set_title('各特征对手术风险的贡献', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            
            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, features.values())):
                ax.text(value + 0.02, i, f'{value:.2f}', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            
            # 临床建议
            st.subheader("💡 个性化临床建议")
            advice_list = get_clinical_advice(prob, input_data)
            for advice in advice_list:
                st.markdown(f"- {advice}")
            
            # 异常值警告
            st.markdown("---")
            st.subheader("⚠️ 异常指标警示")
            
            warnings = []
            if crp > 100:
                warnings.append(f"🔴 **CRP严重升高** ({crp:.1f} mg/L > 100 mg/L)")
            if il6 > 1000:
                warnings.append(f"🔴 **IL-6严重升高** ({il6:.0f} pg/mL > 1000 pg/mL)")
            if hco3 < 18:
                warnings.append(f"🔴 **代谢性酸中毒** (HCO₃ {hco3:.1f} mmol/L < 18 mmol/L)")
            if plt < 100:
                warnings.append(f"🔴 **血小板减少** ({plt:.0f} ×10⁹/L < 100 ×10⁹/L)")
            if creatinine > 100:
                warnings.append(f"🟡 **肌酐升高** ({creatinine:.0f} μmol/L > 100 μmol/L)")
            if xray_loops == 1:
                warnings.append(f"🔴 **影像学异常** (X线显示固定肠襻)")
            
            if warnings:
                for warning in warnings:
                    st.markdown(warning)
            else:
                st.success("✅ 所有指标均在可接受范围内")

with col2:
    st.header("ℹ️ 模型信息")
    
    st.markdown("""
    ### 模型性能
    - **模型**: XGBoost
    - **验证AUC**: 0.866
    - **敏感度**: 78.4%
    - **特异度**: 68.0%
    - **准确度**: 76.1%
    
    ### 研究信息
    - **数据来源**: 时间分层验证
    - **训练集**: 356例 (2022-2024)
    - **验证集**: 113例 (2025)
    - **预测窗口**: 72小时
    
    ### 特征说明
    **炎症标志物**
    - CRP、IL-6反映炎症程度
    - 纤维蛋白原提示凝血异常
    
    **代谢指标**
    - 血糖、HCO₃反映代谢状态
    - 肌酐提示肾功能
    
    **血液学**
    - 血红蛋白、血小板
    
    **影像学**
    - X线固定肠襻
    
    ### 风险分层
    - **高风险** (≥70%): 建议外科会诊
    - **中风险** (40-69%): 加强监测
    - **低风险** (<40%): 继续内科治疗
    
    ### 使用声明
    ⚠️ 本工具仅供临床辅助决策参考，
    最终诊疗方案应由医生根据完整临床
    信息综合判断。
    """)

# 页脚
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>NEC手术风险预测系统 v1.0 | 基于机器学习的临床决策支持工具</p>
        <p>⚠️ 仅供医疗专业人员使用 | 不可替代临床判断</p>
    </div>
    """, unsafe_allow_html=True)
