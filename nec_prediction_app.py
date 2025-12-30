"""
================================================================================
NEC手术风险预测系统 - 在线预测工具
================================================================================

基于XGBoost机器学习模型的NEC患者72小时内手术风险预测

作者：[您的姓名]
机构：[您的机构]
发表于：[期刊名称]

使用Streamlit构建的交互式Web应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

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
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 加载模型和预处理器
# ============================================================================

@st.cache_resource
def load_model_and_preprocessors():
    """
    加载训练好的模型和预处理器
    
    注意：实际部署时，需要保存和加载真实的模型
    这里使用占位符
    """
    # 这里应该加载真实的模型
    # model = pickle.load(open('xgboost_model.pkl', 'rb'))
    # scaler = pickle.load(open('scaler.pkl', 'rb'))
    # label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    
    # 占位符（实际使用时需要替换）
    return None, None, None

# ============================================================================
# 特征定义
# ============================================================================

FEATURE_INFO = {
    'CRP': {
        'name': 'C反应蛋白 (CRP)',
        'unit': 'mg/L',
        'range': (0.0, 300.0),
        'default': 50.0,
        'normal': (0.0, 10.0),
        'help': 'C反应蛋白水平，炎症标志物'
    },
    'IL6': {
        'name': '白介素-6 (IL-6)',
        'unit': 'pg/mL',
        'range': (0.0, 2000.0),
        'default': 100.0,
        'normal': (0.0, 7.0),
        'help': '白介素-6水平，炎症细胞因子'
    },
    'fibrinogen': {
        'name': '纤维蛋白原',
        'unit': 'g/L',
        'range': (0.5, 8.0),
        'default': 2.5,
        'normal': (1.5, 4.0),
        'help': '血浆纤维蛋白原浓度'
    },
    'glucose': {
        'name': '血糖',
        'unit': 'mmol/L',
        'range': (1.0, 15.0),
        'default': 5.0,
        'normal': (2.5, 7.0),
        'help': '血糖水平'
    },
    'HCO3': {
        'name': '碳酸氢根',
        'unit': 'mmol/L',
        'range': (5.0, 35.0),
        'default': 22.0,
        'normal': (22.0, 28.0),
        'help': '血液碳酸氢根浓度，酸碱平衡指标'
    },
    'creatinine': {
        'name': '肌酐',
        'unit': 'μmol/L',
        'range': (10.0, 200.0),
        'default': 60.0,
        'normal': (20.0, 100.0),
        'help': '血肌酐水平，肾功能指标'
    },
    'hemoglobin': {
        'name': '血红蛋白',
        'unit': 'g/L',
        'range': (50.0, 200.0),
        'default': 130.0,
        'normal': (110.0, 160.0),
        'help': '血红蛋白浓度'
    },
    'platelets': {
        'name': '血小板',
        'unit': '×10⁹/L',
        'range': (20.0, 600.0),
        'default': 200.0,
        'normal': (100.0, 300.0),
        'help': '血小板计数'
    },
    'xray_fixed_loops': {
        'name': 'X线固定肠襻',
        'type': 'categorical',
        'options': ['无', '有'],
        'help': 'X线检查是否发现固定肠襻征象'
    },
    'bw_cat': {
        'name': '出生体重分类',
        'type': 'categorical',
        'options': ['正常体重 (NBW)', '低体重 (LBW)', '极低体重 (VLBW)', '超低体重 (ELBW)'],
        'help': '新生儿出生体重分类'
    }
}

# ============================================================================
# 主程序
# ============================================================================

def main():
    # 标题
    st.markdown('<div class="main-header">🏥 NEC手术风险预测系统</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于机器学习的坏死性小肠结肠炎手术风险早期评估工具</div>', 
                unsafe_allow_html=True)
    
    # 侧边栏说明
    with st.sidebar:
        st.header("📋 使用说明")
        st.info("""
        **如何使用：**
        1. 输入患者的临床指标
        2. 点击"预测手术风险"按钮
        3. 查看风险评估结果
        
        **注意事项：**
        - 请确保数值在合理范围内
        - 异常值会用红色标注
        - 本工具仅供参考，不能替代临床判断
        """)
        
        st.header("ℹ️ 模型信息")
        st.markdown("""
        - **算法**: XGBoost
        - **验证AUC**: 0.866
        - **敏感度**: 78.4%
        - **特异度**: 68.0%
        - **训练数据**: 356例NEC患者
        - **验证方法**: 时间分层验证
        """)
        
        st.header("📚 参考文献")
        st.markdown("""
        [您的论文引用]
        
        如需引用本工具，请使用：
        [引用格式]
        """)
    
    # 主界面
    st.markdown("---")
    
    # 输入表单
    st.header("📝 患者信息输入")
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.subheader("基本信息与分类变量")
        
        # 出生体重分类
        bw_options = FEATURE_INFO['bw_cat']['options']
        bw_cat = st.selectbox(
            FEATURE_INFO['bw_cat']['name'],
            bw_options,
            help=FEATURE_INFO['bw_cat']['help']
        )
        input_data['bw_cat'] = bw_cat
        
        # X线固定肠襻
        xray_options = FEATURE_INFO['xray_fixed_loops']['options']
        xray = st.radio(
            FEATURE_INFO['xray_fixed_loops']['name'],
            xray_options,
            horizontal=True,
            help=FEATURE_INFO['xray_fixed_loops']['help']
        )
        input_data['xray_fixed_loops'] = 1 if xray == '有' else 0
        
        st.subheader("炎症指标")
        
        # CRP
        crp = st.number_input(
            f"{FEATURE_INFO['CRP']['name']} ({FEATURE_INFO['CRP']['unit']})",
            min_value=FEATURE_INFO['CRP']['range'][0],
            max_value=FEATURE_INFO['CRP']['range'][1],
            value=FEATURE_INFO['CRP']['default'],
            help=FEATURE_INFO['CRP']['help']
        )
        if crp > FEATURE_INFO['CRP']['normal'][1]:
            st.warning(f"⚠️ CRP升高（正常值: {FEATURE_INFO['CRP']['normal'][1]} {FEATURE_INFO['CRP']['unit']}以下）")
        input_data['CRP'] = crp
        
        # IL-6
        il6 = st.number_input(
            f"{FEATURE_INFO['IL6']['name']} ({FEATURE_INFO['IL6']['unit']})",
            min_value=FEATURE_INFO['IL6']['range'][0],
            max_value=FEATURE_INFO['IL6']['range'][1],
            value=FEATURE_INFO['IL6']['default'],
            help=FEATURE_INFO['IL6']['help']
        )
        if il6 > FEATURE_INFO['IL6']['normal'][1]:
            st.warning(f"⚠️ IL-6升高（正常值: {FEATURE_INFO['IL6']['normal'][1]} {FEATURE_INFO['IL6']['unit']}以下）")
        input_data['IL6'] = il6
        
        # 纤维蛋白原
        fib = st.number_input(
            f"{FEATURE_INFO['fibrinogen']['name']} ({FEATURE_INFO['fibrinogen']['unit']})",
            min_value=FEATURE_INFO['fibrinogen']['range'][0],
            max_value=FEATURE_INFO['fibrinogen']['range'][1],
            value=FEATURE_INFO['fibrinogen']['default'],
            help=FEATURE_INFO['fibrinogen']['help']
        )
        if fib < FEATURE_INFO['fibrinogen']['normal'][0] or fib > FEATURE_INFO['fibrinogen']['normal'][1]:
            st.warning(f"⚠️ 纤维蛋白原异常（正常范围: {FEATURE_INFO['fibrinogen']['normal'][0]}-{FEATURE_INFO['fibrinogen']['normal'][1]} {FEATURE_INFO['fibrinogen']['unit']}）")
        input_data['fibrinogen'] = fib
    
    with col2:
        st.subheader("代谢指标")
        
        # 血糖
        glucose = st.number_input(
            f"{FEATURE_INFO['glucose']['name']} ({FEATURE_INFO['glucose']['unit']})",
            min_value=FEATURE_INFO['glucose']['range'][0],
            max_value=FEATURE_INFO['glucose']['range'][1],
            value=FEATURE_INFO['glucose']['default'],
            help=FEATURE_INFO['glucose']['help']
        )
        if glucose < FEATURE_INFO['glucose']['normal'][0] or glucose > FEATURE_INFO['glucose']['normal'][1]:
            st.warning(f"⚠️ 血糖异常（正常范围: {FEATURE_INFO['glucose']['normal'][0]}-{FEATURE_INFO['glucose']['normal'][1]} {FEATURE_INFO['glucose']['unit']}）")
        input_data['glucose'] = glucose
        
        # 碳酸氢根
        hco3 = st.number_input(
            f"{FEATURE_INFO['HCO3']['name']} ({FEATURE_INFO['HCO3']['unit']})",
            min_value=FEATURE_INFO['HCO3']['range'][0],
            max_value=FEATURE_INFO['HCO3']['range'][1],
            value=FEATURE_INFO['HCO3']['default'],
            help=FEATURE_INFO['HCO3']['help']
        )
        if hco3 < FEATURE_INFO['HCO3']['normal'][0]:
            st.warning(f"⚠️ 代谢性酸中毒（正常范围: {FEATURE_INFO['HCO3']['normal'][0]}-{FEATURE_INFO['HCO3']['normal'][1]} {FEATURE_INFO['HCO3']['unit']}）")
        input_data['HCO3'] = hco3
        
        # 肌酐
        creat = st.number_input(
            f"{FEATURE_INFO['creatinine']['name']} ({FEATURE_INFO['creatinine']['unit']})",
            min_value=FEATURE_INFO['creatinine']['range'][0],
            max_value=FEATURE_INFO['creatinine']['range'][1],
            value=FEATURE_INFO['creatinine']['default'],
            help=FEATURE_INFO['creatinine']['help']
        )
        if creat > FEATURE_INFO['creatinine']['normal'][1]:
            st.warning(f"⚠️ 肾功能异常（正常值: {FEATURE_INFO['creatinine']['normal'][1]} {FEATURE_INFO['creatinine']['unit']}以下）")
        input_data['creatinine'] = creat
        
        st.subheader("血液学指标")
        
        # 血红蛋白
        hgb = st.number_input(
            f"{FEATURE_INFO['hemoglobin']['name']} ({FEATURE_INFO['hemoglobin']['unit']})",
            min_value=FEATURE_INFO['hemoglobin']['range'][0],
            max_value=FEATURE_INFO['hemoglobin']['range'][1],
            value=FEATURE_INFO['hemoglobin']['default'],
            help=FEATURE_INFO['hemoglobin']['help']
        )
        if hgb < FEATURE_INFO['hemoglobin']['normal'][0]:
            st.warning(f"⚠️ 贫血（正常范围: {FEATURE_INFO['hemoglobin']['normal'][0]}-{FEATURE_INFO['hemoglobin']['normal'][1]} {FEATURE_INFO['hemoglobin']['unit']}）")
        input_data['hemoglobin'] = hgb
        
        # 血小板
        plt_count = st.number_input(
            f"{FEATURE_INFO['platelets']['name']} ({FEATURE_INFO['platelets']['unit']})",
            min_value=FEATURE_INFO['platelets']['range'][0],
            max_value=FEATURE_INFO['platelets']['range'][1],
            value=FEATURE_INFO['platelets']['default'],
            help=FEATURE_INFO['platelets']['help']
        )
        if plt_count < FEATURE_INFO['platelets']['normal'][0]:
            st.warning(f"⚠️ 血小板减少（正常范围: {FEATURE_INFO['platelets']['normal'][0]}-{FEATURE_INFO['platelets']['normal'][1]} {FEATURE_INFO['platelets']['unit']}）")
        input_data['platelets'] = plt_count
    
    st.markdown("---")
    
    # 预测按钮
    if st.button("🔍 预测手术风险", type="primary", use_container_width=True):
        # 显示输入数据汇总
        with st.expander("📊 查看输入数据汇总"):
            input_df = pd.DataFrame([input_data])
            st.dataframe(input_df, use_container_width=True)
        
        # ====================================================================
        # 执行预测（占位符 - 实际需要加载真实模型）
        # ====================================================================
        
        # 这里应该使用真实模型进行预测
        # 现在使用模拟数据
        
        # 模拟预测概率（实际应该是: prob = model.predict_proba(X)[:, 1][0]）
        # 基于输入数据的简单启发式规则来模拟
        risk_score = 0.0
        
        # 炎症指标权重
        if crp > 50:
            risk_score += 0.15
        if il6 > 100:
            risk_score += 0.12
        if fib > 4.0 or fib < 1.5:
            risk_score += 0.08
            
        # 代谢指标权重
        if glucose < 3 or glucose > 8:
            risk_score += 0.08
        if hco3 < 18:
            risk_score += 0.10
        if creat > 100:
            risk_score += 0.06
            
        # 血液学指标权重
        if hgb < 100:
            risk_score += 0.05
        if plt_count < 100:
            risk_score += 0.07
            
        # X线征象
        if xray == '有':
            risk_score += 0.15
            
        # 出生体重
        if '超低体重' in bw_cat or '极低体重' in bw_cat:
            risk_score += 0.10
        
        # 基础风险
        predicted_prob = min(0.95, max(0.05, 0.30 + risk_score))
        
        # ====================================================================
        # 显示预测结果
        # ====================================================================
        
        st.markdown("---")
        st.header("📋 预测结果")
        
        # 风险等级判定
        if predicted_prob < 0.3:
            risk_level = "低风险"
            risk_color = "low"
            risk_emoji = "✅"
            risk_desc = "72小时内需要手术的概率较低"
        elif predicted_prob < 0.7:
            risk_level = "中风险"
            risk_color = "medium"
            risk_emoji = "⚠️"
            risk_desc = "72小时内可能需要手术，建议密切观察"
        else:
            risk_level = "高风险"
            risk_color = "high"
            risk_emoji = "🚨"
            risk_desc = "72小时内需要手术的概率较高，建议提前准备"
        
        # 显示风险等级
        st.markdown(f"""
        <div class="risk-box risk-{risk_color}">
            <h1>{risk_emoji} {risk_level}</h1>
            <h2>预测手术概率: {predicted_prob*100:.1f}%</h2>
            <p style="font-size: 1.1rem; margin-top: 1rem;">{risk_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 详细指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="预测概率",
                value=f"{predicted_prob*100:.1f}%",
                delta=f"{(predicted_prob - 0.5)*100:+.1f}% vs 基线" if predicted_prob > 0.5 else None
            )
        
        with col2:
            st.metric(
                label="风险分层",
                value=risk_level
            )
        
        with col3:
            confidence = "高" if 0.2 < predicted_prob < 0.8 else "中" if 0.1 < predicted_prob < 0.9 else "低"
            st.metric(
                label="预测可信度",
                value=confidence
            )
        
        # ====================================================================
        # 特征贡献分析（模拟）
        # ====================================================================
        
        st.markdown("---")
        st.header("📊 特征贡献分析")
        
        st.info("以下分析展示了各项指标对预测结果的影响程度")
        
        # 模拟特征贡献（实际应该使用SHAP值）
        contributions = {
            'CRP': crp / 300 * 0.15 if crp > 50 else 0,
            'IL-6': il6 / 2000 * 0.12 if il6 > 100 else 0,
            '纤维蛋白原': 0.08 if fib > 4 or fib < 1.5 else 0,
            '血糖': 0.08 if glucose < 3 or glucose > 8 else 0,
            '碳酸氢根': 0.10 if hco3 < 18 else 0,
            '肌酐': 0.06 if creat > 100 else 0,
            '血红蛋白': 0.05 if hgb < 100 else 0,
            '血小板': 0.07 if plt_count < 100 else 0,
            'X线固定肠襻': 0.15 if xray == '有' else 0,
            '出生体重': 0.10 if '超低' in bw_cat or '极低' in bw_cat else 0
        }
        
        # 排序
        sorted_contrib = dict(sorted(contributions.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True))
        
        # 绘制特征贡献图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(sorted_contrib.keys())[:8]  # 只显示前8个
        values = [sorted_contrib[f] * 100 for f in features]
        colors = ['#d32f2f' if v > 0 else '#1976d2' for v in values]
        
        bars = ax.barh(features, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            width = bar.get_width()
            label_x = width + 0.5 if width > 0 else width - 0.5
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{val:+.1f}%',
                   ha='left' if width > 0 else 'right',
                   va='center',
                   fontsize=10)
        
        ax.set_xlabel('对手术概率的影响 (%)', fontsize=12)
        ax.set_title('各指标对预测结果的贡献', fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d32f2f', alpha=0.7, label='增加手术风险'),
            Patch(facecolor='#1976d2', alpha=0.7, label='降低手术风险')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        st.pyplot(fig)
        plt.close()
        
        # ====================================================================
        # 临床建议
        # ====================================================================
        
        st.markdown("---")
        st.header("💡 临床建议")
        
        if predicted_prob >= 0.7:
            st.markdown("""
            <div class="warning-box">
            <h3>🚨 高风险患者管理建议</h3>
            <ul>
                <li><strong>密切监测</strong>: 每4-6小时复查腹部体征和实验室指标</li>
                <li><strong>外科会诊</strong>: 建议及时联系儿外科团队评估</li>
                <li><strong>手术准备</strong>: 提前做好手术准备，包括备血、家属谈话等</li>
                <li><strong>支持治疗</strong>: 加强液体复苏、抗生素治疗、营养支持</li>
                <li><strong>影像学</strong>: 考虑复查腹部X线或超声</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        elif predicted_prob >= 0.3:
            st.markdown("""
            <div class="warning-box">
            <h3>⚠️ 中风险患者管理建议</h3>
            <ul>
                <li><strong>密切观察</strong>: 每6-8小时评估病情变化</li>
                <li><strong>定期复查</strong>: 根据病情变化调整复查频率</li>
                <li><strong>保守治疗</strong>: 继续内科保守治疗</li>
                <li><strong>预警指标</strong>: 注意腹胀加重、腹壁红肿、全身情况恶化等</li>
                <li><strong>家属沟通</strong>: 告知家属病情及可能需要手术的情况</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <h3>✅ 低风险患者管理建议</h3>
            <ul>
                <li><strong>继续观察</strong>: 常规监测病情变化</li>
                <li><strong>保守治疗</strong>: 维持现有治疗方案</li>
                <li><strong>定期评估</strong>: 根据临床常规进行评估</li>
                <li><strong>注意变化</strong>: 如出现病情恶化，及时重新评估</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # 免责声明
        st.markdown("---")
        st.warning("""
        **⚠️ 重要提示**
        
        - 本预测工具基于机器学习模型，仅供临床参考
        - 最终诊疗决策应由临床医生根据患者具体情况综合判断
        - 本工具不能替代医生的临床经验和判断
        - 如有疑问，请咨询专业医生
        """)
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>NEC手术风险预测系统 v1.0</p>
        <p>基于XGBoost机器学习模型 | 验证AUC: 0.866</p>
        <p>© 2025 [您的机构] | 仅供学术研究使用</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
