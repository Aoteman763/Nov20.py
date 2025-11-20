import streamlit as st
import pandas as pd
import numpy as np
import json

st.set_page_config(page_title="LLM DSS系统", page_icon="🤖", layout="wide")

def create_sample_data(dataset_type):
    np.random.seed(42)

    if dataset_type == "销售数据":
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
        data = {
            '日期': dates,
            '销售额': np.random.normal(10000, 2000, len(dates)).cumsum() + 100000,
            '客户数': np.random.poisson(50, len(dates)),
            '转化率': np.random.beta(5, 2, len(dates)),
            '广告支出': np.random.normal(5000, 1000, len(dates))
        }
        df = pd.DataFrame(data)
        df['月份'] = df['日期'].dt.month
        return df

    elif dataset_type == "客户分类":
        data = {
            '年龄': np.random.normal(35, 10, 500),
            '收入': np.random.normal(50000, 15000, 500),
            '消费频率': np.random.poisson(5, 500),
            '平均订单价值': np.random.normal(200, 50, 500),
            '客户价值': np.random.normal(1000, 300, 500)
        }
        df = pd.DataFrame(data)
        df['客户类型'] = np.where(df['客户价值'] > 1200, '高价值',
                                  np.where(df['客户价值'] > 800, '中价值', '低价值'))
        return df

    else:
        products = ['产品A', '产品B', '产品C', '产品D', '产品E']
        data = {
            '产品': np.random.choice(products, 200),
            '市场份额': np.random.beta(2, 5, 200) * 100,
            '增长率': np.random.normal(10, 5, 200),
            '客户满意度': np.random.normal(4.2, 0.5, 200),
            '价格': np.random.normal(100, 20, 200)
        }
        df = pd.DataFrame(data)
        return df
def simple_predictor(df, target_column):
    """简单的趋势预测"""
    if target_column in df.columns:
        values = df[target_column].values
        if len(values) > 1:
            last_value = values[-1]
            trend = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
            prediction = last_value + trend * 5
            return max(prediction, 0)
    return 0

def ai_analysis(dataset_type, insights, metrics):

    analysis_templates = {
        "销售数据": f"""
## 📊 销售分析报告

**关键指标**:
- 平均销售额: ¥{metrics.get('avg_sales', 0):,.0f}
- 客户转化率: {metrics.get('conversion_rate', 0):.1%}
- 趋势方向: {'上升' if metrics.get('trend', 0) > 0 else '下降'}
### 🔍 深度洞察:
{insights}

### 💡 行动建议:
1. **优化营销策略**: 基于转化率数据调整广告投放
2. **客户细分**: 识别高价值客户群体重点维护
3. **季节性调整**: 根据销售趋势制定库存计划
4. **渠道优化**: 分析各销售渠道效果分配资源
""",
        "客户分类": f"""
## 👥 客户价值分析
**分类结果**:
- 高价值客户: {metrics.get('high_value_pct', 0):.1%}
- 中价值客户: {metrics.get('medium_value_pct', 0):.1%}
- 低价值客户: {metrics.get('low_value_pct', 0):.1%}

### 🔍 客户洞察:
{insights}
### 💡 客户策略:
1. **精准营销**: 针对不同价值群体定制营销方案
2. **忠诚度计划**: 提升高价值客户粘性
3. **价值提升**: 设计中价值客户升级路径
4. **成本优化**: 合理分配低价值客户服务资源
""",
        "市场数据": f"""
## 📈 市场竞争分析

**市场概况**:
- 平均市场份额: {metrics.get('avg_market_share', 0):.1f}%
- 平均增长率: {metrics.get('avg_growth', 0):.1f}%
- 客户满意度: {metrics.get('avg_satisfaction', 0):.1f}/5
### 🔍 市场洞察:
{insights}

### 💡 竞争策略:
1. **产品定位**: 强化优势产品市场地位
2. **价格策略**: 基于竞争态势调整定价
3. **客户体验**: 提升满意度增强客户忠诚
4. **创新驱动**: 投资高增长潜力产品
"""
    }

    return analysis_templates.get(dataset_type, "分析报告生成中...")

def generate_insights(df, dataset_type):

    if dataset_type == "销售数据":
        avg_sales = df['销售额'].mean()
        max_sales = df['销售额'].max()
        min_sales = df['销售额'].min()
        conversion_rate = df['转化率'].mean()

        insights = f"""
- 销售额范围: ¥{min_sales:,.0f} - ¥{max_sales:,.0f}
- 平均转化率: {conversion_rate:.1%}，有较大提升空间
- 建议重点关注转化率优化，每提升1%可增加约¥{avg_sales * 0.01:,.0f}收入
"""
        metrics = {
            'avg_sales': avg_sales,
            'conversion_rate': conversion_rate,
            'trend': 1 if max_sales > avg_sales else -1
        }
    elif dataset_type == "客户分类":
        value_counts = df['客户类型'].value_counts(normalize=True)
        avg_income = df['收入'].mean()
        avg_value = df['客户价值'].mean()

        insights = f"""
- 高价值客户占比: {value_counts.get('高价值', 0):.1%}
- 平均客户收入: ¥{avg_income:,.0f}
- 平均客户价值: ¥{avg_value:,.0f}
- 客户价值分布显示有显著细分机会
"""
        metrics = {
            'high_value_pct': value_counts.get('高价值', 0),
            'medium_value_pct': value_counts.get('中价值', 0),
            'low_value_pct': value_counts.get('低价值', 0)
        }
    else:
        avg_share = df['市场份额'].mean()
        avg_growth = df['增长率'].mean()
        avg_satisfaction = df['客户满意度'].mean()

        insights = f"""
- 产品平均市场份额: {avg_share:.1f}%
- 平均增长率: {avg_growth:.1f}%
- 平均客户满意度: {avg_satisfaction:.1f}/5分
- 市场存在明显差异化机会
"""
        metrics = {
            'avg_market_share': avg_share,
            'avg_growth': avg_growth,
            'avg_satisfaction': avg_satisfaction
        }

    return insights, metrics
def create_simple_chart(df, chart_type="line"):
    if chart_type == "line" and '销售额' in df.columns and '日期' in df.columns:
        recent_data = df.tail(10)
        max_val = recent_data['销售额'].max()
        min_val = recent_data['销售额'].min()
        chart = "销售额趋势图:\n"
        for _, row in recent_data.iterrows():
            value = row['销售额']
            bar_length = int((value - min_val) / (max_val - min_val) * 50) if max_val > min_val else 25
            chart += f"{row['日期'].strftime('%m-%d')}: {'█' * bar_length} ¥{value:,.0f}\n"
        return chart

    elif chart_type == "bar" and '产品' in df.columns and '市场份额' in df.columns:
        chart = "产品市场份额:\n"
        for product in df['产品'].unique():
            avg_share = df[df['产品'] == product]['市场份额'].mean()
            bar_length = int(avg_share / 2)  # 缩放比例
            chart += f"{product}: {'█' * bar_length} {avg_share:.1f}%\n"
        return chart

    return "图表数据不足"

def main():
    st.title("🤖 基于LLM的决策支持系统")
    st.markdown("---")
    with st.sidebar:
        st.header("系统配置")
        dataset_option = st.selectbox(
            "选择数据类型",
            ["销售数据", "客户分类", "市场数据"]
        )
        analysis_dimension = st.selectbox(
            "分析维度",
            ["趋势分析", "分类分析", "对比分析", "预测分析"]
        )
        if st.button("开始分析", type="primary", use_container_width=True):
            st.session_state.analyze = True
            st.session_state.dataset_type = dataset_option
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 数据概览")
        df = create_sample_data(st.session_state.get('dataset_type', '销售数据'))
        col1_1, col1_2, col1_3 = st.columns(3)
        with col1_1:
            st.metric("数据记录", len(df))
        with col1_2:
            st.metric("数据维度", len(df.columns))
        with col1_3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("数值特征", len(numeric_cols))
        with st.expander("查看数据", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        with st.expander("数据可视化"):
            if st.session_state.get('dataset_type') == "销售数据":
                st.write("**销售额趋势:**")
                chart_text = create_simple_chart(df, "line")
                st.text(chart_text)
                st.write("**统计摘要:**")
                st.write(f"最大值: ¥{df['销售额'].max():,.0f}")
                st.write(f"最小值: ¥{df['销售额'].min():,.0f}")
                st.write(f"平均值: ¥{df['销售额'].mean():,.0f}")
            elif st.session_state.get('dataset_type') == "客户分类":
                st.write("**客户类型分布:**")
                type_counts = df['客户类型'].value_counts()
                for type_name, count in type_counts.items():
                    percentage = count / len(df) * 100
                    st.write(f"- {type_name}: {count}人 ({percentage:.1f}%)")
            else:
                st.write("**产品表现:**")
                chart_text = create_simple_chart(df, "bar")
                st.text(chart_text)
    with col2:
        st.subheader("🔮 预测分析")
        if st.session_state.get('analyze', False):
            target_col = '销售额' if st.session_state.dataset_type == '销售数据' else '客户价值'
            if target_col in df.columns:
                prediction = simple_predictor(df, target_col)
                current_avg = df[target_col].mean()
                st.metric(
                    "未来预测",
                    f"¥{prediction:,.0f}" if '销售额' in target_col or '价值' in target_col else f"{prediction:.1f}%",
                    delta=f"{((prediction - current_avg) / current_avg * 100):.1f}%"
                )
            st.write("**关键统计:**")
            numeric_df = df.select_dtypes(include=[np.number])
            for col in list(numeric_df.columns)[:3]:  # 显示前3个数值列
                mean_val = numeric_df[col].mean()
                st.write(f"- {col}: {mean_val:,.1f}")
            st.write("**数据质量:**")
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.write(f"- 完整性: {completeness:.1f}%")
            st.write(f"- 唯一值: {df.nunique().mean():.0f}个")
    if st.session_state.get('analyze', False):
        st.markdown("---")
        st.subheader("🤖 AI智能分析")
        insights, metrics = generate_insights(df, st.session_state.dataset_type)
        analysis_content = ai_analysis(st.session_state.dataset_type, insights, metrics)
        st.markdown(analysis_content)
        st.markdown("---")
        st.subheader("💬 智能问答")
        question = st.text_input("向AI助手提问:",
                                 placeholder="例如：如何提升销售额？")
        if question:
            qa_pairs = {
                "如何提升销售额": "建议：1.优化产品定价 2.加强数字营销 3.提升客户体验 4.拓展销售渠道",
                "怎样提高转化率": "建议：1.优化落地页设计 2.简化购买流程 3.提供个性化推荐 4.加强客户信任建设",
                "客户分类策略": "建议：1.RFM模型细分 2.行为模式分析 3.生命周期管理 4.个性化服务",
                "市场竞争分析": "建议：1.SWOT分析 2.竞争对手监控 3.差异化定位 4.创新驱动",
                "提升客户满意度": "建议：1.改进产品质量 2.优化客户服务 3.收集用户反馈 4.快速响应问题"
            }
            answer = "我主要专注于数据分析和业务建议。请具体描述您的问题，我会尽力提供有针对性的建议。"
            for key in qa_pairs:
                if key in question:
                    answer = qa_pairs[key]
                    break

            st.info(f"**AI回答**: {answer}")
if 'analyze' not in st.session_state:
    st.session_state.analyze = False
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = "销售数据"

if __name__ == "__main__":
    main()