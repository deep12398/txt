import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 创建示例数据
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    data = {
        '销售额': np.random.normal(1000, 200, 100).cumsum(),
        '成本': np.random.normal(700, 100, 100).cumsum(),
        '客户数': np.random.randint(50, 200, 100)
    }
    
    return pd.DataFrame(data, index=dates)

# 数据分析
def analyze_data(df):
    # 基本统计信息
    print("基本统计信息:")
    print(df.describe())
    
    # 计算利润
    df['利润'] = df['销售额'] - df['成本']
    
    # 按月聚合数据
    monthly_data = df.resample('M').sum()
    
    return df, monthly_data

# 数据可视化
def visualize_data(df, monthly_data):
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 销售额和成本趋势
    axes[0, 0].plot(df.index, df['销售额'], label='销售额')
    axes[0, 0].plot(df.index, df['成本'], label='成本')
    axes[0, 0].plot(df.index, df['利润'], label='利润', linestyle='--')
    axes[0, 0].set_title('销售、成本和利润趋势')
    axes[0, 0].legend()
    
    # 月度销售柱状图
    monthly_data['销售额'].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('月度销售额')
    
    # 散点图：客户数与销售额关系
    axes[1, 0].scatter(df['客户数'], df['销售额'])
    axes[1, 0].set_xlabel('客户数')
    axes[1, 0].set_ylabel('销售额')
    axes[1, 0].set_title('客户数与销售额关系')
    
    # 饼图：最后一个月的成本和利润占比
    last_month = monthly_data.iloc[-1]
    axes[1, 1].pie([last_month['成本'], last_month['利润']], 
                  labels=['成本', '利润'],
                  autopct='%1.1f%%')
    axes[1, 1].set_title('最近一个月的成本和利润占比')
    
    plt.tight_layout()
    plt.savefig('sales_analysis.png')
    plt.show()

if __name__ == "__main__":
    print("开始数据分析...")
    df = generate_sample_data()
    df, monthly_data = analyze_data(df)
    visualize_data(df, monthly_data)
    print("分析完成，图表已保存")
