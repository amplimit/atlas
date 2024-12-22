import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置matplotlib字体和样式
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def calculate_position_size(predicted_return, max_position=0.5):
    """基于预测收益率计算仓位大小"""
    position = 2 * max_position * (1 / (1 + np.exp(-predicted_return)) - 0.5)
    return np.clip(position, -max_position, max_position)

def backtest(results_path, benchmark_path, lstm_path, transaction_cost=0.002, max_position=0.5):
    # 读取所有数据
    df = pd.read_csv(results_path)
    df['date'] = pd.to_datetime(df['Date'])
    
    benchmark_df = pd.read_csv(benchmark_path)
    benchmark_df['date'] = pd.to_datetime(benchmark_df['Date'])
    
    lstm_df = pd.read_csv(lstm_path)
    lstm_df['date'] = pd.to_datetime(lstm_df['Date'])
    
    # 获取共同的时间范围
    start_date = max(df['date'].min(), lstm_df['date'].min(), benchmark_df['date'].min())
    end_date = min(df['date'].max(), lstm_df['date'].max(), benchmark_df['date'].max())
    
    # 筛选共同时间范围的数据
    df_recent = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    benchmark_recent = benchmark_df[(benchmark_df['date'] >= start_date) & (benchmark_df['date'] <= end_date)].copy()
    lstm_recent = lstm_df[(lstm_df['date'] >= start_date) & (lstm_df['date'] <= end_date)].copy()
    
    # 计算benchmark收益率 - 修正为相对第一天的收益率
    first_price = benchmark_recent['Adj Close'].iloc[0]
    benchmark_recent['cumulative_return'] = (benchmark_recent['Adj Close'] / first_price - 1) * 100
    
    # 计算仓位大小
    df_recent['position'] = df_recent['Predicted'].apply(calculate_position_size, max_position=max_position)
    lstm_recent['position'] = lstm_recent['Predicted'].apply(calculate_position_size, max_position=max_position)
    
    # 计算策略每日收益率(考虑交易成本)
    df_recent['position_change'] = df_recent['position'] - df_recent['position'].shift(1)
    df_recent.loc[df_recent.index[0], 'position_change'] = df_recent['position'].iloc[0]
    df_recent['transaction_cost'] = abs(df_recent['position_change']) * transaction_cost
    
    lstm_recent['position_change'] = lstm_recent['position'] - lstm_recent['position'].shift(1)
    lstm_recent.loc[lstm_recent.index[0], 'position_change'] = lstm_recent['position'].iloc[0]
    lstm_recent['transaction_cost'] = abs(lstm_recent['position_change']) * transaction_cost
    
    # 计算考虑交易成本后的日收益率
    df_recent['daily_return'] = df_recent['position'] * df_recent['Actual'] / 100.0 - df_recent['transaction_cost']
    df_recent['cumulative_return'] = ((1 + df_recent['daily_return']).cumprod() - 1) * 100
    
    lstm_recent['daily_return'] = lstm_recent['position'] * lstm_recent['Actual'] / 100.0 - lstm_recent['transaction_cost']
    lstm_recent['cumulative_return'] = ((1 + lstm_recent['daily_return']).cumprod() - 1) * 100
    
    # 计算年化收益率
    days = (df_recent['date'].max() - df_recent['date'].min()).days
    years = days / 365.0
    
    annual_return_strategy = (1 + df_recent['cumulative_return'].iloc[-1]/100) ** (1/years) - 1
    annual_return_benchmark = (1 + benchmark_recent['cumulative_return'].iloc[-1]/100) ** (1/years) - 1
    annual_return_lstm = (1 + lstm_recent['cumulative_return'].iloc[-1]/100) ** (1/years) - 1
    
    # 计算其他指标
    trade_counts = (df_recent['position_change'] != 0).sum()
    lstm_trade_counts = (lstm_recent['position_change'] != 0).sum()
    total_cost = df_recent['transaction_cost'].sum() * 100
    lstm_total_cost = lstm_recent['transaction_cost'].sum() * 100
    
    # 计算最大回撤
    def calculate_max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return min(drawdowns) * 100
    
    max_drawdown_strategy = calculate_max_drawdown(df_recent['daily_return'])
    max_drawdown_lstm = calculate_max_drawdown(lstm_recent['daily_return'])
    
    # 绘制权益曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df_recent['date'], df_recent['cumulative_return'], 
             label='Strategy Returns', color='#1f77b4')
    plt.plot(benchmark_recent['date'], benchmark_recent['cumulative_return'],
             label='S&P 500', color='#ff7f0e')
    plt.plot(lstm_recent['date'], lstm_recent['cumulative_return'],
             label='LSTM Strategy', color='#2ca02c')
             
    plt.title('Cumulative Strategy Returns (2020-2024)')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 计算月度收益
    df_recent['month'] = df_recent['date'].dt.strftime('%Y-%m')
    lstm_recent['month'] = lstm_recent['date'].dt.strftime('%Y-%m')
    
    monthly_returns = df_recent.groupby('month').apply(
        lambda x: ((1 + x['daily_return']).prod() - 1) * 100
    ).to_frame('strategy_return')
    
    monthly_lstm_returns = lstm_recent.groupby('month').apply(
        lambda x: ((1 + x['daily_return']).prod() - 1) * 100
    ).to_frame('lstm_return')
    
    # 绘制原策略月度收益柱状图
    plt.figure(figsize=(15, 6))
    ax = plt.gca()
    x = range(len(monthly_returns))
    ax.bar(x, monthly_returns['strategy_return'], color='#2ecc71', width=0.8)
    
    plt.title('Monthly Strategy Returns')
    plt.xlabel('Month')
    plt.ylabel('Returns (%)')
    
    plt.xticks(x, monthly_returns.index, rotation=45, ha='right')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('monthly_returns.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 绘制策略对比的月度收益柱状图
    plt.figure(figsize=(15, 6))
    
    # 画原策略的柱状图
    x = range(len(monthly_returns))
    plt.bar(x, monthly_returns['strategy_return'], 
           color='#2ecc71', width=0.8, label='Original Strategy')
    
    # 在同一位置画LSTM策略的柱状图，使用透明度
    plt.bar([i+0.3 for i in x], monthly_lstm_returns['lstm_return'], 
           color='#1f77b4', width=0.4, alpha=0.6, label='LSTM Strategy')
    
    plt.title('Strategy Comparison: Monthly Returns')
    plt.xlabel('Month')
    plt.ylabel('Returns (%)')
    plt.legend()
    
    plt.xticks(x, monthly_returns.index, rotation=45, ha='right')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 打印回测结果
    print(f"Backtest Results:")
    print(f"Strategy Total Return: {df_recent['cumulative_return'].iloc[-1]:.2f}%")
    print(f"Benchmark Total Return: {benchmark_recent['cumulative_return'].iloc[-1]:.2f}%")
    print(f"LSTM Strategy Total Return: {lstm_recent['cumulative_return'].iloc[-1]:.2f}%")
    print(f"\nAnnualized Returns:")
    print(f"Strategy: {annual_return_strategy*100:.2f}%")
    print(f"Benchmark: {annual_return_benchmark*100:.2f}%")
    print(f"LSTM Strategy: {annual_return_lstm*100:.2f}%")
    print(f"\nRisk Metrics:")
    print(f"Strategy Max Drawdown: {max_drawdown_strategy:.2f}%")
    print(f"LSTM Strategy Max Drawdown: {max_drawdown_lstm:.2f}%")
    print(f"\nTrading Statistics:")
    print(f"Strategy Trade Count: {trade_counts}")
    print(f"Strategy Total Transaction Cost: {total_cost:.2f}%")
    print(f"LSTM Strategy Trade Count: {lstm_trade_counts}")
    print(f"LSTM Strategy Total Transaction Cost: {lstm_total_cost:.2f}%")
    print(f"\nPosition Information:")
    print(f"Max Position Size: {max_position*100:.1f}%")
    print(f"Average Position Size: {abs(df_recent['position']).mean()*100:.2f}%")
    print(f"Position Size Std: {df_recent['position'].std()*100:.2f}%")
    
    return df_recent

# 使用示例
if __name__ == "__main__":
    results = backtest('results.csv', 
                      'data_short/^GSPC.csv',
                      'prediction_result/predictions_AAPL.csv',
                      transaction_cost=0.002,  # 0.2% 交易成本
                      max_position=0.5)        # 50% 最大仓位