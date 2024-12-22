import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data import load_data_from_csv
from train import generate_event_data
from fusion_model import ATLASCNNFusion, FusionStockDataset, prepare_feature_groups
import matplotlib.pyplot as plt

def predict_stock(symbol, model_path, device='cuda'):
    """
    使用训练好的融合模型进行股票预测
    
    Args:
        symbol (str): 股票代码
        model_path (str): 模型权重文件路径
        device (str): 使用的设备 ('cuda' or 'cpu')
        
    Returns:
        pd.DataFrame: 包含预测结果的DataFrame
    """
    # 参数设置
    input_dim = 21
    hidden_dim = 128
    event_dim = 32
    num_event_types = 10
    
    # 1. 加载数据
    print(f"Loading data for {symbol}...")
    data = load_data_from_csv(f"./data/{symbol}.csv")
    events = generate_event_data(data)
    
    # 2. 初始化模型
    print("Initializing model...")
    model = ATLASCNNFusion(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_dim=event_dim,
        num_event_types=num_event_types,
        feature_groups=prepare_feature_groups()
    ).to(device)
    
    # 加载模型权重
    print("Loading model weights...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 创建数据集和加载器
    dataset = FusionStockDataset(data, events)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # 4. 进行预测
    print("Making predictions...")
    predictions = []
    actual_prices = []
    dates = []
    
    with tqdm(total=len(dataloader), desc="Predicting") as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                sequence = batch['sequence'].to(device)
                events = batch['events'].to(device)
                time_distances = batch['time_distances'].to(device)
                
                # 转换target为价格变化百分比
                target = batch['target'].cpu().numpy()
                current_price = batch['current_price'].cpu().numpy()
                
                # 获取预测
                pred, _, _ = model(sequence, events, time_distances)
                batch_preds = pred[:, -1].cpu().numpy()
                
                predictions.extend(batch_preds.flatten())
                actual_prices.extend(target.flatten())
                
                start_idx = idx * dataloader.batch_size
                end_idx = min((idx + 1) * dataloader.batch_size, len(dataset))
                current_dates = data.index[start_idx:end_idx]
                dates.extend(current_dates)

                pbar.update(1)
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    actual_prices = np.array(actual_prices)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actual_prices,
        'Predicted': predictions,
    })
    
    # 计算误差
    results_df['Error'] = np.abs(results_df['Predicted'] - results_df['Actual'])
    
    # 计算相对误差(使用实际价格而不是变化率)
    close_prices = data.loc[results_df['Date'], 'Close'].values
    results_df['Error_Percentage'] = (results_df['Error'] / np.abs(close_prices)) * 100
    
    # 计算评估指标
    mse = float(np.mean(np.square(results_df['Error'])))
    mae = float(np.mean(results_df['Error']))
    mape = float(np.mean(results_df['Error_Percentage']))
    
    # 计算方向准确率
    actual_direction = np.sign(results_df['Actual'])
    pred_direction = np.sign(results_df['Predicted'])
    direction_acc = float(np.mean(actual_direction == pred_direction) * 100)
    
    print("\nPrediction Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {direction_acc:.2f}%")
    
    # 添加一些额外的有用信息
    print("\nPrediction Statistics:")
    print(f"Actual Range: [{results_df['Actual'].min():.4f}, {results_df['Actual'].max():.4f}]")
    print(f"Predicted Range: [{results_df['Predicted'].min():.4f}, {results_df['Predicted'].max():.4f}]")
    
    # 计算相关系数
    correlation = np.corrcoef(results_df['Actual'], results_df['Predicted'])[0,1]
    print(f"Correlation: {correlation:.4f}")
    
    return results_df

def main():
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 设置要预测的股票
    # symbol = "AAPL"  # 可以改成其他股票代码
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", 
        "^GSPC", "^NDX", "^DJI", "^IXIC",
        "UNH", "ABBV", "LLY",
        "FANG", "DLR", "PSA", "BABA", "JD", "BIDU",
        "QQQ"
    ]
    model_path = 'checkpoints/fusion/stage2_best_model.pt'  # 使用第二阶段的最佳模型

    for symbol in tqdm(symbols):
        try:
            # 进行预测
            results = predict_stock(symbol, model_path, device)
            
            # 保存结果
            save_path = f'prediction_result/predictions_{symbol}.csv'
            results.to_csv(save_path, index=False)
            print(f"\nPredictions saved to {save_path}")
            
            # 显示最近的几个预测结果
            print("\nRecent Predictions:")
            print(results.tail())
            
            # 绘制预测结果
            plot_predictions(results, symbol)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")

# 修改plot_predictions函数
def plot_predictions(results, symbol):
    plt.style.use('seaborn-whitegrid')  # 使用更专业的样式
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 上半部分:预测值和实际值对比
    ax1.plot(results['Date'], results['Actual'], label='Actual', color='#2E86C1', alpha=0.7)
    ax1.plot(results['Date'], results['Predicted'], label='Predicted', color='#E67E22', alpha=0.7)
    
    # 添加重要事件标注
    ax1.axvspan('2020-03-01', '2020-04-01', alpha=0.2, color='red', label='COVID-19 Crash')
    
    ax1.set_title(f'{symbol} Stock Price Prediction Analysis', fontsize=14, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price Change (%)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 下半部分:误差分析
    ax2.fill_between(results['Date'], results['Error'], 
                     color='#E74C3C', alpha=0.3, label='Prediction Error')
    ax2.plot(results['Date'], results['Error'], color='#E74C3C', alpha=0.5)
    
    ax2.set_title('Prediction Error Analysis', fontsize=14, pad=20)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存为高质量图片
    plt.savefig(f'predictions_{symbol}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()