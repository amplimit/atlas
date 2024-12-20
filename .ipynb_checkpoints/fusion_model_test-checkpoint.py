import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
from datetime import datetime
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import warnings
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from fusion_model import ATLASCNNFusion, FusionStockDataset, prepare_feature_groups
from data import load_data_from_csv
from train import generate_event_data
warnings.filterwarnings('ignore')

@dataclass
class EvaluationConfig:
    """评估配置类"""
    model_path: str
    save_dir: str
    device: str
    test_period: str = '2022'
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    rolling_window: int = 252
    
class MetricsCalculator:
    """指标计算器基类"""
    @staticmethod
    def calculate_prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """计算预测相关指标"""
        mse = np.mean(np.square(actual - predicted))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / np.abs(actual))) * 100
        
        ss_total = np.sum(np.square(actual - np.mean(actual)))
        ss_residual = np.sum(np.square(actual - predicted))
        r2 = 1 - (ss_residual / ss_total)
        
        direction_actual = np.sign(np.diff(actual))
        direction_pred = np.sign(np.diff(predicted))
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100
        
        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2),
            'Direction_Accuracy': float(direction_accuracy)
        }
    
    @staticmethod
    def calculate_statistical_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """计算统计显著性指标"""
        errors_model = actual - predicted
        errors_naive = actual - np.roll(actual, 1)
        errors_naive = errors_naive[1:]
        errors_model = errors_model[1:]
        
        d = np.square(errors_naive) - np.square(errors_model)
        dm_stat, dm_pvalue = stats.ttest_1samp(d, 0)
        
        corr, corr_pvalue = stats.pearsonr(actual, predicted)
        
        acf_model = np.correlate(errors_model, errors_model, mode='full') / len(errors_model)
        acf_model = acf_model[len(acf_model)//2:]
        
        return {
            'DM_Statistic': float(dm_stat),
            'DM_PValue': float(dm_pvalue),
            'Correlation': float(corr),
            'Correlation_PValue': float(corr_pvalue),
            'ACF_1': float(acf_model[1]) if len(acf_model) > 1 else np.nan,
            'ACF_5': float(acf_model[5]) if len(acf_model) > 5 else np.nan
        }
    
    @staticmethod
    def calculate_financial_metrics(returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """计算金融指标"""
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-6) * np.sqrt(252)
        
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (rolling_max - cumulative_returns) / rolling_max
        max_drawdown = np.max(drawdowns)
        
        downside_returns = np.where(returns < 0, returns, 0)
        downside_std = np.std(downside_returns) * np.sqrt(252)
        sortino_ratio = annual_return / (downside_std + 1e-6)
        
        calmar_ratio = annual_return / (max_drawdown + 1e-6)
        
        return {
            'Annual_Return': float(annual_return),
            'Annual_Volatility': float(annual_volatility),
            'Sharpe_Ratio': float(sharpe_ratio),
            'Sortino_Ratio': float(sortino_ratio),
            'Max_Drawdown': float(max_drawdown),
            'Calmar_Ratio': float(calmar_ratio)
        }
    
    @staticmethod
    def calculate_risk_metrics(returns: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        """计算风险指标"""
        var = float(np.percentile(returns, (1 - confidence_level) * 100))
        cvar = float(np.mean(returns[returns <= var]))
        
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))
        
        upside_returns = np.where(returns > 0, returns, 0)
        downside_returns = np.where(returns < 0, returns, 0)
        upside_vol = float(np.std(upside_returns) * np.sqrt(252))
        downside_vol = float(np.std(downside_returns) * np.sqrt(252))
        
        return {
            'VaR': var,
            'CVaR': cvar,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Upside_Volatility': upside_vol,
            'Downside_Volatility': downside_vol
        }

class Visualizer:
    """可视化工具类"""
    @staticmethod
    def plot_predictions(results: pd.DataFrame, save_path: str):
        """绘制预测结果图"""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(results.index, results['Actual'], label='Actual', alpha=0.7)
        plt.plot(results.index, results['Predicted'], label='Predicted', alpha=0.7)
        plt.title('Price Prediction')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(results.index, results['Error'], label='Prediction Error', color='red', alpha=0.5)
        plt.title('Prediction Error')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_returns_distribution(returns: np.ndarray, save_path: str):
        """绘制收益分布图"""
        plt.figure(figsize=(10, 6))
        sns.histplot(returns, kde=True)
        plt.title('Returns Distribution')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_cumulative_returns(returns: np.ndarray, save_path: str):
        """绘制累积收益曲线"""
        plt.figure(figsize=(15, 8))
        cumulative_returns = np.cumprod(1 + returns) - 1
        plt.plot(cumulative_returns * 100)
        plt.title('Cumulative Returns (%)')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_rolling_metrics(returns: np.ndarray, window: int, save_path: str):
        """绘制滚动指标"""
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 1, 1)
        rolling_sharpe = pd.Series(returns).rolling(window).mean() / \
                        (pd.Series(returns).rolling(window).std() + 1e-6) * np.sqrt(252)
        plt.plot(rolling_sharpe)
        plt.title(f'Rolling Sharpe Ratio ({window} days)')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
        plt.plot(rolling_vol)
        plt.title(f'Rolling Volatility ({window} days)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        rolling_dd = pd.Series(returns).rolling(window).apply(
            lambda x: (np.cumprod(1 + x) / np.maximum.accumulate(np.cumprod(1 + x)) - 1).min()
        )
        plt.plot(rolling_dd)
        plt.title(f'Rolling Maximum Drawdown ({window} days)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class BacktestEngine:
    """回测引擎"""
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
    
    def backtest(self, predictions: pd.DataFrame, transaction_cost: float = 0.001) -> Dict[str, float]:
        """执行回测"""
        portfolio = self.initial_capital
        position = 0  # 0: 空仓, 1: 持仓
        trades = []
        daily_returns = []
        
        for i in range(1, len(predictions)):
            signal = 1 if predictions.iloc[i]['Predicted'] > predictions.iloc[i-1]['Predicted'] else 0
            price_change = predictions.iloc[i]['Actual'] - predictions.iloc[i-1]['Actual']
            
            if position != signal:
                cost = portfolio * transaction_cost
                portfolio -= cost
                position = signal
            
            if position:
                returns = price_change / predictions.iloc[i-1]['Actual']
                portfolio_change = portfolio * returns
                portfolio += portfolio_change
                daily_returns.append(returns)
            else:
                daily_returns.append(0)
            
            trades.append({
                'date': predictions.index[i],
                'signal': signal,
                'portfolio': portfolio,
                'returns': daily_returns[-1]
            })
        
        trades_df = pd.DataFrame(trades)
        daily_returns = np.array(daily_returns)
        
        return {
            'Final_Portfolio': float(portfolio),
            'Total_Return': float((portfolio - self.initial_capital) / self.initial_capital),
            'Annual_Return': float(np.mean(daily_returns) * 252),
            'Annual_Volatility': float(np.std(daily_returns) * np.sqrt(252)),
            'Sharpe_Ratio': float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)),
            'Max_Drawdown': float(self._calculate_max_drawdown(trades_df['portfolio'].values)),
            'Win_Rate': float(np.mean(daily_returns > 0))
        }
    
    @staticmethod
    def _calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
        """计算最大回撤"""
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (running_max - portfolio_values) / running_max
        return np.max(drawdown)

class ModelEvaluator:
    """模型评估器主类"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = config.device
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        os.makedirs(config.save_dir, exist_ok=True)
    
    def evaluate_model(self, symbols: List[str], market_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """评估模型"""
        results = defaultdict(list)
        
        # 评估每个股票
        for symbol in symbols:
            print(f"\nEvaluating {symbol}...")
            market = market_types.get(symbol, 'Unknown') if market_types else 'Unknown'
            
            pred_results = self._get_predictions(symbol)
            if pred_results is None:
                continue
            
            metrics = self._calculate_all_metrics(pred_results)
            metrics['Symbol'] = symbol
            metrics['Market'] = market
            
            self._generate_stock_report(pred_results, metrics, symbol)
            results[market].append(metrics)
        
        # 生成汇总报告
        summary_df = self._generate_summary_report(results)
        
        # 生成LaTeX表格
        self._generate_latex_tables(summary_df.groupby('Market').agg(['mean', 'std']))
        
        return summary_df

    def _get_predictions(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取预测结果"""
        try:
            # 加载数据
            data = load_data_from_csv(f"./data_short/{symbol}.csv")
            events = generate_event_data(data)
            
            # 创建数据集和加载器
            dataset = FusionStockDataset(data, events)
            dataloader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4
            )
            
            # 初始化模型
            model = ATLASCNNFusion(
                input_dim=21,
                hidden_dim=128,
                event_dim=32,
                num_event_types=10,
                feature_groups=prepare_feature_groups()
            ).to(self.device)
            
            # 加载模型权重
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 进行预测
            predictions = []
            actuals = []
            dates = []
            
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    sequence = batch['sequence'].to(self.device)
                    events = batch['events'].to(self.device)
                    time_distances = batch['time_distances'].to(self.device)
                    target = batch['target'].cpu().numpy()
                    current_price = batch['current_price'].cpu().numpy()
                    
                    pred, _, _ = model(sequence, events, time_distances)
                    batch_preds = pred[:, -1].cpu().numpy()
                    
                    predictions.extend(batch_preds.flatten())
                    actuals.extend(target.flatten())
                    
                    # 获取对应的日期
                    start_idx = idx * dataloader.batch_size
                    end_idx = min((idx + 1) * dataloader.batch_size, len(dataset))
                    batch_dates = data.index[start_idx:end_idx]
                    dates.extend(batch_dates)
            
            # 构建结果DataFrame
            results_df = pd.DataFrame({
                'Date': dates,
                'Actual': actuals,
                'Predicted': predictions
            })
            results_df.set_index('Date', inplace=True)
            
            # 计算误差
            results_df['Error'] = np.abs(results_df['Predicted'] - results_df['Actual'])
            
            # 计算收益率
            results_df['Returns'] = results_df['Actual'].pct_change()
            
            return results_df
            
        except Exception as e:
            print(f"Error getting predictions for {symbol}: {str(e)}")
            return None
    
    def _calculate_all_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """计算所有评估指标"""
        actual = results['Actual'].values
        predicted = results['Predicted'].values
        returns = results['Returns'].values if 'Returns' in results else None
        
        metrics = {}
        # 预测指标
        metrics.update(self.metrics_calculator.calculate_prediction_metrics(actual, predicted))
        # 统计指标
        metrics.update(self.metrics_calculator.calculate_statistical_metrics(actual, predicted))
        
        # 金融和风险指标(如果有收益率数据)
        if returns is not None:
            metrics.update(self.metrics_calculator.calculate_financial_metrics(
                returns, self.config.risk_free_rate))
            metrics.update(self.metrics_calculator.calculate_risk_metrics(
                returns, self.config.confidence_level))
        
        return metrics
    
    def _generate_stock_report(self, results: pd.DataFrame, metrics: Dict[str, float], symbol: str):
        """生成个股评估报告"""
        stock_dir = os.path.join(self.config.save_dir, symbol)
        os.makedirs(stock_dir, exist_ok=True)
        
        # 保存指标
        pd.DataFrame([metrics]).to_csv(os.path.join(stock_dir, 'metrics.csv'))
        
        # 生成图表
        self.visualizer.plot_predictions(results, os.path.join(stock_dir, 'predictions.png'))
        
        if 'Returns' in results:
            returns = results['Returns'].values
            self.visualizer.plot_returns_distribution(
                returns, os.path.join(stock_dir, 'returns_distribution.png'))
            self.visualizer.plot_cumulative_returns(
                returns, os.path.join(stock_dir, 'cumulative_returns.png'))
            self.visualizer.plot_rolling_metrics(
                returns, self.config.rolling_window,
                os.path.join(stock_dir, 'rolling_metrics.png'))
    
    def _generate_summary_report(self, results: Dict[str, List[Dict[str, float]]]) -> pd.DataFrame:
        """生成汇总报告"""
        all_results = []
        for market, market_results in results.items():
            df = pd.DataFrame(market_results)
            df['Market'] = market
            all_results.append(df)
        
        if not all_results:
            print("No results to summarize")
            return pd.DataFrame()
        
        summary_df = pd.concat(all_results)
        
        # 添加回测结果
        backtest_engine = BacktestEngine(initial_capital=1000000)
        for idx, row in summary_df.iterrows():
            symbol = row['Symbol']
            predictions = pd.read_csv(os.path.join(self.config.save_dir, symbol, 'predictions.csv'))
            backtest_results = backtest_engine.backtest(predictions)
            for key, value in backtest_results.items():
                summary_df.loc[idx, f'Backtest_{key}'] = value
        
        # 计算市值分类
        summary_df['Market_Cap_Category'] = summary_df.apply(
            lambda x: self._categorize_market_cap(x['Symbol']), axis=1)
        
        # 计算波动率分类
        summary_df['Volatility_Category'] = pd.qcut(
            summary_df['Annual_Volatility'], q=3, labels=['Low', 'Medium', 'High'])
        
        # 按不同维度统计
        dimensions = {
            'Market': summary_df.groupby('Market'),
            'Market_Cap': summary_df.groupby('Market_Cap_Category'),
            'Volatility': summary_df.groupby('Volatility_Category')
        }
        
        # 计算各维度统计
        for dim_name, dim_group in dimensions.items():
            stats_df = dim_group.agg({
                'MAPE': ['mean', 'std', 'min', 'max'],
                'Direction_Accuracy': ['mean', 'std', 'min', 'max'],
                'Sharpe_Ratio': ['mean', 'std', 'min', 'max'],
                'Annual_Return': ['mean', 'std', 'min', 'max'],
                'Max_Drawdown': ['mean', 'std', 'min', 'max'],
                'Backtest_Win_Rate': ['mean', 'std', 'min', 'max']
            })
            stats_df.to_csv(os.path.join(self.config.save_dir, f'{dim_name}_statistics.csv'))
        
        # 计算稳健性测试结果
        robustness_results = self._perform_robustness_tests(summary_df)
        pd.DataFrame(robustness_results).to_csv(os.path.join(self.config.save_dir, 'robustness_tests.csv'))
        
        # 保存详细结果
        summary_df.to_csv(os.path.join(self.config.save_dir, 'detailed_results.csv'))
        
        return summary_df
    
    def _generate_latex_tables(self, market_stats: pd.DataFrame):
        """生成LaTeX格式的表格"""
        key_metrics = [
            'MAPE', 'Direction_Accuracy', 'Sharpe_Ratio', 'Information_Ratio',
            'Annual_Return', 'Annual_Volatility', 'Max_Drawdown', 'Calmar_Ratio'
        ]
        
        latex_str = """
\\begin{table}[htbp]
\\centering
\\caption{Model Performance by Market Type}
\\begin{tabular}{l""" + "c" * len(market_stats.index) + "}\n\\hline\n"
        
        # 添加市场类型作为列标题
        latex_str += "Metric"
        for market in market_stats.index:
            latex_str += f" & {market}"
        latex_str += " \\\\ \\hline\n"
        
        # 添加每个指标的行
        for metric in key_metrics:
            latex_str += metric.replace("_", " ")
            for market in market_stats.index:
                mean = market_stats.loc[market, (metric, 'mean')]
                std = market_stats.loc[market, (metric, 'std')]
                latex_str += f" & {mean:.2f} ($\\pm${std:.2f})"
            latex_str += " \\\\\n"
        
        latex_str += """\\hline
\\end{tabular}
\\label{tab:market_performance}
\\end{table}
"""
        
        # 保存LaTeX表格
        with open(os.path.join(self.config.save_dir, 'performance_table.tex'), 'w') as f:
            f.write(latex_str)
    
    @staticmethod
    def _categorize_market_cap(symbol: str) -> str:
        """根据股票代码分类市值"""
        large_cap = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        mid_cap = ["GE", "BA", "CAT", "UNH", "PFE"]
        
        if symbol in large_cap:
            return "Large Cap"
        elif symbol in mid_cap:
            return "Mid Cap"
        else:
            return "Small Cap"
    
    def _perform_robustness_tests(self, summary_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """执行稳健性测试"""
        results = {}
        
        # 市场条件测试
        market_conditions = {
            'Bull': summary_df['Annual_Return'] > 0.15,
            'Bear': summary_df['Annual_Return'] < -0.15,
            'Neutral': (summary_df['Annual_Return'] >= -0.15) & 
                      (summary_df['Annual_Return'] <= 0.15)
        }
        
        for condition, mask in market_conditions.items():
            subset = summary_df[mask]
            if len(subset) > 0:
                results[f'{condition}_Market'] = {
                    'MAPE': float(subset['MAPE'].mean()),
                    'Direction_Accuracy': float(subset['Direction_Accuracy'].mean()),
                    'Sharpe_Ratio': float(subset['Sharpe_Ratio'].mean())
                }
        
        # 波动率测试
        vol_levels = summary_df.groupby('Volatility_Category')
        for level in vol_levels.groups.keys():
            subset = vol_levels.get_group(level)
            results[f'{level}_Volatility'] = {
                'MAPE': float(subset['MAPE'].mean()),
                'Direction_Accuracy': float(subset['Direction_Accuracy'].mean()),
                'Sharpe_Ratio': float(subset['Sharpe_Ratio'].mean())
            }
        
        # 市值测试
        cap_levels = summary_df.groupby('Market_Cap_Category')
        for level in cap_levels.groups.keys():
            subset = cap_levels.get_group(level)
            results[f'{level}_MarketCap'] = {
                'MAPE': float(subset['MAPE'].mean()),
                'Direction_Accuracy': float(subset['Direction_Accuracy'].mean()),
                'Sharpe_Ratio': float(subset['Sharpe_Ratio'].mean())
            }
        
        return results

def main():
    """主函数"""
    # 配置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 配置参数
    config = EvaluationConfig(
        model_path='checkpoints/fusion/stage2_best_model.pt',
        save_dir=f'evaluation_results/{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        device=device,
        test_period='2022',
        risk_free_rate=0.02,
        confidence_level=0.95,
        rolling_window=252
    )
    
    # 定义要测试的股票及其市场类型
    market_types = {
        # 大型科技股
        "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", 
        "AMZN": "Tech", "META": "Tech",
        
        # 金融股
        "JPM": "Financial", "BAC": "Financial", "GS": "Financial",
        
        # 工业股
        "GE": "Industrial", "BA": "Industrial", "CAT": "Industrial",
        
        # 医疗保健
        "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare"
    }
    
    # 初始化评估器
    evaluator = ModelEvaluator(config)
    
    try:
        # 运行评估
        results = evaluator.evaluate_model(list(market_types.keys()), market_types)
        print("\nEvaluation completed successfully!")
        print("Results saved in:", config.save_dir)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()