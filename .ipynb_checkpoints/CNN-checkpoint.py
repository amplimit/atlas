import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import load_data_from_csv, download_and_prepare_data

def combine_stock_data(symbols, start_date, end_date):
    """
    下载多只股票的数据并拼接
    
    Args:
        symbols (list): 股票代码列表
        start_date (str): 开始日期
        end_date (str): 结束日期
    
    Returns:
        pd.DataFrame: 拼接后的数据
    """
    all_data = []
    
    for symbol in tqdm(symbols):
        # 获取单个股票数据
        # data = download_and_prepare_data(symbol, start_date, end_date)
        data = load_data_from_csv(f"./data_short/{symbol}.csv")

        if not data.empty:
            # 将数据添加到列表中
            all_data.append(data)
    
    # 直接拼接所有数据
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    
    return combined_data

def deform_conv2d(input, offset, weight, stride=1, padding=0, bias=None):
    """Deformable convolution implementation"""
    # 获取输入和权重的维度
    b, c, h, w = input.shape
    out_channels, in_channels, k_h, k_w = weight.shape
    
    # 确保padding正确
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # 先进行常规卷积
    out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding)
    
    # 处理偏移
    offset = offset.view(b, 2, k_h, k_w, h, w)
    
    # 创建基础网格
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(k_h-1)//2, (k_h-1)//2 + 1, device=input.device),
        torch.arange(-(k_w-1)//2, (k_w-1)//2 + 1, device=input.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).to(input.dtype)
    
    # 添加batch和spatial维度
    grid = grid.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)  # [1, kh, kw, 1, 1, 2]
    
    # 生成采样位置
    grid = grid.expand(b, -1, -1, h, w, -1)
    grid = grid + offset.permute(0, 2, 3, 4, 5, 1)
    
    # 归一化采样位置到[-1, 1]
    grid_x = 2.0 * grid[..., 0] / (w - 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / (h - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    
    # 重塑输入用于采样
    x = input.unsqueeze(1).expand(-1, k_h * k_w, -1, -1, -1)
    x = x.reshape(b * k_h * k_w, c, h, w)
    
    # 重塑网格用于采样
    grid = grid.view(b * k_h * k_w, h, w, 2)
    
    # 使用grid_sample采样
    sampled = F.grid_sample(
        x, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    # 重塑采样结果
    sampled = sampled.view(b, k_h * k_w, c, h, w)
    
    # 应用卷积权重
    weight = weight.view(out_channels, -1)
    sampled = sampled.reshape(b, k_h * k_w * c, h * w)
    out = torch.bmm(weight.unsqueeze(0).expand(b, -1, -1), sampled)
    out = out.view(b, out_channels, h, w)
    
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    return out

class DeformableConv2d(nn.Module):
    """可变形卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 常规卷积权重
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
            
        # 偏移量预测器
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size[0] * kernel_size[1],  # x和y方向的偏移
            kernel_size=3, 
            padding=1
        )
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # 预测偏移量
        offset = self.offset_conv(x)
        
        # 应用可变形卷积
        return deform_conv2d(
            x, offset, self.weight, self.stride, self.padding, self.bias
        )

class MultiScaleConv(nn.Module):
    """多尺度卷积块"""
    def __init__(self, channels):
        super().__init__()
        
        # 长期趋势识别
        self.trend_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(3, 50),  # 纵向小,横向大,捕捉长期趋势
            padding=(1, 25)
        )
        
        # 形态识别
        self.pattern_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(5, 25),  # 用于识别头肩顶等形态
            padding=(2, 12)
        )
        
        # 价格-成交量关系
        self.price_volume_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(7, 15),  # 较大的纵向核,捕捉价格与成交量关系
            padding=(3, 7)
        )
        
        # 短期关系
        self.short_term_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(10, 10),  # 正方形核,捕捉局部模式
            padding=5
        )
        
        # 指标关联
        self.indicator_conv = DeformableConv2d(
            channels, channels,
            kernel_size=(15, 3),  # 纵向大,横向小,捕捉指标间关系
            padding=(7, 1)
        )

    def forward(self, x):
        # 提取不同尺度的特征
        trend = self.trend_conv(x)
        pattern = self.pattern_conv(x)
        pv_relation = self.price_volume_conv(x)
        short_term = self.short_term_conv(x)
        indicator = self.indicator_conv(x)
        
        # 特征融合
        return trend + pattern + pv_relation + short_term + indicator

class AdaptiveFinancialBlock(nn.Module):
    """自适应金融特征提取块"""
    def __init__(self, channels, input_dim):
        super().__init__()
        
        # 多尺度卷积替换原来的可变形卷积
        self.multi_scale_conv = MultiScaleConv(channels)
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1
        )
        
        # 动态卷积生成器 - 也使用更大的核
        self.dynamic_conv_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels * 125, 1),  # 生成5x25卷积核
            nn.ReLU()
        )
        
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        # 多尺度特征
        multi_scale_feat = self.multi_scale_conv(x)
        
        # 自注意力特征
        b, c, h, w = x.shape
        flat_x = x.view(b, c, -1).permute(2, 0, 1)
        att_out, _ = self.self_attention(flat_x, flat_x, flat_x)
        att_out = att_out.permute(1, 2, 0).view(b, c, h, w)
        
        # 动态卷积 - 使用5x25的核
        kernel = self.dynamic_conv_gen(x).view(b * c, 1, 5, 25)
        dynamic_groups = F.conv2d(
            x.view(1, b * c, h, w),
            kernel,
            padding=(2, 12),
            groups=b * c
        ).view(b, c, h, w)
        
        # 特征融合
        out = multi_scale_feat + att_out + dynamic_groups
        return self.norm(out)

class DynamicWeightFusion(nn.Module):
    """动态特征融合"""
    def __init__(self, channels):
        super().__init__()
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 5, 1),  # 5个分支的权重
            nn.Softmax(dim=1)
        )
        
    def forward(self, features):
        # features是一个包含5个特征图的列表
        # 生成权重
        concat_features = torch.cat([f.unsqueeze(1) for f in features], dim=1)
        weights = self.weight_gen(features[0])  # 使用第一个特征图生成权重
        
        # 加权融合
        weighted_sum = (concat_features * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return weighted_sum
    

class ResidualFinancialBlock(nn.Module):
    """添加残差连接的金融特征提取块"""
    def __init__(self, channels, input_dim):
        super().__init__()
        
        # 第一个特征提取分支
        self.branch1 = nn.Sequential(
            MultiScaleConv(channels),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # 第二个特征提取分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),  # 1x1卷积调整通道
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1
        )
        
        # 最终的归一化层
        self.final_norm = nn.BatchNorm2d(channels)
        
        # 如果输入输出维度不匹配，使用1x1卷积进行调整
        self.shortcut = nn.Identity()
        if input_dim != channels:
            self.shortcut = nn.Conv2d(input_dim, channels, 1)
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        # 主路径
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        
        # 自注意力特征
        b, c, h, w = x.shape
        flat_x = x.view(b, c, -1).permute(2, 0, 1)
        att_out, _ = self.self_attention(flat_x, flat_x, flat_x)
        att_out = att_out.permute(1, 2, 0).view(b, c, h, w)
        
        # 特征融合
        out = branch1_out + branch2_out + att_out
        
        # 添加残差连接
        out = self.final_norm(out + identity)
        return out

class EnhancedFinancialCNN(nn.Module):
    """改进后的金融CNN，包含残差连接"""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        
        # 初始特征提取
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # 使用残差块替代原来的自适应块
        self.residual_blocks = nn.ModuleList([
            ResidualFinancialBlock(64, 64)
            for _ in range(4)
        ])
        
        # 市场状态感知
        self.market_state = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 8, 1),
            nn.GELU(),
            nn.Conv2d(8, 64, 1),
            nn.Sigmoid()
        )
        
        # 全局特征池化
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((input_dim, 1)),
            nn.Flatten(),
            nn.Linear(input_dim * 64, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        
        # 初始特征提取
        x = self.input_proj(x)
        
        # 残差特征提取
        for block in self.residual_blocks:
            # 注入市场状态信息
            market_weight = self.market_state(x)
            x = x * market_weight
            # 应用残差块
            x = block(x)
        
        # 分类
        return self.global_pool(x)

class AdaptiveFinancialCNN(nn.Module):
    """自适应金融CNN"""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        
        # 初始特征提取
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),  # 改用 BatchNorm2d 替代 LayerNorm
            nn.GELU()
        )

        
        # 自适应特征提取块
        self.adaptive_blocks = nn.ModuleList([
            AdaptiveFinancialBlock(64, input_dim)
            for _ in range(4)
        ])
        
        # 全局特征池化
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((input_dim, 1)),
            nn.Flatten(),
            nn.Linear(input_dim * 64, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 市场状态感知
        self.market_state = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 8, 1),
            nn.GELU(),
            nn.Conv2d(8, 64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        
        # 初始特征提取
        x = self.input_proj(x)
        
        # 自适应特征提取
        for block in self.adaptive_blocks:
            # 注入市场状态信息
            market_weight = self.market_state(x)
            x = x * market_weight
            # 特征提取
            x = x + block(x)
        
        # 分类
        return self.global_pool(x)

class EnhancedStockDataset(Dataset):
    """改进的股票数据集类,支持2D特征组织"""
    def __init__(self, data, sequence_length=250, prediction_horizon=5):
        """
        Args:
            data: DataFrame containing stock data
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of days to predict ahead
        """
        # 定义指标顺序
        self.feature_order = [
            'Close', 'Open', 'High', 'Low',  # 价格指标 (y=1-4)
            'MA5', 'MA20',                   # 均线指标 (y=5-6)
            'MACD', 'MACD_Signal', 'MACD_Hist',  # MACD族 (y=7-9)
            'RSI', 'Upper', 'Middle', 'Lower',    # RSI和布林带 (y=10-13)
            'CRSI', 'Kalman_Price', 'Kalman_Trend',  # 高级指标 (y=14-16)
            'FFT_21', 'FFT_63',              # FFT指标 (y=17-18)
            'Volume', 'Volume_MA5'           # 成交量指标 (y=19-20)
        ]
        
        # 重组数据为2D格式
        self.data = self._organize_data(data)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def _organize_data(self, data):
        """将DataFrame重组为2D张量格式"""
        # 确保所有需要的特征都存在
        for feature in self.feature_order:
            if feature not in data.columns:
                raise ValueError(f"Missing feature: {feature}")
        
        # 提取并堆叠特征
        feature_data = []
        for feature in self.feature_order:
            feature_data.append(data[feature].values)
            
        # 转换为(N, F, T)格式的张量,其中:
        # N是样本数，F是特征数，T是时间步
        return torch.FloatTensor(np.stack(feature_data, axis=0))
    
    def __len__(self):
        return self.data.shape[1] - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # 获取特征序列
        X = self.data[:, idx:idx + self.sequence_length]
        
        # 计算未来收益率
        future_price = self.data[0, idx + self.sequence_length + self.prediction_horizon - 1]
        current_price = self.data[0, idx + self.sequence_length - 1]
        returns = (future_price - current_price) / current_price
        
        # 转换为分类标签
        if returns < -0.02:
            y = 0  # 显著下跌
        elif returns > 0.02:
            y = 2  # 显著上涨
        else:
            y = 1  # 横盘
            
        return X, y

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """改进的训练函数"""
    writer = SummaryWriter('runs/enhanced_stock_prediction')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 使用余弦退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 早停
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        class_correct = [0] * 3
        class_total = [0] * 3
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
                
                # 计算每个类别的准确率
                for i in range(len(y)):
                    label = y[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        # 计算平均指标
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        
        # 记录训练指标
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        # 记录每个类别的准确率
        for i in range(3):
            class_acc = 100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            writer.add_scalar(f'Class_Accuracy/class_{i}', class_acc, epoch)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'CNN_best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
       
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
   
    writer.close()

def main():
    # 加载数据
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", #"INTC", "CRM", 
        "^GSPC", "^NDX", "^DJI", "^IXIC",
        # "UNH", "ABBV","LLY",
        # "FANG", "DLR", "PSA", "BABA", "JD", "BIDU",
        # "QQQ"
    ]
    data = combine_stock_data(symbols, '2020-01-01', '2024-01-01')

    # 创建数据集
    dataset = EnhancedStockDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    # 创建模型
    # model = AdaptiveFinancialCNN(input_dim=len(dataset.feature_order))
    model = EnhancedFinancialCNN(input_dim=len(dataset.feature_order))

    # 训练模型
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
   main()