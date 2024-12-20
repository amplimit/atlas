import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedTMDO(nn.Module):
    def __init__(self, n_features, alpha=0.5, beta=0.5):
        super().__init__()
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        # 可学习的指标间权重矩阵
        self.weight_matrix = nn.Parameter(torch.randn(n_features, n_features))
        # 添加拉普拉斯层
        self.laplacian = LaplacianStockLayer(n_features)
        
    def forward(self, x):
        # 原始TMDO特征
        # 计算时间维度二阶差分
        time_diff = torch.zeros_like(x)
        time_diff[:, 1:-1, :] = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
        
        # 计算指标维度差异
        weights = torch.softmax(self.weight_matrix, dim=1)
        indicator_diff = torch.zeros_like(x)
        
        for i in range(self.n_features):
            for j in range(self.n_features):
                if i != j:
                    diff = x[:, :, i:i+1] - x[:, :, j:j+1]
                    indicator_diff[:, :, i:i+1] += weights[i,j] * diff
        
        tmdo_features = self.alpha * time_diff + self.beta * indicator_diff
        
        # 结合拉普拉斯特征
        lap_features = self.laplacian(x)
        
        return tmdo_features, lap_features

class CombinedFeatureProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_groups):
        super().__init__()
        self.tmdo = EnhancedTMDO(input_dim)
        
        # 特征组处理
        self.group_layers = nn.ModuleDict({
            name: FeatureGroupLayer(indices, hidden_dim)
            for name, indices in feature_groups.items()
        })
        
        # 修改特征融合层，确保输出维度正确
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        # TMDO特征
        tmdo_features, lap_features = self.tmdo(x)
        
        # 分组特征处理
        group_outputs = []
        for group_layer in self.group_layers.values():
            group_out = group_layer(x)
            group_outputs.append(group_out)
            
        # 合并并进行维度调整
        combined_features = torch.cat([x, tmdo_features, lap_features], dim=-1)
        fused_features = self.fusion(combined_features)  # 确保输出维度是hidden_dim
        
        return fused_features, tmdo_features, lap_features

class LaplacianStockLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # 创建2D拉普拉斯核
        kernel_2d = torch.Tensor([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])
        
        # 扩展为完整的卷积核
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1
        )
        
        # 初始化卷积权重
        with torch.no_grad():
            self.conv.weight.data = kernel_2d.unsqueeze(0).unsqueeze(0)
            self.conv.bias.data.zero_()
        
        # 冻结参数
        for param in self.conv.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        # print(f"Input shape: {x.shape}")
        
        # 将输入转换为图像格式
        x = x.unsqueeze(1)  # (batch, 1, seq_len, features)
        # print(f"After unsqueeze shape: {x.sha/pe}")
        
        # 应用2D拉普拉斯
        laplacian = self.conv(x)
        # print(f"After conv shape: {laplacian.shape}")
        
        # 确保输出维度正确
        output = laplacian.squeeze(1)  # 移除channel维度
        # print(f"Output shape: {output.shape}")
        
        assert output.shape == (batch_size, seq_len, n_features), \
            f"Output shape {output.shape} doesn't match expected shape {(batch_size, seq_len, n_features)}"
        
        return output
        
# multiprocessing.set_start_method('spawn', force=True)
class StockLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # 标准LSTM门
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)  # 输入和隐藏都是hidden_size
        
        # 修正投影层维度：从input_size到hidden_size
        self.indicator_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 输入维度是input_size
            nn.ReLU()
        )
        
        # 技术指标门和事件门保持不变
        self.indicator_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.event_gate = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x, h_prev, c_prev, indicators, event_impact):
        """
        参数:
        x: [batch_size, hidden_size]
        h_prev, c_prev: [batch_size, hidden_size]
        indicators: [batch_size, input_size]
        event_impact: [batch_size, hidden_size]
        """
        # 检查输入维度
        assert x.size(-1) == self.hidden_size, f"Expected x dim {self.hidden_size}, got {x.size(-1)}"
        assert indicators.size(-1) == self.input_size, f"Expected indicators dim {self.input_size}, got {indicators.size(-1)}"
        
        # 基础LSTM步骤
        h_next, c_next = self.lstm(x, (h_prev, c_prev))
        
        # 将indicators投影到hidden_size维度
        indicators_hidden = self.indicator_projection(indicators)
        
        # 技术指标整合
        indicator_combined = torch.cat([h_next, indicators_hidden], dim=-1)
        indicator_gate = torch.sigmoid(self.indicator_gate(indicator_combined))
        h_next = h_next * indicator_gate + indicators_hidden * (1 - indicator_gate)
        
        # 事件影响整合
        event_combined = torch.cat([h_next, event_impact], dim=-1)
        event_gate = torch.sigmoid(self.event_gate(event_combined))
        h_next = h_next * event_gate + event_impact * (1 - event_gate)
        
        return h_next, c_next

class FeatureGroupLayer(nn.Module):
    def __init__(self, group_features, hidden_dim=64):  # 添加hidden_dim参数
        super().__init__()
        self.group_features = group_features
        
        # 添加投影层,将输入特征投影到hidden_dim维度
        self.projection = nn.Linear(len(group_features), hidden_dim)
        
        # 现在使用固定的hidden_dim作为embed_dim
        self.group_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,  # 使用hidden_dim
            num_heads=8,  # 8头注意力,因为64可以被8整除
            batch_first=True
        )
        
        # 添加输出投影,将结果投影回原始维度
        self.output_projection = nn.Linear(hidden_dim, len(group_features))
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        group_data = x[:, :, self.group_features]
        
        # 投影到高维空间
        projected = self.projection(group_data)
        
        # 组内注意力
        attn_out, _ = self.group_attention(
            projected, projected, projected
        )
        
        # 投影回原始维度
        output = self.output_projection(attn_out)
        
        return output

class EventProcessor(nn.Module):
    def __init__(self, event_dim, hidden_dim, num_event_types):
        super().__init__()
        self.event_embed = nn.Embedding(num_event_types, event_dim)
        
        # 事件影响编码器
        self.impact_encoder = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 时间衰减注意力
        self.time_decay_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, events, market_state, time_distances):
        # 事件编码
        event_embeds = self.event_embed(events)
        
        # 计算事件影响
        impact = self.impact_encoder(event_embeds)
        
        # 考虑时间衰减的注意力
        decay_weights = torch.exp(-0.1 * time_distances).unsqueeze(-1)
        impact = impact * decay_weights
        
        # 与市场状态的交互
        attn_out, _ = self.time_decay_attention(
            market_state.unsqueeze(1),
            impact,
            impact
        )
        
        return attn_out.squeeze(1)

class EnhancedStockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, event_dim, num_event_types, feature_groups):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 特征处理器
        self.feature_processor = CombinedFeatureProcessor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_groups=feature_groups
        )
        
        # 事件处理器
        self.event_processor = EventProcessor(
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            num_event_types=num_event_types
        )
        
        # LSTM - 注意维度设置
        self.lstm = StockLSTMCell(
            input_size=input_dim,    # 原始特征维度
            hidden_size=hidden_dim   # 隐藏层维度
        )
        
        # 预测层保持不变
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, events, time_distances):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 特征处理
        fused_features, tmdo_features, lap_features = self.feature_processor(x)
        
        # 初始化状态
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        group_features_list = []
        
        for t in range(seq_len):
            # 获取当前时间步的特征
            current_features = fused_features[:, t, :]      # [batch_size, hidden_dim]
            current_indicators = x[:, t, :]                 # [batch_size, input_dim] - 使用原始特征
            
            # 事件处理
            current_events = events[:, t]
            current_distances = time_distances[:, t]
            event_impact = self.event_processor(
                current_events, h, current_distances
            )  # [batch_size, hidden_dim]
            
            # 打印维度信息进行调试
            # print(f"current_features: {current_features.shape}")
            # print(f"current_indicators: {current_indicators.shape}")
            # print(f"h: {h.shape}")
            # print(f"event_impact: {event_impact.shape}")
            
            # LSTM步进
            h, c = self.lstm(
                current_features,     # [batch_size, hidden_dim]
                h, c,                # [batch_size, hidden_dim]
                current_indicators,  # [batch_size, input_dim]
                event_impact        # [batch_size, hidden_dim]
            )
            
            # 保存特征组信息
            group_features_list.append(current_features)
            
            # 预测
            pred = self.predictor(h)
            outputs.append(pred)
        
        predictions = torch.stack(outputs, dim=1)
        group_features = torch.stack(group_features_list, dim=1)
        
        return predictions, tmdo_features, group_features


def prepare_feature_groups():
    return {
        'price': [0, 1, 2, 3, 4],  # OHLC + Adj Close
        'volume': [5, 15],         # Volume, Volume_MA5
        'ma': [6, 7],             # MA5, MA20
        'macd': [8, 9, 10],       # MACD, Signal, Hist
        'momentum': [11, 16],      # RSI, CRSI
        'bollinger': [12, 13, 14], # Upper, Middle, Lower
        'kalman': [17, 18],        # Kalman_Price, Kalman_Trend
        'fft': [19, 20]           # FFT_21, FFT_63
    }