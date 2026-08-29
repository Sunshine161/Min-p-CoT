# 动态 Min-p 算法更新说明

## 更新概述
本次更新修改了 `dynamics_minp.py` 中的动态 min_p 算法策略，实现了基于熵值的双模式控制：

### 原算法
- 所有熵值都使用线性映射：`min_p = min_p_min + entropy_ratio * (min_p_max - min_p_min)`
- 熵越高，min_p 越大（更严格的过滤）
- 熵越低，min_p 越小（更宽松的过滤）

### 新算法
- **高熵模式**（熵值 > 阈值）：使用线性映射
  - `min_p = min_p_min + entropy_ratio * (min_p_max - min_p_min)`
  - 适用于模型不确定的情况，需要更严格的过滤
  
- **低熵模式**（熵值 ≤ 阈值）：使用固定值
  - `min_p = 0.2`（可配置）
  - 适用于模型确定的情况，使用稳定的过滤策略

## 主要修改

### 1. `EntropyDynamicMinPLogitsProcessor` 类的更新

#### 新增参数
```python
def __init__(self, ..., 
             entropy_threshold: float = None,  # 熵阈值
             fixed_low_minp: float = 0.2):     # 低熵时的固定min_p值
```

- `entropy_threshold`: 区分高熵和低熵的阈值
  - 如果为 `None`，默认使用 `h_max / 2`
  - 如果启用滑动窗口，会动态调整为 `current_h_max * 0.4`
  
- `fixed_low_minp`: 低熵时使用的固定 min_p 值，默认为 0.2

#### 核心逻辑修改（`__call__` 方法）

```python
# 判断熵值是否超过阈值
current_threshold = self.entropy_threshold
if self.use_sliding_window and len(self.entropy_window) >= self.min_samples_for_update:
    # 动态调整阈值为当前h_max的40%
    current_threshold = current_h_max * 0.4

# 创建掩码
high_entropy_mask = entropy > current_threshold

# 高熵样本：使用线性映射
if high_entropy_mask.any():
    entropy_ratio = (entropy - current_threshold) / (current_h_max - current_threshold)
    entropy_ratio = torch.clamp(entropy_ratio, 0.0, 1.0)
    linear_min_p = self.min_p_min + entropy_ratio * (self.min_p_max - self.min_p_min)
    dynamic_min_p[high_entropy_mask] = linear_min_p[high_entropy_mask].unsqueeze(-1)

# 低熵样本：使用固定值
low_entropy_mask = ~high_entropy_mask
if low_entropy_mask.any():
    dynamic_min_p[low_entropy_mask] = self.fixed_low_minp
```

### 2. 配置文件更新

在 `dynamics_minp_config.json` 中添加了新参数：

```json
"dynamic_minp": {
    ...
    "entropy_threshold": null,    // 熵阈值，null表示自动计算
    "fixed_low_minp": 0.2         // 低熵时的固定min_p值
}
```

### 3. 日志输出增强

- 添加了策略信息的日志输出
- 调试信息中包含当前使用的策略（Linear/Fixed）
- 显示熵阈值和当前熵值的对比

## 使用建议

### 参数调优

1. **entropy_threshold（熵阈值）**
   - 设置为 `null` 让系统自动调整
   - 或设置为固定值（如 3.0-5.0）
   - 较低的阈值：更多使用固定 min_p
   - 较高的阈值：更多使用线性映射

2. **fixed_low_minp（固定 min_p 值）**
   - 推荐范围：0.1-0.3
   - 0.2 是一个平衡的默认值
   - 较低的值：低熵时更宽松
   - 较高的值：低熵时更严格

3. **min_p_min 和 min_p_max（高熵线性映射范围）**
   - 仅影响高熵时的行为
   - 建议 min_p_min: 0.3-0.5
   - 建议 min_p_max: 0.8-1.0

### 适用场景

这种双模式策略特别适合：

1. **数学问题求解**：计算步骤（低熵）使用固定策略，推理步骤（高熵）使用动态策略
2. **代码生成**：语法结构（低熵）使用固定策略，逻辑设计（高熵）使用动态策略
3. **对话系统**：事实陈述（低熵）使用固定策略，创造性回答（高熵）使用动态策略

## 测试验证

运行 `test_new_minp_algorithm.py` 可以验证新算法的行为：

```bash
python test_new_minp_algorithm.py
```

该测试脚本会：
1. 测试低熵情况下是否使用固定 min_p
2. 测试高熵情况下是否使用线性映射
3. 测试边界情况的处理

## 向后兼容性

新算法完全向后兼容：
- 如果不指定新参数，将使用默认值
- 现有的配置文件可以继续使用
- API 接口保持不变

## 性能影响

- 计算开销略有增加（需要判断熵值范围）
- 内存使用基本不变
- 可能提高生成质量的稳定性
