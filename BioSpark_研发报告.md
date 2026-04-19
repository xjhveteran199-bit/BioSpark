# BioSpark 项目研发报告

> **版本:** v0.8.0 | **日期:** 2026-04-19 | **状态:** 已上线运营

---

## 一、项目概述

**BioSpark** 是一个面向科研人员的生物信号（ECG心电、EEG脑电、EMG肌电）端到端 AI 分析与训练平台。用户无需编程能力，只需上传数据即可完成信号分析、模型推理、自定义 CNN 训练，并生成**出版质量的科研图表**。

**核心定位：** 面向科研人员的零代码生物信号数据处理工具，一站式完成从原始数据到论文图表的全流程。

**在线地址：**
- Railway (全功能): https://efficient-integrity-production-736e.up.railway.app
- Vercel (轻量前端): https://infallible-blackwell.vercel.app
- GitHub: https://github.com/xjhveteran199-bit/BioSpark

---

## 二、技术架构

### 2.1 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **后端框架** | FastAPI + Uvicorn | 异步 Python Web 框架，支持 WebSocket |
| **用户认证** | JWT + bcrypt | 注册/登录/Token 鉴权 |
| **数据库** | SQLAlchemy + PostgreSQL/SQLite | 异步 ORM，开发用 SQLite，生产用 PostgreSQL |
| **深度学习** | PyTorch 2.x (CPU) | 1D-CNN 训练与推理 |
| **模型可解释性** | 🆕 Grad-CAM (Hook-based) | PyTorch forward/backward hook 提取注意力热力图 |
| **信号处理** | NeuroKit2 / MNE / SciPy | ECG R-peak 检测、EEG 分段、EMG 滤波 |
| **实时信号处理** | 🆕 SOS Butterworth 滤波 | 二阶节带通滤波，适用于流式低延迟场景 |
| **数据分析** | NumPy / Pandas / Scikit-learn | t-SNE、混淆矩阵、特征提取 |
| **科研可视化** | Matplotlib / Seaborn | 300 DPI PNG + SVG 出版质量图表 |
| **推理引擎** | PyTorch + ONNX Runtime | .pt 主力推理 + 浏览器端 ONNX |
| **实时通信** | 🆕 WebSocket (FastAPI native) | 双向流式推理，支持 Demo/Device 双模式 |
| **前端** | 原生 HTML/CSS/JS + Plotly.js + Canvas | 零构建步骤，交互式可视化，Canvas 粒子动画 |
| **部署** | Docker / Railway / Vercel | 容器化 + Serverless 双模式 |

### 2.2 系统架构图

```
+------------------------------------------------------------------------+
|                           前端 (Vanilla JS)                             |
|  +----------+ +----------+ +----------+ +-----------+ +--------------+ |
|  | 登录/注册 | | 文件上传  | | 推理结果  | | 训练控制台 | | 🆕 实时监控   | |
|  +----+-----+ +----+-----+ +----+-----+ +-----+-----+ +------+-----+ |
|  | 图表预览  | | 信号可视化| | 下载管理  | |自动优化配置| |🆕Grad-CAM  | |
|  +----+-----+ +----+-----+ +----+-----+ +-----+-----+ +------+-----+ |
+-------+------------+------------+-------------+---------------+-------+
        | REST API   | Plotly.js  | REST API    | WebSocket     | 🆕 WS
        | + JWT Auth |            | + JWT Auth  | (训练监控)     | (流式推理)
+-------+------------+------------+-------------+---------------+-------+
|                          FastAPI 后端                                    |
|  +----------+ +----------+ +----------+ +----------+ +--------------+  |
|  |用户认证   | |格式解析器 | |模型推理器 | |CNN训练引擎| |🆕流式推理引擎 |  |
|  |JWT/bcrypt | |CSV/EDF/  | |PyTorch/  | |Signal1D  | |StreamSession |  |
|  |           | |MAT/ZIP   | |ONNX/Demo | |CNN+优化   | |SOS滤波+滑窗  |  |
|  +----------+ +----------+ +----------+ +----------+ +--------------+  |
|  +----------+ +----------+ +----------+ +----------------------------+ |
|  |出版图表   | |架构图生成 | |信号预处理 | |🆕 Grad-CAM 注意力分析引擎  | |
|  |Matplotlib | |纯Patches | |ECG/EEG/  | |PyTorch Hook + 梯度加权     | |
|  +----------+ +----------+ |EMG多通道  | +----------------------------+ |
|                             +----------+                                |
+-----------------------------------+------------------------------------+
                                    |
                        +-----------+-----------+
                        |  PostgreSQL / SQLite   |
                        |  users, sessions       |
                        +-----------------------+
```

---

## 三、预训练模型矩阵

### 3.1 ECG 心律失常检测 — **已上线**

| 项目 | 详情 |
|------|------|
| **架构** | ECGArrhythmiaCNN — 3 层 Conv1d + GlobalAvgPool + FC |
| **参数量** | 44,293 |
| **输入** | (1, 187) — 单通道，187 样本/心跳 |
| **数据集** | MIT-BIH Arrhythmia Database (PhysioNet) |
| **分类** | 5 类 AAMI: Normal / Supraventricular / Ventricular / Fusion / Unknown |
| **准确率** | **94.1%** |
| **状态** | ✅ 已部署，可在线推理 |

### 3.2 EEG 睡眠分期 — **训练中**

| 项目 | 详情 |
|------|------|
| **架构** | EEGSleepCNN — 4 层 Conv1d（大核 25→15→7→3）+ GlobalAvgPool + FC |
| **参数量** | ~150,000 |
| **输入** | (1, 3000) — 单通道，30秒 @ 100Hz |
| **数据集** | Sleep-EDF Expanded (PhysioNet)，20 被试 |
| **分类** | 5 类 AASM: Wake / N1 / N2 / N3 / REM |
| **准确率** | 训练进行中 |
| **状态** | ⏳ 数据下载+训练中 |

### 3.3 EMG 手势识别 — **已上线**

| 项目 | 详情 |
|------|------|
| **架构** | EMGGestureCNN — 4 层 Conv1d 多通道输入 + GlobalAvgPool + FC |
| **参数量** | 387,701 |
| **输入** | (16, 80) — 16 通道 sEMG，400ms @ 200Hz |
| **数据集** | NinaPro DB5 真实数据，10 被试 |
| **分类** | 53 类（52 手势 + Rest），涵盖 3 组运动 |
| **准确率** | **42.7%**（随机基线 1.9%，为基线的 22 倍） |
| **状态** | ✅ 已部署，可在线推理 |

**EMG 53 类手势分布：**

| Exercise | 类型 | 手势数 | 准确率 | 示例 |
|----------|------|--------|--------|------|
| **E1** | 基础手指动作 | 12 | 48.1% | 食指弯曲、拇指外展、小指伸展 |
| **E2** | 手腕动作 | 17 | 41.5% | 握拳、指点、腕屈/伸、旋前/旋后 |
| **E3** | 抓握动作 | 23 | 35.4% | 圆柱抓握、精密捏取、三指抓握 |
| **Rest** | 静息 | 1 | 84.8% | — |

---

## 四、v0.4 新增功能（P0 训练升级）

### 4.1 自动超参数优化系统

> **新增模块：** `backend/services/auto_optimizer.py`

面向科研用户设计的一键自动优化功能。勾选训练配置中的 **Auto-Optimize** 即可启用，系统自动完成以下流程：

| 功能 | 技术方案 | 说明 |
|------|---------|------|
| **LR Range Test** | Smith 2017 方法 | 100 个 mini-batch 线性扫描 LR，找到 loss 梯度最陡处 |
| **架构自适应** | 规则引擎 | 根据信号长度、类别数、样本量自动选择卷积核大小、通道宽度、Dropout |
| **类别权重** | 逆频率加权 | `n_samples / (n_classes × class_count)`，解决数据不平衡 |
| **Early Stopping** | 验证集监控 | val_loss 不再下降时自动停止，保留最优模型权重 |

**架构自适应规则：**

| 条件 | 调整策略 |
|------|---------|
| signal_length < 64 | 小卷积核 [3,3,3]，避免信息丢失 |
| signal_length ≥ 256 | 大卷积核 [7,5,3]，宽通道 [64,128,256] |
| n_classes > 10 | 扩展网络宽度至 [64,128,256] |
| n_samples < 200 | 增强 Dropout (0.5/0.3)，防过拟合 |
| n_samples > 5000 | 降低 Dropout (0.2/0.1)，释放容量 |

**`Signal1DCNN` 架构升级：** 原固定 3-block 结构升级为动态架构，支持通过 `arch_config` 参数指定 `kernel_sizes`、`channels`、`dropout1/2`、`fc_hidden`。向后兼容——无参数调用时保持原有默认值。

**API 变更：** `POST /api/train/start` 新增 3 个可选字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auto_mode` | bool | false | 启用自动 LR/架构/权重优化 |
| `early_stopping_patience` | int | 10 | 验证 loss 无改善的容忍轮数 |
| `use_class_weights` | bool | true | 是否自动计算类别权重 |

### 4.2 出版质量图表生成

> **新增模块：** `backend/services/publication_figures.py` + `backend/routers/figures.py`

训练完成后自动进入 **T6 出版图表** 阶段，基于 Matplotlib 服务端渲染，生成 5 种图表：

| 图表 | 函数 | 内容 |
|------|------|------|
| **Training Curves** | `render_training_curves()` | Loss + Accuracy 双面板，标注 best val_acc epoch，虚线/实线区分 train/val |
| **Confusion Matrix** | `render_confusion_matrix()` | Count + Normalized 并排热力图，逐单元格标注数值，colorbar 色阶 |
| **t-SNE** | `render_tsne()` | 按类别着色散点图，ConvexHull 凸包边界，图外 Legend |
| **Per-Class Metrics** | `render_per_class_metrics()` | Precision/Recall/F1 分组柱状图，数值标签，Overall Accuracy 参考线 |
| **Architecture Diagram** | `render_architecture_diagram()` | CNN 结构示意图，每层标注类型/参数/tensor shape，彩色分层 |

**三种期刊样式一键切换：**

| 样式 | 字体 | 字号 | 配色 | 单栏宽度 | 双栏宽度 |
|------|------|------|------|---------|---------|
| **Nature** | Arial | 8pt | Blues 蓝色系 | 3.5 in (89mm) | 7.0 in (183mm) |
| **IEEE** | Times New Roman | 9pt | YlOrRd 暖色系 | 3.5 in | 7.16 in |
| **Science** | Helvetica | 7pt | RdBu_r 双色系 | 2.3 in (55mm) | 4.8 in (120mm) |

**输出格式：**
- **PNG** — 300 DPI，适合直接插入 Word/LaTeX
- **SVG** — 矢量格式，可在 Illustrator/Inkscape 中编辑
- **ZIP 打包** — 10 个文件（5种图 × 2格式）+ README.txt

### 4.3 模型架构图生成

> **技术方案：** 纯 Matplotlib Patches 绘制，无需 Graphviz 系统依赖

**工作原理：**
1. `_compute_layer_shapes()` — 注册 forward hook → 传入零 tensor → 收集每层 output shape
2. `_group_into_blocks()` — 将 Conv+BN+ReLU+Pool 组合为逻辑 block
3. `render_architecture_diagram()` — 绘制彩色方框 + 箭头连接 + shape 标注

**颜色编码：**

| 层类型 | 颜色 |
|--------|------|
| Conv1d | 蓝色 #3b82f6 |
| BatchNorm1d | 灰色 #9ca3af |
| ReLU | 绿色 #22c55e |
| MaxPool / AdaptiveAvgPool | 橙色 #f59e0b |
| Linear | 紫色 #8b5cf6 |
| Dropout | 浅灰 #d1d5db |

---

## 五、v0.8 新增功能（智能引导训练 + 数据质量评估）

> **发布日期：** 2026-04-19 | **目标用户：** 无深度学习背景的研究生

### 5.1 智能训练模式选择

新增"训练模式引导"步骤（T2.5），替代直接暴露超参数表单的旧流程：

**四种预设模式：**
| 模式 | 轮数 | 学习率搜索 | 场景 |
|------|------|-----------|------|
| 智能自动（⚡ Auto） | 50 轮 + 早停 | ✓ | 大多数数据集 |
| 快速测试（🔬 Fast） | 20 轮 | ✗ | 验证数据质量 |
| 发表级别（🏆 Thorough） | 100 轮 + 早停 | ✓ | 论文投稿前 |
| 自定义（⚙️ Custom） | 用户设定 | 可选 | 高级调参 |

用户点击模式卡片即可自动填充最优参数，无需手动理解每个超参数含义。

**后端实现：**
- `GET /api/train/presets` — 返回 4 种预设的中英文描述及参数配置
- `TrainStartRequest.preset` 字段 — 非 custom 模式时服务端自动覆盖参数

### 5.2 数据质量评估

上传数据集后自动调用 `POST /api/train/assess`，在模式选择页面顶部显示质量横幅：

**检测项目：**
- NaN / Inf 值（阻断级错误）
- 样本不足（<50 阻断，<300 警告）
- 类别不平衡（比例 >10× 严重警告，>5× 一般警告）
- 空类别（阻断级错误）
- 平坦信号通道（>50% 列方差极低时警告）

**质量评分（0–100）：** 错误扣 30 分，警告扣 15 分，提示扣 5 分。前端以色环仪表盘展示（绿/蓝/黄/红），配合具体问题列表和修复建议。

**后端实现：**
- `backend/services/auto_optimizer.py` — 新增 `DataQualityAssessor` 类

### 5.3 训练进度自然语言叙述

训练 Dashboard 增加 `#train-narration` 叙述框，每个 epoch 实时更新，将技术指标翻译为自然语言：

**叙述逻辑：**
- 连续 3 epoch val_loss 下降 → "模型正在持续改善"
- val_loss 趋于平稳 → "验证损失趋于平稳，早停将自动触发"
- train_acc - val_acc > 15% → "存在过拟合迹象，建议增加数据增强"
- val_acc ≥ 90% → "验证准确率优秀"

支持中英文实时切换。

### 5.4 训练结果解读面板（T5.5）

训练完成后自动显示 `GET /api/train/{job_id}/interpret` 结果，以四格卡片布局呈现：

1. **发表就绪度** — 基于准确率评级（Excellent/Strong/Moderate/Weak）+ 中英文指导文本
2. **训练动态** — 过拟合诊断 + 使用轮数/是否早停
3. **最薄弱类别** — 识别 F1 最低的类别 + 增加样本建议
4. **建议下一步** — k 折验证、统计检验、导出图表的操作引导

---

## 六、v0.7 新增功能（模型可解释性 + 实时流式推理）

### 5.1 Grad-CAM 注意力热力图

> **新增模块：** `backend/services/gradcam.py` + `frontend/js/gradcam.js`

基于梯度加权类激活映射（Gradient-weighted Class Activation Mapping）的 1D-CNN 可解释性系统。可视化模型在做出分类决策时"关注"的信号区域，帮助科研人员理解和信任模型行为。

**核心原理：**

```
Input Signal → Forward Pass → Target Layer Activations
                                      ↓
           Backward Pass ← One-hot Target Class Gradient
                                      ↓
         GAP(Gradients) → Weights → Σ(Weight × Activation)
                                      ↓
                    ReLU → Upsample(interp) → Attention Heatmap [0,1]
```

**技术实现：**

| 组件 | 技术方案 | 说明 |
|------|---------|------|
| **激活捕获** | `register_forward_hook()` | 自动定位最后一层 Conv1d，捕获前向传播特征图 |
| **梯度捕获** | `register_full_backward_hook()` | 捕获目标类别反向传播梯度 |
| **权重计算** | Global Average Pooling | 对梯度沿空间维度取均值，得到各通道权重 |
| **热力图生成** | 加权求和 + ReLU + 线性插值上采样 | 从 feature_map_size → input_length |
| **批量计算** | `compute_gradcam_for_segments()` | 支持多段信号并行计算 |
| **自动层检测** | `_find_last_conv()` | 递归搜索模型最后一个 Conv1d 层 |

**前端可视化：**

| 功能 | 实现 | 说明 |
|------|------|------|
| **双轴叠加图** | Plotly 主轴(信号) + 次轴(注意力) | 信号曲线 + 蓝→红色注意力散点 |
| **高注意力区域标注** | 红色半透明矩形 shapes | 阈值 > 0.7 的连续高注意力区域自动标注 |
| **信息面板** | 3 列网格 | 预测结果、类别概率、Top-3 注意力区域 |
| **段选择器** | 下拉菜单 | 切换不同信号段的 Grad-CAM 结果 |
| **目标类别选择** | 下拉菜单 | 查看模型对不同类别的注意力差异 |

**API 端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/gradcam/{file_id}` | POST | 对上传文件计算 Grad-CAM（参数：model_id, channel, target_class, max_segments） |
| `/api/train/{job_id}/gradcam` | GET | 对训练任务验证集样本计算 Grad-CAM |

### 5.2 实时流式推理（WebSocket）

> **新增模块：** `backend/services/streaming.py` + `backend/routers/streaming.py` + `frontend/js/streaming.js`

面向可穿戴设备和实时监测场景的流式推理系统。通过 WebSocket 双向通信，实现信号的实时接收、预处理、推理和告警。

**系统架构：**

```
可穿戴设备/Demo生成器
        │
        ▼ (WebSocket JSON)
  ┌─────────────────────────────┐
  │     StreamingSession        │
  │  ┌───────────────────────┐  │
  │  │ Deque Buffer          │  │  ← samples 持续追加
  │  │ (maxlen = seg_len×4)  │  │
  │  └───────────┬───────────┘  │
  │              ▼              │
  │  ┌───────────────────────┐  │
  │  │ SOS Bandpass Filter   │  │  ← 实时二阶节带通滤波
  │  │ (scipy sosfilt)       │  │
  │  └───────────┬───────────┘  │
  │              ▼              │
  │  ┌───────────────────────┐  │
  │  │ Sliding Window Infer  │  │  ← 每 stride 个样本触发一次
  │  │ stride = seg_len/2    │  │
  │  └───────────┬───────────┘  │
  │              ▼              │
  │  ┌───────────────────────┐  │
  │  │ Alert Detection       │  │  ← 可配置类别+置信度阈值
  │  └───────────────────────┘  │
  └─────────────────────────────┘
        │
        ▼ (WebSocket JSON)
  前端 Monitor 仪表盘
```

**双模式支持：**

| 模式 | 数据源 | 说明 |
|------|--------|------|
| **Demo 模式** | 服务端合成信号 | 内置 ECG PQRST 波形生成器（含 PVC 异常注入）+ EEG 多频段合成 |
| **Device 模式** | 客户端推送 | 可穿戴设备/采集卡通过 WebSocket 发送原始样本 |

**合成信号生成器：**

| 类型 | 算法 | 说明 |
|------|------|------|
| **ECG** | PQRST 高斯组合波形 | P波+QRS复合波+T波，支持心率配置，~5% 概率注入 PVC 异常 |
| **EEG** | 多频段正弦叠加 | Delta(0.5-4Hz) + Theta(4-8Hz) + Alpha(8-13Hz) + Beta(13-30Hz) |

**信号处理参数（按信号类型）：**

| 参数 | ECG | EEG | EMG |
|------|-----|-----|-----|
| 默认采样率 | 360 Hz | 100 Hz | 200 Hz |
| 段长度 | 187 samples | 3000 samples | 80 samples |
| 推理步进 | 93 (~50%) | 500 (~17%) | 40 (50%) |
| 带通范围 | 0.5–40 Hz | 0.5–45 Hz | 20–450 Hz |

**WebSocket 通信协议：**

| 方向 | 消息类型 | 说明 |
|------|---------|------|
| Client→Server | `start` | 启动会话（model_id, mode, sampling_rate, heart_rate） |
| Client→Server | `samples` | 推送原始样本（Device 模式） |
| Client→Server | `stop` | 停止流式推理 |
| Client→Server | `configure_alerts` | 配置告警类别和置信度阈值 |
| Server→Client | `config` | 连接成功配置信息（模型/类别/采样率） |
| Server→Client | `samples` | 转发信号数据（用于前端绘图） |
| Server→Client | `prediction` | 实时推理结果（类别/置信度/概率分布） |
| Server→Client | `alert` | 异常告警通知（高置信度异常检测） |
| Server→Client | `stats` | 会话统计（样本数/预测数/有效采样率） |

**前端 Monitor 仪表盘：**

| 组件 | 技术 | 说明 |
|------|------|------|
| **实时信号图** | Plotly.js scattergl + setInterval(80ms) | ~12 FPS 滚动信号波形，2000 样本显示窗口 |
| **实时预测面板** | DOM 动态更新 | 当前预测类别 + 各类概率横条图 |
| **类别分布饼图** | Plotly.js donut chart | 实时更新的预测类别分布统计 |
| **告警日志** | 带动画的告警条目列表 | slide-in 动画 + 脉冲徽章计数器 |
| **会话统计栏** | 5 指标面板 | 运行时间/样本数/预测数/有效采样率/告警数 |

---

## 六、v0.6 新增功能（科幻登录页重设计）

### 5.1 "神经网关" 沉浸式登录体验

> **新增模块：** `frontend/js/login-animations.js`（270+ 行）

将原有简约登录页升级为全屏沉浸式科幻交互体验，包含 **12 种独立动画效果**，核心亮点：

1. **Canvas 粒子神经网络**：90 个发光粒子节点在画布上漂浮，距离阈值内自动连线形成网络拓扑图，粒子会被鼠标吸引并跟随移动
2. **实时生物信号波形**：Canvas 实时绘制 ECG 心电图 QRS 复合波 + EEG 脑电图多频叠加波，持续流动不间断
3. **HUD 军事风数据面板**：四角浮动面板显示系统状态（神经链路延迟、信号采样率、加密算法、CNN 参数量）
4. **鼠标光晕追踪**：实时径向渐变光晕跟随鼠标位置渲染
5. **Logo 扫描环**：三层旋转弧线 + CSS 双层反向旋转边框 + 呼吸脉冲光效

**性能优化：** 进入主应用后自动销毁 Canvas 动画和 DOM 元素，零内存泄漏。

---

## 七、v0.5 功能（P0 用户认证系统）

### 7.0.1 JWT 用户认证

> **新增模块：** `backend/auth.py` + `backend/routers/auth.py` + `backend/database.py` + `backend/models/user.py`

面向多用户场景的完整认证系统。用户注册/登录后获取 JWT Token，前端自动携带 Token 访问所有 API。

| 功能 | 技术方案 | 说明 |
|------|---------|------|
| **用户注册** | bcrypt 密码哈希 + JWT 签发 | 邮箱+用户名唯一约束，密码最少6位 |
| **用户登录** | 用户名或邮箱均可登录 | 验证密码后签发 24h 有效 JWT |
| **Token 鉴权** | OAuth2 Bearer Token | FastAPI 依赖注入，可选或强制鉴权 |
| **会话持久化** | localStorage 存储 Token | 页面刷新后自动恢复登录状态 |

**API 端点：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/auth/register` | POST | 注册（email, username, password）→ 返回 JWT |
| `/api/auth/login` | POST | 登录（username/email + password）→ 返回 JWT |
| `/api/auth/me` | GET | 获取当前用户信息（需 Bearer Token） |

### 7.0.2 数据库层

> **新增模块：** `backend/database.py` — SQLAlchemy 异步 ORM

| 环境 | 数据库 | 配置方式 |
|------|--------|---------|
| **本地开发** | SQLite（`biospark.db`） | 默认，零配置 |
| **Railway 生产** | PostgreSQL 16 | `DATABASE_URL` 环境变量自动注入 |
| **Docker Compose** | PostgreSQL 16 Alpine | `docker-compose.yml` 已配置 |

**数据表：**

| 表名 | 字段 | 说明 |
|------|------|------|
| `users` | id, email, username, hashed_password, is_active, created_at | 用户账户表 |

应用启动时自动执行 `init_db()` 创建表结构，无需手动迁移。

### 7.0.3 前端认证 UI

> **模块：** `frontend/js/auth.js` + `frontend/js/login-animations.js`（🆕 v0.6 新增）

- **登录/注册按钮**：Header 右上角，与语言切换按钮并排
- **登录后显示**：用户名 + 退出按钮
- **全局 Auth Headers**：所有 `fetch` 请求自动携带 `Authorization: Bearer <token>`

#### 🆕 v0.6 科幻"神经网关"登录页

> **新增模块：** `frontend/js/login-animations.js` — Canvas 粒子网络 + 生物信号波形 + HUD 装饰

全页面沉浸式科幻登录体验，包含大量人机交互动画：

| 动画效果 | 技术实现 | 说明 |
|---------|---------|------|
| **粒子神经网络** | Canvas 2D + requestAnimationFrame | 90 个粒子节点，自动连线，**跟随鼠标交互** |
| **生物信号波形** | Canvas 实时绘制 ECG/EEG 波形 | 顶部 ECG 心电波形 + 底部 EEG 脑电波形，持续流动 |
| **Logo 扫描环** | Canvas 旋转弧线 + CSS 旋转环 | 三层旋转扫描环 + CSS 双层旋转边框 |
| **HUD 数据面板** | DOM + CSS 动画 | 四角浮动数据面板（神经链路/信号处理/安全模块/AI模型） |
| **鼠标光晕追踪** | Canvas 径向渐变 | 光标位置实时渲染发光光晕 |
| **状态栏** | CSS 动画脉冲 | 顶部 "NEURAL GATEWAY v2.0" + 呼吸灯状态指示 |
| **扫描线** | CSS @keyframes | 卡片顶部渐变扫描线持续流动 |
| **打字机效果** | JS setInterval | 副标题逐字打出 + 光标闪烁 |
| **六角形装饰** | CSS clip-path + 旋转动画 | 6 个随机浮动的六角形边框 |
| **HUD 角标** | CSS 固定定位边框 | 四角 L 型科技边框 |
| **信号条** | SVG stroke-dashoffset 动画 | 底部 ECG 波形 SVG 绘制动画 |
| **交错入场** | JS 延迟 + CSS transition | 页面元素依次从下方淡入 |

**设计风格：**
- 深空蓝背景 `#060d1f` + 霓虹青/靛/粉三色光效
- Courier New 等宽字体 + 全大写字母间距，营造终端/HUD 质感
- 毛玻璃卡片 `backdrop-filter: blur(30px)` + 半透明层叠
- 响应式适配：移动端自动隐藏 HUD 面板和角标装饰

---

## 八、全部已完成功能

### 8.1 推理分析模块

| 功能 | 状态 | 说明 |
|------|------|------|
| 多格式数据上传 | ✅ | CSV, EDF, MAT, TXT, 拖拽上传 |
| 信号自动识别 | ✅ | 根据采样率和通道名自动判断 ECG/EEG/EMG |
| 时域/频域可视化 | ✅ | Plotly 交互图表，支持 FFT 频谱分析 |
| EEG 频段标注 | ✅ | Delta/Theta/Alpha/Beta/Gamma 彩色标注 |
| 多模型推理 | ✅ | ECG 5分类 + EMG 53分类（EEG 待完成）|
| 多通道 EMG 推理 | ✅ | 16 通道 sEMG 输入，自动预处理 |
| 置信度分析 | ✅ | 逐段预测 + 概率分布 + 汇总统计 |
| 结果导出 | ✅ | JSON / CSV 格式导出 |
| 🆕 **Grad-CAM 注意力热力图** | ✅ | **PyTorch Hook 捕获梯度，双轴叠加可视化，高注意力区域标注** |
| 🆕 **实时流式推理** | ✅ | **WebSocket 双向通信，滑窗推理，Demo/Device 双模式** |
| 🆕 **异常告警系统** | ✅ | **可配置告警类别+置信度阈值，实时告警通知** |

### 8.2 模型训练模块（Phase 1-7）

| 阶段 | 功能 | 说明 |
|------|------|------|
| **Phase 1** | 标注数据上传 | CSV（含 label 列）或 ZIP（文件夹分类）|
| **Phase 2** | CNN 训练引擎 | 可配置 epochs、lr、batch_size、val_split、n_channels |
| **Phase 2** | 实时训练监控 | WebSocket 推送每轮 loss/accuracy，Plotly 实时绘图 |
| **Phase 2** | 自动超参优化 | LR finder + 架构自适应 + 类别权重 + Early Stopping |
| **Phase 3** | 混淆矩阵 | 热力图 + 计数/百分比切换 + 每类 P/R/F1 |
| **Phase 3** | t-SNE 可视化 | 倒数第二层特征 → 2维散点图 |
| **Phase 4** | 模型导出 | .pt 模型文件、训练历史 JSON |
| **Phase 4** | HTML 报告 | 自包含 Plotly 图表的完整训练报告 |
| **Phase 5** | 多通道支持 | 自动检测 ch{N}_ 前缀列，Conv1d 多通道输入 |
| **Phase 6** | 出版质量图表 | 5种图表 × 3种期刊样式 × PNG+SVG，一键ZIP下载 |
| **Phase 6** | 模型架构图 | 自动生成 CNN 结构示意图，可直接用于论文 |
| **Phase 7** | 🆕 **训练后 Grad-CAM** | **验证集样本注意力分析，含真实标签对比** |

### 8.3 UI/UX

- **暗色科技风主题**：深色背景(#0f172a)、毛玻璃卡片、霓虹青/粉色调
- **Plotly 深色适配**：所有图表统一深色主题
- **中英双语**：EN/中文 一键切换
- **响应式布局**：移动端到桌面端
- **Auto-Optimize 开关**：训练配置区一键启用自动优化
- **期刊样式选择器**：Nature / IEEE / Science 三种学术风格一键切换
- **图表预览面板**：T6 区块展示 5 张出版级图表，逐图 PNG/SVG 下载
- **用户认证 UI**：Header 登录/注册按钮 + 模态弹窗，登录后显示用户名
- **科幻登录页**：Canvas 粒子神经网络 + ECG/EEG 波形动画 + HUD 面板 + 鼠标交互光效
- 🆕 **Grad-CAM 可视化面板**：双轴信号+注意力叠加图、段选择器、类别选择器、3列信息面板
- 🆕 **Monitor 实时监控仪表盘**：实时信号图、预测概率条、分布饼图、告警日志、统计面板
- 🆕 **三模式切换**：Analyze / Train / Monitor 顶部标签页一键切换

### 8.4 部署

| 平台 | 状态 | 功能范围 |
|------|------|---------|
| **Railway** | ✅ 已上线 | 全功能（推理+训练+流式推理+Grad-CAM+出版图表+用户认证+PostgreSQL） |
| **Vercel** | ✅ 已上线 | 前端 + 轻量 API |
| **Docker** | ✅ 可用 | 本地/私有云一键部署（含 PostgreSQL 服务） |

---

## 九、CNN 模型架构详情

### Signal1DCNN（通用训练架构，v0.4 升级）

```
Input: (batch, in_channels, signal_length)

  ┌── arch_config 动态配置 ──────────────────────────────────┐
  │  kernel_sizes: [k1, k2, k3]  (默认 [7, 5, 3])          │
  │  channels:     [c1, c2, c3]  (默认 [32, 64, 128])       │
  │  dropout1/2:   d1, d2        (默认 0.3, 0.2)            │
  │  fc_hidden:    h             (默认 64)                   │
  └──────────────────────────────────────────────────────────┘

  Conv1d(in_ch, c1, k=k1) -> BN -> ReLU -> MaxPool(2)
  Conv1d(c1, c2, k=k2) -> BN -> ReLU -> MaxPool(2)
  Conv1d(c2, c3, k=k3) -> BN -> ReLU -> AdaptiveAvgPool(1)
  Dropout(d1) -> FC(c3 -> h) -> ReLU -> Dropout(d2) -> FC(h -> n_classes)

Output: n_classes probabilities
```

### ECGArrhythmiaCNN

```
Input: (batch, 1, 187)
  Conv1d(1, 32, k=7) -> BN -> ReLU -> MaxPool(2)
  Conv1d(32, 64, k=5) -> BN -> ReLU -> MaxPool(2)
  Conv1d(64, 128, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)
  FC(128->64) -> ReLU -> Dropout(0.2) -> FC(64->5)
Output: 5-class probabilities  |  Params: 44K
```

### EEGSleepCNN

```
Input: (batch, 1, 3000)
  Conv1d(1, 32, k=25) -> BN -> ReLU -> MaxPool(4)     # 3000->750
  Conv1d(32, 64, k=15) -> BN -> ReLU -> MaxPool(4)     # 750->187
  Conv1d(64, 128, k=7) -> BN -> ReLU -> MaxPool(4)     # 187->46
  Conv1d(128, 256, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)
  FC(256->128) -> ReLU -> FC(128->64) -> ReLU -> FC(64->5)
Output: 5-class probabilities  |  Params: ~150K
```

### EMGGestureCNN

```
Input: (batch, 16, 80)     # 16 sEMG channels x 400ms
  Conv1d(16, 64, k=5) -> BN -> ReLU -> MaxPool(2)      # 80->40
  Conv1d(64, 128, k=5) -> BN -> ReLU -> MaxPool(2)     # 40->20
  Conv1d(128, 256, k=3) -> BN -> ReLU -> MaxPool(2)    # 20->10
  Conv1d(256, 256, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)
  FC(256->128) -> ReLU -> FC(128->64) -> ReLU -> FC(64->53)
Output: 53-class probabilities  |  Params: 388K
```

---

## 十、API 接口清单

### 10.1 推理 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/models` | GET | 获取可用模型列表（ECG/EEG/EMG） |
| `/api/upload` | POST | 上传并解析信号文件 |
| `/api/analyze/{id}` | POST | 运行模型推理（支持多通道） |

### 10.2 训练 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/train/upload` | POST | 上传标注训练数据集 |
| `/api/train/start` | POST | 启动训练任务（支持 auto_mode） |
| `/api/train/ws/{id}` | WebSocket | 实时训练指标流 |
| `/api/train/{id}/status` | GET | 轮询训练状态 |
| `/api/train/{id}/confusion_matrix` | GET | 获取混淆矩阵 |
| `/api/train/{id}/tsne` | GET | 获取 t-SNE 投影 |
| `/api/train/{id}/gradcam` | GET | 🆕 训练后 Grad-CAM 注意力分析 |

### 10.3 导出 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/train/{id}/export/model` | GET | 下载 .pt 模型 |
| `/api/train/{id}/export/history` | GET | 下载训练历史 JSON |
| `/api/train/{id}/export/confusion_matrix_csv` | GET | 下载混淆矩阵 CSV |
| `/api/train/{id}/export/tsne_csv` | GET | 下载 t-SNE CSV |
| `/api/train/{id}/export/report` | GET | 下载 HTML 报告 |

### 10.4 出版图表 API（v0.4 新增）

| 端点 | 方法 | 参数 | 功能 |
|------|------|------|------|
| `/api/train/{id}/figures/training_curves` | GET | style, fmt | 训练曲线图 |
| `/api/train/{id}/figures/confusion_matrix` | GET | mode, style, fmt | 混淆矩阵热力图 |
| `/api/train/{id}/figures/tsne` | GET | style, fmt | t-SNE 散点图 |
| `/api/train/{id}/figures/per_class_metrics` | GET | style, fmt | 各类别指标柱状图 |
| `/api/train/{id}/figures/architecture` | GET | style, fmt | 模型架构示意图 |
| `/api/train/{id}/figures/all.zip` | GET | style | 全部图表打包下载 |

**公共参数：** `style` = nature | ieee | science，`fmt` = png | svg

### 10.5 用户认证 API（v0.5 新增）

| 端点 | 方法 | 请求体 | 功能 |
|------|------|--------|------|
| `/api/auth/register` | POST | `{email, username, password}` | 用户注册，返回 JWT Token |
| `/api/auth/login` | POST | `{username, password}` | 用户登录（支持邮箱/用户名），返回 JWT Token |
| `/api/auth/me` | GET | — (Bearer Token) | 获取当前登录用户信息 |

**响应格式（register/login）：**
```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer",
  "user": {"id": 1, "email": "...", "username": "...", "created_at": "..."}
}
```

### 10.6 🆕 Grad-CAM API（v0.7 新增）

| 端点 | 方法 | 参数 | 功能 |
|------|------|------|------|
| `/api/gradcam/{file_id}` | POST | model_id, channel, target_class, max_segments | 对上传文件计算 Grad-CAM 注意力热力图 |
| `/api/train/{job_id}/gradcam` | GET | max_segments | 对训练任务验证集计算 Grad-CAM |

**响应格式：**
```json
{
  "gradcam": [
    {
      "segment_index": 0,
      "heatmap": [0.12, 0.34, ...],
      "predicted_class": "Normal (N)",
      "predicted_index": 0,
      "confidence": 0.97,
      "probabilities": {"Normal (N)": 0.97, ...}
    }
  ],
  "classes": ["Normal (N)", "Supraventricular (S)", ...],
  "signal_type": "ecg"
}
```

### 10.7 🆕 实时流式推理 API（v0.7 新增）

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/stream/ws` | WebSocket | 双向流式推理（JSON 协议，支持 demo/device 模式） |

**WebSocket 消息协议：** 见第五章 5.2 节通信协议表。

---

## 十一、项目文件结构（v0.7）

```
backend/
├── main.py                       # FastAPI 主入口 + 路由注册 + Lifespan DB 初始化
├── config.py                     # MODEL_REGISTRY, PREPROCESS_CONFIG
├── auth.py                       # JWT 创建/验证、bcrypt 哈希、FastAPI 鉴权依赖
├── database.py                   # SQLAlchemy 异步引擎、Session 工厂、init_db()
├── routers/
│   ├── auth.py                   # 用户认证 API（register/login/me）
│   ├── upload.py                 # 文件上传解析
│   ├── analysis.py               # 模型推理 + 🆕 Grad-CAM 端点 + _prepare_segments()
│   ├── models.py                 # 模型列表
│   ├── training.py               # 训练 API + WebSocket + 🆕 训练后 Grad-CAM
│   ├── figures.py                # 出版图表 API (Phase 6)
│   └── streaming.py              # 🆕 WebSocket 流式推理端点（demo/device 模式）
├── models/
│   ├── user.py                   # User 数据库模型（SQLAlchemy ORM）
│   ├── ecg_arrhythmia_cnn.pt     # 94.1% 准确率
│   ├── eeg_sleep_staging.pt      # 训练中
│   └── emg_gesture_cnn.pt        # 42.7% 准确率
└── services/
    ├── format_parser.py          # CSV/EDF/MAT 解析
    ├── preprocess.py             # ECG/EEG/EMG 预处理
    ├── predictor.py              # PyTorch/ONNX/Demo 推理
    ├── trainer.py                # Signal1DCNN + 训练循环（支持动态架构）
    ├── dataset_loader.py         # 标注数据集解析
    ├── auto_optimizer.py         # LR finder / Early Stopping / 类别权重 / 架构选择
    ├── publication_figures.py    # Matplotlib 出版图表渲染（5种）
    ├── gradcam.py                # 🆕 GradCAM1D — PyTorch Hook 注意力热力图引擎
    └── streaming.py              # 🆕 StreamingSession — 流式推理 + 合成信号生成

frontend/
├── index.html                    # 主页（🔄 v0.7 新增 Grad-CAM 区块 + Monitor 模式）
├── css/style.css                 # 暗色主题（🔄 v0.7 +385行 Grad-CAM/Streaming 样式）
└── js/
    ├── auth.js                   # 认证模块（登录/注册/Token 管理/模态弹窗）
    ├── login-animations.js       # 科幻登录页动画（Canvas粒子网络/波形/HUD）
    ├── app.js                    # 语言切换、模式切换（🔄 Monitor 模式 + Grad-CAM 触发）
    ├── uploader.js               # 推理文件上传
    ├── visualizer.js             # Plotly 信号可视化
    ├── results.js                # 推理结果展示
    ├── trainer.js                # 训练控制台（🔄 训练后 Grad-CAM 调用）
    ├── figures.js                # 出版图表预览 + 下载
    ├── gradcam.js                # 🆕 Grad-CAM 双轴叠加可视化 + 信息面板
    └── streaming.js              # 🆕 实时流式推理 WebSocket 客户端 + Monitor 仪表盘

training/
├── train_ecg_arrhythmia.py       # MIT-BIH → 94.1%
├── train_eeg_sleep.py            # Sleep-EDF（进行中）
├── train_emg_gesture.py          # NinaPro DB5 → 53 类
└── export_onnx.py                # PyTorch → ONNX 转换
```

---

## 十二、未来目标

### 12.1 短期目标（1-3 个月）

| 目标 | 优先级 | 说明 |
|------|--------|------|
| **EEG 睡眠分期模型完善** | P0 | 完成 Sleep-EDF 训练，优化准确率 |
| **EMG 模型精度提升** | P0 | 数据增强、注意力机制、per-subject fine-tuning |
| ~~注意力热力图~~ | ~~P1~~ | ✅ **v0.7 已完成** — Grad-CAM 1D 注意力可视化 |
| ~~用户认证系统~~ | ~~P0~~ | ✅ **v0.5 已完成** — JWT + bcrypt + PostgreSQL |
| ~~数据库集成~~ | ~~P0~~ | ✅ **v0.5 已完成** — SQLAlchemy async ORM + PostgreSQL |
| **训练历史持久化** | P0 | 将训练记录关联用户，存入数据库（v0.5 基础设施已就绪） |
| **批量处理 API** | P1 | 支持上传多文件批量分析 |

### 12.2 中期目标（3-6 个月）

| 目标 | 说明 |
|------|------|
| **预训练模型市场** | 用户可上传/分享训练好的模型 |
| **HuggingFace 集成** | 接入 ECGFounder、U-Sleep 等社区模型 |
| **数据增强策略** | 时间扭曲、噪声注入、窗口滑动等信号增强 |
| ~~实时流式推理~~ | ✅ **v0.7 已完成** — WebSocket 流式推理 + Demo/Device 双模式 + 异常告警 |
| **团队协作** | 多用户项目空间，共享数据集和模型 |

### 12.3 长期愿景（6-12 个月）

| 目标 | 说明 |
|------|------|
| **联邦学习** | 多机构数据不出本地，协同训练 |
| **FDA/CE 合规** | 医疗器械软件认证流程 |
| **移动端 SDK** | iOS/Android SDK 嵌入 APP |
| **边缘推理** | TensorRT/CoreML 优化，支持嵌入式设备 |

---

## 十三、盈利策略

### 13.1 分层定价

| 版本 | 价格 | 目标用户 | 核心功能 |
|------|------|---------|---------|
| **免费版** | ¥0 | 学生/入门 | 单文件上传、基础推理、50次/月 |
| **专业版** | ¥99/月 | 研究者/小实验室 | 无限推理、CNN训练(GPU)、模型导出、出版图表 |
| **企业版** | ¥4,999+/月 | 医院/医疗公司 | 私有部署、API批量调用、合规审计 |

### 13.2 增值收入

| 收入来源 | 模式 | 预估 |
|----------|------|------|
| 模型市场 | 社区交易抽成 15% | 随规模增长 |
| GPU 算力 | 按时计费 ¥2-5/GPU·h | 训练用户 |
| 定制开发 | 企业定制模型 | ¥50K-500K/项目 |
| 学术授权 | 高校年度授权 | ¥5K-20K/年 |

---

## 十四、关键指标

### 14.1 技术指标

| 指标 | 当前值 |
|------|--------|
| ECG 心律失常分类准确率 | 94.1%（MIT-BIH） |
| EMG 手势识别准确率 | 42.7%（NinaPro DB5，53类） |
| EMG 基础手指动作准确率 | 48.1%（E1，12类） |
| EMG Rest 识别准确率 | 84.8% |
| 预训练模型数量 | 3 个（ECG/EEG/EMG） |
| EMG 支持手势数 | 52 种 + Rest |
| EMG 输入通道数 | 16 通道 sEMG |
| 支持信号格式数 | 4 种（CSV/EDF/MAT/TXT） |
| API 端点数 | 31 个（🆕 +4：2 Grad-CAM + 1 Streaming WS + 1 训练Grad-CAM） |
| 出版图表类型 | 5 种 |
| 支持期刊样式 | 3 种（Nature/IEEE/Science） |
| 图表输出分辨率 | 300 DPI PNG + SVG 矢量 |
| 用户认证 | JWT + bcrypt，24h Token 有效期 |
| 数据库 | PostgreSQL（生产）/ SQLite（开发） |
| 🆕 模型可解释性 | Grad-CAM 1D 注意力热力图（PyTorch Hook） |
| 🆕 实时推理延迟 | ~50ms/prediction（滑窗推理 + SOS 滤波） |
| 🆕 流式推理模式 | 2 种（Demo 合成信号 / Device 设备接入） |
| Docker 镜像大小 | ~1.5 GB |

### 14.2 项目进度

| 里程碑 | 完成日期 | 内容 |
|--------|---------|------|
| v0.1 — 基础平台 | 2026-04-01 | 文件上传、信号可视化、ECG 推理 |
| v0.2 — 训练系统 | 2026-04-02 | 5 阶段训练管线、WebSocket、导出 |
| v0.2.1 — 部署 | 2026-04-05 | Railway/Vercel 双平台上线 |
| v0.2.2 — UI 重设计 | 2026-04-05 | 暗色科技风主题、Plotly 深色适配 |
| v0.3 — 模型扩充 | 2026-04-06 | EEG 睡眠分期 + EMG 53类手势识别（NinaPro DB5 真实数据） |
| v0.4 — 科研升级 | 2026-04-08 | 自动超参优化 + 出版质量图表(5种) + 模型架构图 + 3种期刊样式 |
| v0.5 — 用户系统 | 2026-04-14 | JWT 用户认证 + PostgreSQL 数据库 + 前端登录/注册 UI |
| v0.6 — 科幻 UI | 2026-04-15 | 神经网关沉浸式登录页（12种动画 + Canvas 粒子网络） |
| **v0.7 — 可解释性+实时** | **2026-04-16** | **Grad-CAM 注意力热力图 + WebSocket 流式推理 + Monitor 仪表盘 + 异常告警** |

---

## 十五、新增依赖

| 包 | 版本 | 用途 |
|---|------|------|
| sqlalchemy[asyncio] | ≥2.0.0 | 异步 ORM，支持 SQLite / PostgreSQL |
| aiosqlite | ≥0.20.0 | SQLite 异步驱动（本地开发） |
| asyncpg | ≥0.30.0 | PostgreSQL 异步驱动（生产环境） |
| python-jose[cryptography] | ≥3.3.0 | JWT Token 签发与验证 |
| bcrypt | ≥4.0.0 | 密码哈希（直接使用，不依赖 passlib） |
| pydantic[email] | ≥2.0.0 | EmailStr 验证 |

**v0.7 新增依赖：** 无新增 pip 包（Grad-CAM 仅依赖已有的 PyTorch + NumPy，流式推理使用 FastAPI 内置 WebSocket + SciPy）

**v0.4 已有依赖（保留）：** matplotlib ≥3.8.0, seaborn ≥0.13.0

**环境变量：**

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DATABASE_URL` | 生产环境必需 | `sqlite+aiosqlite:///./biospark.db` | Railway PostgreSQL 插件自动注入 |
| `JWT_SECRET_KEY` | 生产环境必需 | `biospark-dev-secret-...` | 用于签发 JWT，生产环境务必设为随机强密码 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 可选 | `1440`（24小时） | JWT Token 有效期 |

---

## 十六、风险与应对

| 风险 | 等级 | 应对策略 |
|------|------|---------|
| 数据隐私合规 | 高 | HIPAA/GDPR 合规设计，支持私有化部署 |
| EMG 模型准确率需提升 | 中 | 数据增强、注意力机制、迁移学习、per-subject 微调 |
| 用户获取成本高 | 中 | 免费层引流 + 学术论文合作 + 开源社区 |
| 技术壁垒低 | 中 | 构建数据壁垒（标注数据集）+ 先发优势 |
| 图表字体兼容性 | 低 | 多字体回退链（Arial→Helvetica→DejaVu Sans） |

---

## 十七、总结

BioSpark v0.7 在前版基础上新增了**模型可解释性（Grad-CAM 注意力热力图）**和**实时流式推理（WebSocket Monitor）**两大核心功能，平台从离线分析扩展到实时监测场景，同时通过可解释性技术增强了模型的可信度。

**v0.7 核心价值：**
1. **Grad-CAM 注意力热力图** — PyTorch Hook 自动捕获梯度，双轴叠加可视化，支持推理和训练两种场景，帮助科研人员理解 CNN 决策依据
2. **实时流式推理** — WebSocket 双向通信，SOS 带通滤波 + 滑窗推理，Demo 合成信号（ECG PQRST + PVC 异常注入）和 Device 设备接入双模式
3. **Monitor 仪表盘** — 12 FPS 实时信号图 + 预测概率条 + 分布饼图 + 告警日志 + 统计面板，临床级监护体验
4. **异常告警系统** — 可配置告警类别和置信度阈值，脉冲动画徽章实时通知

**平台核心竞争力：**
1. **零代码门槛** — 浏览器即用，无需 Python/MATLAB 知识
2. **三信号全覆盖** — ECG 心律失常 + EEG 睡眠分期 + EMG 手势识别
3. **端到端训练** — 上传数据到导出模型 + 论文图表，7 步完成
4. **科研级输出** — 出版质量图表直接满足 Nature/IEEE/Science 投稿要求
5. **真实数据验证** — 基于 MIT-BIH、NinaPro DB5 等权威数据集
6. **多用户支持** — 注册登录系统 + PostgreSQL 持久化，具备商业化基础
7. 🆕 **模型可解释性** — Grad-CAM 注意力热力图，增强模型可信度和科研价值
8. 🆕 **实时监护能力** — WebSocket 流式推理 + 异常告警，支持可穿戴设备接入

**下一步重点：** 训练历史持久化（关联用户） → EMG 精度优化 → EEG 模型完成 → 团队协作 → 商业化。

---

*本报告由 BioSpark 团队编写，基于 v0.8.0 版本（2026-04-19）。*
