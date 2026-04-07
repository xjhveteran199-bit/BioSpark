# BioSpark 项目研发报告

> **版本:** v0.4.0 | **日期:** 2026-04-08 | **状态:** 已上线运营

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
| **深度学习** | PyTorch 2.x (CPU) | 1D-CNN 训练与推理 |
| **信号处理** | NeuroKit2 / MNE / SciPy | ECG R-peak 检测、EEG 分段、EMG 滤波 |
| **数据分析** | NumPy / Pandas / Scikit-learn | t-SNE、混淆矩阵、特征提取 |
| **科研可视化** | Matplotlib / Seaborn | 300 DPI PNG + SVG 出版质量图表 |
| **推理引擎** | PyTorch + ONNX Runtime | .pt 主力推理 + 浏览器端 ONNX |
| **前端** | 原生 HTML/CSS/JS + Plotly.js | 零构建步骤，交互式可视化 |
| **部署** | Docker / Railway / Vercel | 容器化 + Serverless 双模式 |

### 2.2 系统架构图

```
+--------------------------------------------------------------+
|                        前端 (Vanilla JS)                      |
|  +----------+ +----------+ +----------+ +------------------+ |
|  | 文件上传  | | 信号可视化| | 推理结果  | | 训练控制台(WS)   | |
|  +----+-----+ +----+-----+ +----+-----+ +--------+---------+ |
|  | 图表预览  | | 样式选择  | | 下载管理  | | 自动优化配置     | |
|  +----+-----+ +----+-----+ +----+-----+ +--------+---------+ |
+-------+------------+------------+-----------------+----------+
        | REST API   | Plotly.js  | REST API        | WebSocket
+-------+------------+------------+-----------------+----------+
|                     FastAPI 后端                               |
|  +----------+ +----------+ +----------+ +------------------+ |
|  |格式解析器 | |信号预处理 | |模型推理器 | | CNN训练引擎      | |
|  |CSV/EDF/  | |ECG/EEG/  | |PyTorch/  | | Signal1DCNN      | |
|  |MAT/ZIP   | |EMG多通道  | |ONNX/Demo | | + 自动超参优化    | |
|  +----------+ +----------+ +----------+ +------------------+ |
|  +----------+ +----------+                                    |
|  |出版图表   | |架构图生成 |  ← v0.4 新增                      |
|  |Matplotlib | |纯Patches |                                    |
|  +----------+ +----------+                                    |
+--------------------------------------------------------------+
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

## 五、全部已完成功能

### 5.1 推理分析模块

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

### 5.2 模型训练模块（Phase 1-6）

| 阶段 | 功能 | 说明 |
|------|------|------|
| **Phase 1** | 标注数据上传 | CSV（含 label 列）或 ZIP（文件夹分类）|
| **Phase 2** | CNN 训练引擎 | 可配置 epochs、lr、batch_size、val_split、n_channels |
| **Phase 2** | 实时训练监控 | WebSocket 推送每轮 loss/accuracy，Plotly 实时绘图 |
| **Phase 2** | **🆕 自动超参优化** | **LR finder + 架构自适应 + 类别权重 + Early Stopping** |
| **Phase 3** | 混淆矩阵 | 热力图 + 计数/百分比切换 + 每类 P/R/F1 |
| **Phase 3** | t-SNE 可视化 | 倒数第二层特征 → 2维散点图 |
| **Phase 4** | 模型导出 | .pt 模型文件、训练历史 JSON |
| **Phase 4** | HTML 报告 | 自包含 Plotly 图表的完整训练报告 |
| **Phase 5** | 多通道支持 | 自动检测 ch{N}_ 前缀列，Conv1d 多通道输入 |
| **Phase 6** | **🆕 出版质量图表** | **5种图表 × 3种期刊样式 × PNG+SVG，一键ZIP下载** |
| **Phase 6** | **🆕 模型架构图** | **自动生成 CNN 结构示意图，可直接用于论文** |

### 5.3 UI/UX

- **暗色科技风主题**：深色背景(#0f172a)、毛玻璃卡片、霓虹青/粉色调
- **Plotly 深色适配**：所有图表统一深色主题
- **中英双语**：EN/中文 一键切换
- **响应式布局**：移动端到桌面端
- 🆕 **Auto-Optimize 开关**：训练配置区一键启用自动优化
- 🆕 **期刊样式选择器**：Nature / IEEE / Science 三种学术风格一键切换
- 🆕 **图表预览面板**：T6 区块展示 5 张出版级图表，逐图 PNG/SVG 下载

### 5.4 部署

| 平台 | 状态 | 功能范围 |
|------|------|---------|
| **Railway** | ✅ 已上线 | 全功能（推理+训练+WebSocket+出版图表） |
| **Vercel** | ✅ 已上线 | 前端 + 轻量 API |
| **Docker** | ✅ 可用 | 本地/私有云一键部署 |

---

## 六、CNN 模型架构详情

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

## 七、API 接口清单

### 7.1 推理 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/models` | GET | 获取可用模型列表（ECG/EEG/EMG） |
| `/api/upload` | POST | 上传并解析信号文件 |
| `/api/analyze/{id}` | POST | 运行模型推理（支持多通道） |

### 7.2 训练 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/train/upload` | POST | 上传标注训练数据集 |
| `/api/train/start` | POST | 启动训练任务（🆕 支持 auto_mode） |
| `/api/train/ws/{id}` | WebSocket | 实时训练指标流 |
| `/api/train/{id}/status` | GET | 轮询训练状态 |
| `/api/train/{id}/confusion_matrix` | GET | 获取混淆矩阵 |
| `/api/train/{id}/tsne` | GET | 获取 t-SNE 投影 |

### 7.3 导出 API

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/train/{id}/export/model` | GET | 下载 .pt 模型 |
| `/api/train/{id}/export/history` | GET | 下载训练历史 JSON |
| `/api/train/{id}/export/confusion_matrix_csv` | GET | 下载混淆矩阵 CSV |
| `/api/train/{id}/export/tsne_csv` | GET | 下载 t-SNE CSV |
| `/api/train/{id}/export/report` | GET | 下载 HTML 报告 |

### 7.4 🆕 出版图表 API（v0.4 新增）

| 端点 | 方法 | 参数 | 功能 |
|------|------|------|------|
| `/api/train/{id}/figures/training_curves` | GET | style, fmt | 训练曲线图 |
| `/api/train/{id}/figures/confusion_matrix` | GET | mode, style, fmt | 混淆矩阵热力图 |
| `/api/train/{id}/figures/tsne` | GET | style, fmt | t-SNE 散点图 |
| `/api/train/{id}/figures/per_class_metrics` | GET | style, fmt | 各类别指标柱状图 |
| `/api/train/{id}/figures/architecture` | GET | style, fmt | 模型架构示意图 |
| `/api/train/{id}/figures/all.zip` | GET | style | 全部图表打包下载 |

**公共参数：** `style` = nature | ieee | science，`fmt` = png | svg

---

## 八、项目文件结构（v0.4）

```
backend/
├── main.py                       # FastAPI 主入口 + 路由注册
├── config.py                     # MODEL_REGISTRY, PREPROCESS_CONFIG
├── routers/
│   ├── upload.py                 # 文件上传解析
│   ├── analysis.py               # 模型推理
│   ├── models.py                 # 模型列表
│   ├── training.py               # 训练 API + WebSocket (Phase 1-5)
│   └── figures.py                # 🆕 出版图表 API (Phase 6)
├── services/
│   ├── format_parser.py          # CSV/EDF/MAT 解析
│   ├── preprocess.py             # ECG/EEG/EMG 预处理
│   ├── predictor.py              # PyTorch/ONNX/Demo 推理
│   ├── trainer.py                # Signal1DCNN + 训练循环（🔄 支持动态架构）
│   ├── dataset_loader.py         # 标注数据集解析
│   ├── auto_optimizer.py         # 🆕 LR finder / Early Stopping / 类别权重 / 架构选择
│   └── publication_figures.py    # 🆕 Matplotlib 出版图表渲染（5种）
└── models/
    ├── ecg_arrhythmia_cnn.pt     # 94.1% 准确率
    ├── eeg_sleep_staging.pt      # 训练中
    └── emg_gesture_cnn.pt        # 42.7% 准确率

frontend/
├── index.html                    # 主页（🔄 新增 T6 图表区块 + Auto-Optimize UI）
├── css/style.css                 # 暗色主题（🔄 新增图表面板样式）
└── js/
    ├── app.js                    # 语言切换、模式切换
    ├── uploader.js               # 推理文件上传
    ├── visualizer.js             # Plotly 信号可视化
    ├── results.js                # 推理结果展示
    ├── trainer.js                # 训练控制台（🔄 新增 getJobId / 图表触发）
    └── figures.js                # 🆕 出版图表预览 + 下载

training/
├── train_ecg_arrhythmia.py       # MIT-BIH → 94.1%
├── train_eeg_sleep.py            # Sleep-EDF（进行中）
├── train_emg_gesture.py          # NinaPro DB5 → 53 类
└── export_onnx.py                # PyTorch → ONNX 转换
```

---

## 九、未来目标

### 9.1 短期目标（1-3 个月）

| 目标 | 优先级 | 说明 |
|------|--------|------|
| **EEG 睡眠分期模型完善** | P0 | 完成 Sleep-EDF 训练，优化准确率 |
| **EMG 模型精度提升** | P0 | 数据增强、注意力机制、per-subject fine-tuning |
| **注意力热力图** | P1 | Grad-CAM 可视化 CNN 关注的信号区域 |
| **用户认证系统** | P0 | JWT 登录/注册，训练历史持久化 |
| **数据库集成** | P0 | PostgreSQL 存储用户数据、模型版本、训练记录 |
| **批量处理 API** | P1 | 支持上传多文件批量分析 |

### 9.2 中期目标（3-6 个月）

| 目标 | 说明 |
|------|------|
| **预训练模型市场** | 用户可上传/分享训练好的模型 |
| **HuggingFace 集成** | 接入 ECGFounder、U-Sleep 等社区模型 |
| **数据增强策略** | 时间扭曲、噪声注入、窗口滑动等信号增强 |
| **实时流式推理** | WebSocket 接入可穿戴设备实时数据流 |
| **团队协作** | 多用户项目空间，共享数据集和模型 |

### 9.3 长期愿景（6-12 个月）

| 目标 | 说明 |
|------|------|
| **联邦学习** | 多机构数据不出本地，协同训练 |
| **FDA/CE 合规** | 医疗器械软件认证流程 |
| **移动端 SDK** | iOS/Android SDK 嵌入 APP |
| **边缘推理** | TensorRT/CoreML 优化，支持嵌入式设备 |

---

## 十、盈利策略

### 10.1 分层定价

| 版本 | 价格 | 目标用户 | 核心功能 |
|------|------|---------|---------|
| **免费版** | ¥0 | 学生/入门 | 单文件上传、基础推理、50次/月 |
| **专业版** | ¥99/月 | 研究者/小实验室 | 无限推理、CNN训练(GPU)、模型导出、出版图表 |
| **企业版** | ¥4,999+/月 | 医院/医疗公司 | 私有部署、API批量调用、合规审计 |

### 10.2 增值收入

| 收入来源 | 模式 | 预估 |
|----------|------|------|
| 模型市场 | 社区交易抽成 15% | 随规模增长 |
| GPU 算力 | 按时计费 ¥2-5/GPU·h | 训练用户 |
| 定制开发 | 企业定制模型 | ¥50K-500K/项目 |
| 学术授权 | 高校年度授权 | ¥5K-20K/年 |

---

## 十一、关键指标

### 11.1 技术指标

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
| API 端点数 | 24 个（🆕 +6 图表端点） |
| 出版图表类型 | 🆕 5 种 |
| 支持期刊样式 | 🆕 3 种（Nature/IEEE/Science） |
| 图表输出分辨率 | 🆕 300 DPI PNG + SVG 矢量 |
| Docker 镜像大小 | ~1.5 GB |

### 11.2 项目进度

| 里程碑 | 完成日期 | 内容 |
|--------|---------|------|
| v0.1 — 基础平台 | 2026-04-01 | 文件上传、信号可视化、ECG 推理 |
| v0.2 — 训练系统 | 2026-04-02 | 5 阶段训练管线、WebSocket、导出 |
| v0.2.1 — 部署 | 2026-04-05 | Railway/Vercel 双平台上线 |
| v0.2.2 — UI 重设计 | 2026-04-05 | 暗色科技风主题、Plotly 深色适配 |
| v0.3 — 模型扩充 | 2026-04-06 | EEG 睡眠分期 + EMG 53类手势识别（NinaPro DB5 真实数据） |
| **v0.4 — 科研升级** | **2026-04-08** | **自动超参优化 + 出版质量图表(5种) + 模型架构图 + 3种期刊样式** |

---

## 十二、v0.4 新增依赖

| 包 | 版本 | 用途 |
|---|------|------|
| matplotlib | ≥3.8.0 | 出版质量图表渲染（PNG/SVG） |
| seaborn | ≥0.13.0 | 色彩方案辅助 |

*无需 Graphviz 系统依赖——架构图使用纯 Matplotlib Patches 绘制。*

---

## 十三、风险与应对

| 风险 | 等级 | 应对策略 |
|------|------|---------|
| 数据隐私合规 | 高 | HIPAA/GDPR 合规设计，支持私有化部署 |
| EMG 模型准确率需提升 | 中 | 数据增强、注意力机制、迁移学习、per-subject 微调 |
| 用户获取成本高 | 中 | 免费层引流 + 学术论文合作 + 开源社区 |
| 技术壁垒低 | 中 | 构建数据壁垒（标注数据集）+ 先发优势 |
| 图表字体兼容性 | 低 | 多字体回退链（Arial→Helvetica→DejaVu Sans） |

---

## 十四、总结

BioSpark v0.4 完成了面向科研人员的重大功能升级。在 v0.3 三信号全覆盖的基础上，新增了**自动超参优化**和**出版质量图表**两大核心能力：

**v0.4 核心价值：**
1. **零调参训练** — Auto-Optimize 一键启用 LR finder + 架构自适应 + Early Stopping + 类别权重
2. **论文级图表** — 5 种图表 × 3 种期刊样式（Nature/IEEE/Science），300 DPI PNG + SVG
3. **模型架构图** — 自动从训练模型生成结构示意图，可直接插入论文
4. **一键打包** — ZIP 下载全部图表（10文件），附 README 说明

**平台核心竞争力：**
1. **零代码门槛** — 浏览器即用，无需 Python/MATLAB 知识
2. **三信号全覆盖** — ECG 心律失常 + EEG 睡眠分期 + EMG 手势识别
3. **端到端训练** — 上传数据到导出模型 + 论文图表，6 步完成
4. **科研级输出** — 出版质量图表直接满足 Nature/IEEE/Science 投稿要求
5. **真实数据验证** — 基于 MIT-BIH、NinaPro DB5 等权威数据集

**下一步重点：** EMG 精度优化 → EEG 模型完成 → 用户系统 → 实时流式推理 → 商业化。

---

*本报告由 BioSpark 团队编写，基于 v0.4.0 版本（2026-04-08）。*
