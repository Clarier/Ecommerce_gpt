# 从零构建多模态大模型 (MLLM from Scratch)

> **面试项目说明**：本项目从零实现了一个完整的多模态大模型（MLLM），涵盖 Transformer → ViT → GPT → MLLM → RL 微调的全链路，全部基于纯 PyTorch 实现，不依赖 HuggingFace transformers / timm 等高封装库。适合在 A100 40G 单卡上完整训练。

---

## 📁 项目结构

```
MLLM_from_scratch/
├── configs/                          # 所有模块的 YAML 配置文件
│   ├── vit_config.yaml
│   ├── llm_config.yaml
│   ├── mllm_config.yaml
│   └── rl_mllm_config.yaml
│
├── transformer_from_scratch/         # 第一层：手写 Transformer 基础组件
│   ├── attention.py                  #   ScaledDotProductAttention + MultiHeadAttention
│   ├── layers.py                     #   FFN、PositionalEncoding、LayerNorm
│   ├── blocks.py                     #   EncoderBlock、DecoderBlock
│   └── model.py                      #   TransformerEncoder / Decoder / Full Transformer
│
├── vision_transformer/               # 第二层：Vision Transformer (ViT)
│   ├── vit.py                        #   PatchEmbedding + ViT（分类 / 特征提取双模式）
│   ├── train_vit.py                  #   CIFAR-10 训练脚本
│   └── predict_vit.py                #   单张图片推理脚本
│
├── language_model/                   # 第三层：GPT-style 语言模型
│   ├── tokenizer.py                  #   字符级 Tokenizer（含特殊 token 管理）
│   ├── llm.py                        #   Decoder-only GPT 模型
│   ├── train_llm.py                  #   Tiny Shakespeare 训练脚本
│   └── generate_text.py              #   自回归文本生成
│
├── multimodal_model/                 # 第四层：多模态融合 MLLM
│   ├── connector.py                  #   视觉 → 语言 空间映射（Linear / MLP）
│   ├── mllm.py                       #   MLLM 组装（ViT + Connector + GPT）
│   ├── train_mllm.py                 #   Flickr8k 图文对训练（SFT）
│   ├── inference_mllm.py             #   "看图说话" 推理
│   ├── train_rl_mllm.py              #   SCST 强化学习微调
│   └── inference_rl_mllm.py          #   RL 模型推理
│
├── datasets/                         # 数据集封装
│   ├── data_utils.py                 #   通用下载工具
│   ├── cifar10.py                    #   CIFAR-10 数据集
│   ├── tinyshakespeare.py            #   Tiny Shakespeare 数据集
│   └── Flickr8k.py                   #   Flickr8k 图文对数据集
│
├── utils/                            # 工具函数
│   ├── training_utils.py             #   set_seed / get_device / save_checkpoint
│   └── config_parser.py              #   YAML 配置解析
│
├── tests/                            # 单元测试
│   ├── test_attention.py
│   ├── test_blocks.py
│   └── test_transformer.py
│
├── script/                           # 一键训练/测试脚本
│   ├── train_vit.sh
│   ├── test_vit.sh
│   ├── train_llm.sh
│   ├── test_llm.sh
│   ├── train_mllm.sh
│   ├── test_mllm.sh
│   ├── train_rl_mllm.sh
│   └── test_rl_mllm.sh
│
├── main.py                           # 统一入口（任务分发器）
├── requirements.txt
└── README.md
```

---

## 🧩 各模块详细说明

### 1. `transformer_from_scratch/` — Transformer 基础组件

这是整个项目的**地基**，所有后续模型都复用这些组件。

| 文件 | 核心内容 | 面试考点 |
|------|----------|----------|
| `attention.py` | **Scaled Dot-Product Attention**：实现 `softmax(QK^T / √d_k) V` 的完整计算，支持 mask 机制；**Multi-Head Attention**：将 d_model 拆分为 n_heads 个子空间，独立做注意力后合并，包含残差连接 + LayerNorm | 为什么要除以 √d_k？多头注意力的并行化如何实现？ |
| `layers.py` | **FFN**：两层线性变换 + ReLU + 残差 + LayerNorm；**Positional Encoding**：正弦/余弦位置编码；**LayerNorm**：手写层归一化（含可学习参数 γ/β） | Pre-Norm vs Post-Norm 区别？位置编码为什么用 sin/cos？ |
| `blocks.py` | **EncoderBlock**：Self-Attention → FFN；**DecoderBlock**：Masked Self-Attention → Cross-Attention → FFN | Encoder 和 Decoder 的 mask 有什么不同？ |
| `model.py` | **TransformerEncoder/Decoder**：N 层 Block 堆叠；**Transformer**：完整 Seq2Seq 模型 + padding mask + causal mask 生成 | 如何生成因果掩码？Xavier 初始化为什么重要？ |

### 2. `vision_transformer/` — Vision Transformer

将图像视为 patch 序列，复用 TransformerEncoder 做图像分类或特征提取。

| 文件 | 核心内容 |
|------|----------|
| `vit.py` | **PatchEmbedding**：用 Conv2d(kernel=patch_size, stride=patch_size) 高效切 patch；**ViT 双模式**：`num_classes!=None` 为分类模式（取 [CLS] token → MLP Head），`num_classes=None` 为特征提取模式（输出全序列供 MLLM 使用）；`forward_features()` 解耦了特征提取逻辑 |
| `train_vit.py` | CIFAR-10 上训练，含数据增强（RandomResizedCrop + HorizontalFlip）、CosineAnnealing 学习率调度、loss/accuracy 曲线可视化 |
| `predict_vit.py` | 加载 checkpoint → 预处理 → softmax → argmax → 类别映射 |

**关键设计**：ViT 通过 `forward_features()` 方法暴露中间特征，这是与 MLLM 对接的接口。

### 3. `language_model/` — GPT-style 语言模型

Decoder-only 架构，字符级自回归语言模型。

| 文件 | 核心内容 |
|------|----------|
| `tokenizer.py` | 字符级 Tokenizer，管理 `<pad>/<sos>/<eos>/<unk>` 特殊 token，支持 save/load vocab |
| `llm.py` | **GPTModel**：Token Embedding + Learnable Position Embedding + N 层 DecoderBlock + LM Head。关键方法：`forward(idx)` 从 token ID 开始；`forward_from_embeddings(emb)` 从 embedding 开始（**MLLM 融合的关键接口**） |
| `train_llm.py` | Tiny Shakespeare 训练，含验证集评估 + 训练中文本采样 + loss 曲线 |
| `generate_text.py` | 自回归生成：logits → softmax → multinomial sampling → 拼接 → 循环 |

**关键设计**：`forward_from_embeddings()` 允许 MLLM 将视觉 embedding 与文本 embedding 拼接后直接传入 GPT，无需经过 token embedding 层。

### 4. `multimodal_model/` — 多模态融合 MLLM

**这是项目的核心**，将 ViT + Connector + GPT 组装为完整的多模态模型。

| 文件 | 核心内容 |
|------|----------|
| `connector.py` | **Connector**：将 ViT 输出 (D_vision) 映射到 GPT 的嵌入空间 (D_language)。支持 Linear 和 MLP（Linear → GELU → Linear）两种模式 |
| `mllm.py` | **MLLM 核心**：(1) `forward()` 训练：ViT 提取视觉特征 → Connector 投影 → 与文本 embedding 拼接 → GPT 自回归生成 logits；(2) `generate()` 推理：自回归 loop，每步拼接新 token embedding；(3) `sample_with_logprobs()` / `greedy_with_logprobs()`：为 RL 训练提供采样 + log prob |
| `train_mllm.py` | **训练关键**：构造 labels 时只在文本 token 位置放真实 target，视觉 token 位置填充 ignore_index，确保 loss 只计算文本预测 |
| `inference_mllm.py` | 加载 checkpoint → 图像预处理 → `mllm.generate()` → 输出描述 |
| `train_rl_mllm.py` | **SCST 强化学习微调**：采样输出算 reward、贪心输出算 baseline、`advantage = R_sample - R_greedy`、REINFORCE 梯度 + 可选 CE 混合损失 |
| `inference_rl_mllm.py` | RL 模型推理入口 |

**多模态融合流程**（面试重点）：
```
Image → ViT.forward_features() → [B, N_vis+1, D_vis]
                                      ↓
                              Connector (MLP)
                                      ↓
                              [B, N_vis+1, D_lang]
                                      ↓
              torch.cat([visual_emb, text_emb], dim=1)  ← 序列维度拼接
                                      ↓
                    GPT.forward_from_embeddings()
                                      ↓
                          logits [B, T_total, vocab]
```

### 5. `datasets/` — 数据集

| 文件 | 数据集 | 用途 |
|------|--------|------|
| `cifar10.py` | CIFAR-10 | ViT 图像分类训练 |
| `tinyshakespeare.py` | Tiny Shakespeare (~1MB 文本) | GPT 语言模型训练 |
| `Flickr8k.py` | Flickr8k (8000 张图 + 5 条描述/图) | MLLM 图文对训练 |

### 6. `utils/` — 工具

- `training_utils.py`：随机种子设置、设备选择、checkpoint 保存
- `config_parser.py`：YAML 配置文件解析

---

## 🚀 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 数据准备

- **CIFAR-10** 和 **Tiny Shakespeare**：训练脚本会自动下载
- **Flickr8k**：需手动从 [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) 下载，解压到 `./data/flickr8k/`，目录下应包含 `Images/` 文件夹和 `captions.txt`

### 分步训练

```bash
# Step 1: 训练 ViT（CIFAR-10 图像分类）
bash script/train_vit.sh

# Step 2: 训练 GPT LM（Tiny Shakespeare 文本生成）
bash script/train_llm.sh

# Step 3: 训练 MLLM（Flickr8k 图文对，SFT 阶段）
bash script/train_mllm.sh

# Step 4: RL 微调（SCST，基于 SFT checkpoint）
bash script/train_rl_mllm.sh
```

### 推理测试

```bash
bash script/test_vit.sh       # ViT 单张图片分类
bash script/test_llm.sh       # GPT 文本生成
bash script/test_mllm.sh      # MLLM 看图说话
bash script/test_rl_mllm.sh   # RL 模型推理
```

---

## 💡 A100 40G 单卡资源估计

| 阶段 | 模型参数量 | batch_size | 显存占用 | 训练时间 |
|------|-----------|------------|----------|----------|
| ViT (CIFAR-10) | ~30M | 256 | ~8 GB | ~2h (100 epochs) |
| GPT (Shakespeare) | ~20M | 64 | ~4 GB | ~1h (30 epochs) |
| MLLM SFT (Flickr8k) | ~80M | 8 | ~18 GB | ~3h (15 epochs) |
| MLLM RL (SCST) | ~80M | 2 | ~12 GB | ~1h (1 epoch) |

全部训练合计约 **7 小时**，远在 A100 40G 单卡能力范围内。

---

## 🎯 面试要点总结

1. **Transformer 核心机制**：Scaled Dot-Product Attention、Multi-Head Attention 的数学推导和实现、Causal Mask 与 Padding Mask 的区别
2. **ViT 设计**：图像如何转化为序列？[CLS] token 的作用？位置编码为什么用可学习的？
3. **GPT 架构**：Decoder-only 与 Encoder-Decoder 的取舍、forward vs forward_from_embeddings 的设计意图
4. **多模态融合**：视觉 token 与文本 token 如何拼接？Loss 计算为什么只算文本部分？Connector 的设计选择
5. **RL 微调 (SCST)**：Self-Critical Sequence Training 的原理、为什么用贪心输出做 baseline、REINFORCE 的方差问题

---

## 📚 参考资料

- Sebastian Raschka,《LLMs from Scratch》 [GitHub](https://github.com/rasbt/LLMs-from-scratch)
- Vaswani et al., "Attention Is All You Need" (2017)
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020)
- Rennie et al., "Self-Critical Sequence Training for Image Captioning" (SCST, 2017)
