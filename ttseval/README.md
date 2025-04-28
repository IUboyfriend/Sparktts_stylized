# 语音质量评估脚本（批量优化版）

> **支持快速、批量对比**参考语音 (`ref_wavs/`) 与生成语音 (`gen_wavs/`)，输出常见的语音质量指标结果。

---

## 目录
- [简介](#简介)
- [环境要求](#环境要求)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [注意事项](#注意事项)

---

## 简介
本脚本一次性计算 4 项常用指标，并将结果导出为 **JSON**，方便后续分析。

| 指标 | 说明 | 适用范围 |
|------|------|----------|
| **SIM** | 说话人相似度（ECAPA 嵌入余弦相似度） | 衡量音色/身份一致性 |
| **UTMOS** | 主观 MOS 预测模型 | 音质 / 自然度 |
| **WER** | Word Error Rate（Whisper 转写） | 语义可懂度 |
| **STOI\*** | 短时客观可懂度 | *需时序完全对齐；风格化语速不一致时不推荐* |

> \* 若风格化后语速与参考差异较大，可在脚本中注释掉 STOI 相关逻辑。

---

## 环境要求
- **Python ≥ 3.8**
- 依赖安装（GPU 环境建议提前安装匹配的 `torch` / `torchaudio`）：

```bash
pip install torch torchaudio jiwer pandas pystoi transformers speechbrain
pip install git+https://github.com/sarulab-speech/UTMOSv2.git
```

---

## 文件结构
```
├─ ref_wavs/   # 参考音频 (.wav)
├─ gen_wavs/   # 生成音频 (.wav)
└─ eval_results/
   └─ evaluation_results.json  # 运行后自动生成
```
> 两文件夹内 **WAV 数量必须一致**，且文件顺序 / 命名应能一一对应（如 `0001.wav`）。

---

## 快速开始
```bash
# 1. 准备同名或有序的参考 / 生成 wav 文件
# 2. 运行脚本
python audio_eval.py  # 或你保存的文件名

# 运行完成后
cat eval_results/evaluation_results.json      # 查看每条明细
# 终端会自动打印四项指标的平均值
```

---


---

## 注意事项
1. **首次运行需下载模型**（Whisper‑Medium、ECAPA、UTMOS-v2），可能占用数 GB。
2. 若 GPU 显存不足，可降低 `BATCH_SIZE` 或改用 `whisper-small`。
3. 仅需部分指标时，可直接在脚本中删除对应计算段，其他逻辑无依赖。
4. STOI 对语速/节奏敏感，如非完全对齐场景可跳过。

---



