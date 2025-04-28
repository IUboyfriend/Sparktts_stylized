# 语音质量评估脚本（批量优化版）

## 目录
- [简介](#简介)
- [使用环境要求](#使用环境要求)
- [文件结构要求](#文件结构要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [主要优化点](#主要优化点)
- [注意事项](#注意事项)
- [常见问题](#常见问题)
- [更新日志](#更新日志)
- [联系方式](#联系方式)

## 简介
本脚本用于批量评估参考语音 (`ref_wavs/`) 与生成语音 (`gen_wavs/`) 的四项指标：  

- **STOI**：短时客观可懂度（因为我们输出的风格化音频可能会与参考音频的语速不太一致，因此stoi可能会显著的低（stoi假设两段语音时序严格对齐，任何整体“拉伸”或“压缩”都会被算法误判成信息丢失，分数自然被拉低），可以考虑去掉这个指标）
- **UTMOS**：基于深度学习的 MOS 音质预测，主管听感打分
- **WER**：单词错误率，评估语义可懂度  
- **SIM**：说话人相似度（余弦相似度）  
 
评估结果保存至 `eval_results/evaluation_results.csv`。

## 使用环境要求
- Python ≥ 3.8   
- 依赖安装：
  ```bash
  pip install torch torchaudio jiwer pandas pystoi transformers speechbrain
  pip install git+https://github.com/sarulab-speech/UTMOSv2.git

##文件结构要求
  ref_wavs/   # 参考音频
  gen_wavs/   # 生成音频
  两文件夹内 .wav 数量必须一致，一一对应

  建议相同排序或同名文件（如 0001.wav）

运行后将自动：

  加载模型（ECAPA、Whisper-Medium、UTMOS-v2）

  批量推理并计算 SIM / UTMOS / STOI / WER

  结果写入 eval_results/evaluation_results.json

配置说明

SAMPLE_RATE	16000	音频采样率
DEVICE	自动检测	GPU 优先
BATCH_SIZE	8	批大小，显存越大可调越高
RESULT_CSV	eval_results/evaluation_results.csv	结果输出路径
Whisper 模型	openai/whisper-medium	使用 medium 提升速度


##注意事项
  首次运行需下载模型（Whisper-medium 与 UTMOS-v2）

  模型下载后缓存本地，后续运行无需重复下载

  可根据显存调节 BATCH_SIZE

  若仅需部分指标，可按需裁剪脚本逻辑



##更新日志
  v1.0：单条推理版，Whisper-large-v3

  v2.0：改用 Whisper-medium，加入批量推理

  v2.1：新增缓存管理与 FutureWarning 静音处理






