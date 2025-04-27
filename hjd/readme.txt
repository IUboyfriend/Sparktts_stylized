1. To get activations 
We need to get the prior input to o_proj, which is the headwise attention. We need to use baukit.
conda install torchvision==0.20.1
注意使用--no-deps，否则sparktts环境会乱
pip install --no-deps git+https://github.com/davidbau/baukit


Run the below to get activations:
python get_tts_activations.py SparkTTS --stylized_dir "../dataset/stylized" --neutral_dir "../dataset/original" --transcript_file "../dataset/transcription_results.json" --model_dir "../pretrained_models/Spark-TTS-0.5B"

2.