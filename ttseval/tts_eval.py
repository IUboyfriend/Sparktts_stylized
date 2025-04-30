import os
import json
from typing import List

import torch
import torchaudio
import jiwer
from pystoi import stoi
from speechbrain.inference import EncoderClassifier
from transformers import pipeline
import utmosv2
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
SAMPLE_RATE = 16000
REF_DIR = "ref_wavs"
GEN_DIR = "gen_wavs"
RESULT_JSON = "eval_results/evaluation_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# Model loading
def load_models():
    print("Loading models …")

    spk_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa",
        run_opts={"device": DEVICE},
    )

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        device=0 if DEVICE == "cuda" else -1,
        batch_size=BATCH_SIZE,
        generate_kwargs={"task": "transcribe", "language": "en"},
    )

    utmos_model = utmosv2.create_model(pretrained=True, device=DEVICE)

    def utmos_fn(wav_path: str) -> float:
        return utmos_model.predict(input_path=wav_path)

    return spk_encoder, asr_pipe, utmos_fn

# Utility functions
def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav.squeeze(0)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()

def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# Main function
def main() -> None:
    os.makedirs("eval_results", exist_ok=True)

    spk_encoder, asr_pipe, utmos_fn = load_models()

    ref_files: List[str] = sorted(f for f in os.listdir(REF_DIR) if f.lower().endswith(".wav"))
    gen_files: List[str] = sorted(f for f in os.listdir(GEN_DIR) if f.lower().endswith(".wav"))

    if len(ref_files) != len(gen_files):
        raise ValueError("ref_wavs and gen_wavs must contain the same number of WAV files.")

    print(f"Processing {len(ref_files)} samples...")

    results = []
    
    with tqdm(total=len(ref_files), desc="Total Progress") as pbar:
        for ref_batch, gen_batch in zip(batchify(ref_files, BATCH_SIZE), batchify(gen_files, BATCH_SIZE)):
            ref_paths = [os.path.join(REF_DIR, f) for f in ref_batch]
            gen_paths = [os.path.join(GEN_DIR, f) for f in gen_batch]

            gen_transcripts = asr_pipe(gen_paths)
            ref_transcripts = asr_pipe(ref_paths)

            for idx in range(len(gen_batch)):
                ref_audio = load_audio(ref_paths[idx])
                gen_audio = load_audio(gen_paths[idx])

                min_len = min(ref_audio.shape[-1], gen_audio.shape[-1])
                ref_audio = ref_audio[:min_len]
                gen_audio = gen_audio[:min_len]

                ref_emb = spk_encoder.encode_batch(ref_audio.unsqueeze(0)).squeeze(0)
                gen_emb = spk_encoder.encode_batch(gen_audio.unsqueeze(0)).squeeze(0)
                sim = cosine_similarity(ref_emb, gen_emb)

                utmos_score = utmos_fn(gen_paths[idx])

                ref_text = ref_transcripts[idx]["text"].lower().strip()
                gen_text = gen_transcripts[idx]["text"].lower().strip()
                wer_score = jiwer.wer(ref_text, gen_text)

                stoi_score = stoi(ref_audio.numpy(), gen_audio.numpy(), SAMPLE_RATE, extended=False)

                results.append({
                    "reference": ref_batch[idx],
                    "generated": gen_batch[idx],
                    "SIM": sim,
                    "UTMOS": utmos_score,
                    "STOI": stoi_score,
                    "WER": wer_score,
                })
                
                pbar.update(1)

    with open(RESULT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    avg_sim = sum(r["SIM"] for r in results) / len(results)
    avg_utmos = sum(r["UTMOS"] for r in results) / len(results)
    avg_stoi = sum(r["STOI"] for r in results) / len(results)
    avg_wer = sum(r["WER"] for r in results) / len(results)

    print("\nAverage Evaluation Metrics:")
    print(f"SIM:   {avg_sim:.4f}")
    print(f"UTMOS: {avg_utmos:.4f}")
    print(f"STOI:  {avg_stoi:.4f}")
    print(f"WER:   {avg_wer:.4f}")

    print("\nDetailed evaluation results saved to:", RESULT_JSON)

if __name__ == "__main__":
    main()
