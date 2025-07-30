import torch
from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F

BATCH_SIZE = 8
DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ai4bharat/indicwav2vec-hindi"

transcripts = []

ds = iter(load_dataset("ai4bharat/IndicVoices", "hindi", split="valid", streaming=True))

iwav2vec = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE_ID)
processor_w2v = AutoProcessor.from_pretrained(MODEL_ID)

for i in range(BATCH_SIZE):
    sample = next(ds)
    resampled_audio = F.resample(torch.tensor(sample["audio_filepath"]["array"]), 48000, 16000).numpy()

    input_values = processor_w2v(resampled_audio, sampling_rate=16_000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = iwav2vec(input_values.to(DEVICE_ID)).logits.cpu()
    
    output_str = processor_w2v.batch_decode(logits.numpy()).text
    transcripts.append(output_str[0])

print(f"Greedy Decoding: {transcripts}")
