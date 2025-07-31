import torch
from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F

BATCH_SIZE = 1
DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ai4bharat/indicwav2vec-hindi"

# transcripts = []

ds = iter(load_dataset("ai4bharat/Lahaja", split="test", streaming=True))

iwav2vec = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE_ID)
processor_w2v = AutoProcessor.from_pretrained(MODEL_ID)

sample = next(ds)
resampled_audio = F.resample(torch.tensor(sample["audio_filepath"]["array"]), 48000, 16000).numpy()

input_values = processor_w2v(resampled_audio, sampling_rate=16_000, return_tensors="pt").input_values
print(input_values)
with torch.no_grad():
    logits = iwav2vec(input_values.to(DEVICE_ID)).logits.cpu()
    print(logits)

output_str = processor_w2v.batch_decode(logits.numpy()).text


print(f"Greedy Decoding: {output_str[0]}")
