import torch
from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F
from jiwer import wer, compute_measures

BATCH_SIZE = 1
DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ai4bharat/indicwav2vec-hindi"

# transcripts = []

ds = iter(load_dataset("ai4bharat/Lahaja", split="test", streaming=True))

iwav2vec = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE_ID)
processor_w2v = AutoProcessor.from_pretrained(MODEL_ID)

sample = next(ds)
resampled_audio = F.resample(torch.tensor(sample["audio_filepath"]["array"]), 48000, 16000).numpy()

model = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

input_values = processor(resampled_audio, return_tensors="pt").input_values

with torch.no_grad():
    logits = model(input_values.to(DEVICE_ID)).logits.cpu()
    
prediction_ids = torch.argmax(logits, dim=-1)
output_str = processor.batch_decode(prediction_ids)[0]
print(f"Greedy Decoding: {output_str}")

reference = sample["verbatim"]

print("WER:", wer(reference, output_str[0]))