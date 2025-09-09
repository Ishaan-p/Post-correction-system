import psycopg2
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp
from datasets import load_dataset
import torch
import nemo.collections.asr as nemo_asr
import torchaudio.functional as F
import numpy as np
import soundfile as sf
import tempfile
from transformers import AutoModelForCTC, AutoProcessor
import pandas as pd
import csv
import os

BATCH_SIZE = 8
TOTAL_SIZE = 6152


ds = iter(load_dataset("ai4bharat/Lahaja",split="test", streaming=True))


DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ai4bharat/indicwav2vec-hindi"
device = torch.device(DEVICE_ID)

iwav2vec = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE_ID)
processor_w2v = AutoProcessor.from_pretrained(MODEL_ID)

def run_asr_batch(dataset):
    wav2vec_transcripts = []
    results = []

    for i in range(BATCH_SIZE):
        sample = next(dataset)
        print("Current sample:", sample['verbatim'])

        #IndicWav2Vec

        resampled_audio = F.resample(torch.tensor(sample["audio_filepath"]["array"]), 48000, 16000).numpy()

        input_values = processor_w2v(resampled_audio, sampling_rate=16000, return_tensors="pt").input_values

        with torch.no_grad():
            logits = iwav2vec(input_values.to(DEVICE_ID)).logits.cpu()
        
        output_str = processor_w2v.batch_decode(logits.numpy()).text
        wav2vec_transcripts.append(output_str[0])


    for _ in range(BATCH_SIZE):
        h1 = wav2vec_transcripts[_]
        results.append((h1))

    return results

def create_database_new(batch_data, csv_file="wav2vec_transcripts.csv"):
    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    # Append new rows
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header if file didn't exist
        if not file_exists:
            writer.writerow(["id", "w2v"])

        # Figure out the next ID
        if file_exists:
            with open(csv_file, mode="r", encoding="utf-8") as fr:
                reader = list(csv.reader(fr))
                next_id = len(reader)  # header counts as row 0
        else:
            next_id = 1

        # Write each tuple in batch_data
        for row in batch_data:
            writer.writerow([next_id] + list(row))
            next_id += 1

    # Read the CSV back into batch_data-style list
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = [(row["id"], row["w2v"]) for row in reader]

    return results

def main():
    ds = iter(load_dataset("ai4bharat/Lahaja",split="test", streaming=True))

    all_asr = []
    for i in range(0, TOTAL_SIZE, BATCH_SIZE):
        results_arr = run_asr_batch(ds)
        all_asr.extend(results_arr)
        print(f"ASR batch {i//BATCH_SIZE + 1}/{TOTAL_SIZE//BATCH_SIZE}: done")
    
    print("Pre ASR:", all_asr)

    final_database = create_database_new(all_asr)

    print('ALL ASR:', final_database)

if __name__ == '__main__':
    main()
