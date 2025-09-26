import psycopg2
#from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
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
from oov_generator import return_database

BATCH_SIZE = 6
TOTAL_SIZE = 6


ds = iter(load_dataset("ai4bharat/Lahaja",split="test", streaming=True))

iconformer = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")
DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ai4bharat/indicwav2vec-hindi"
device = torch.device(DEVICE_ID)


def run_asr_batch(dataset):
    file_paths = []
    results = []

    for file in dataset:
        #sample = next(dataset)
        #print("Current sample:", sample['verbatim'])

        #IndicConformer

        audio_np = file.astype(np.float32)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, audio_np, samplerate=16000)
        file_paths.append(tmp_wav.name)

        print("File paths:", file_paths)


    #IndicConformer

    iconformer.freeze() # inference mode
    iconf = iconformer.to(device) # transfer model to device

    iconf.cur_decoder = "ctc"
    ctc_text = iconf.transcribe(file_paths, batch_size=BATCH_SIZE, logprobs=False, language_id='hi')[0]

    print("CTC Text:", ctc_text, "| Length:", len(ctc_text))

    for _ in range(BATCH_SIZE):
        h1 = ctc_text[_]

        results.append((h1))

    return results

def create_database_new(batch_data, csv_file="oov_conformer_transcripts.csv"):
    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    # Append new rows
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header if file didn't exist
        if not file_exists:
            writer.writerow(["id", "conf"])

        # Figure out the next ID
        if file_exists:
            with open(csv_file, mode="r", encoding="utf-8") as fr:
                reader = list(csv.reader(fr))
                next_id = len(reader)  # header counts as row 0
        else:
            next_id = 1

        # Write each tuple in batch_data
        for i in range(0, len(batch_data)):
            print("Row:", batch_data[i])
            print("Listed Row:", [batch_data[i]])
            writer.writerow([next_id] + [batch_data[i]])
            next_id += 1

    # Read the CSV back into batch_data-style list
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = [(row["id"], row["conf"]) for row in reader]

    return results

def main():
    #ds = iter(load_dataset("ai4bharat/IndicVoices",split="test", streaming=True))
    audio_list = return_database()

    all_asr = []
    for i in range(0, TOTAL_SIZE, BATCH_SIZE):
        results_arr = run_asr_batch(audio_list)
        all_asr.extend(results_arr)
        print(f"ASR batch {i//BATCH_SIZE + 1}/{TOTAL_SIZE//BATCH_SIZE}: done")
    
    print("Pre ASR:", all_asr)

    final_database = create_database_new(all_asr)

    print('ALL ASR:', final_database)

if __name__ == '__main__':
    main()
