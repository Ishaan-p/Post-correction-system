import psycopg2
# from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
# import jax.numpy as jnp
from datasets import load_dataset
# import nemo.collections.asr as nemo_asr
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
import tempfile
import soundfile as sf
from dataclasses import dataclass
import pandas as pd
import csv
import os

BATCH_SIZE = 8
TOTAL_SIZE = 6152


# ds = iter(load_dataset("ai4bharat/IndicVoices","hindi", split="train", streaming=True))


MODEL_ID = "ai4bharat/indicwav2vec-hindi"

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()

def run_asr_batch(dataset_iterator, batch_size=BATCH_SIZE):
    """
    Pull up to `batch_size` samples from dataset_iterator, resample to 16k,
    run the model in a single batched forward pass, and return transcripts.
    Assumes audio is at sample["audio_filepath"]["array"] and (optionally)
    sample["audio_filepath"]["sampling_rate"].
    """
    resampled_list = []
    verbatim_list = []
    transcripts = []

    for _ in range(BATCH_SIZE):
        sample = next(dataset_iterator)

        file = sample["audio_filepath"]["array"]
        audio_np = file.astype(np.float32)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, audio_np, samplerate=16000)
        SPEECH_FILE = tmp_wav.name
        wav, sr = librosa.load(SPEECH_FILE, sr=16000)   
        input_values = processor(wav, return_tensors="pt", sampling_rate=16000).input_values.to(device)

        # Run inference
        with torch.no_grad():
           logits = model(input_values).logits
           emissions = torch.log_softmax(logits, dim=-1).cpu()

        tokens = torch.argmax(emissions, dim=-1)[0].tolist()
        labels = processor.tokenizer.convert_ids_to_tokens(tokens)

        # Convert token strings to numeric vocabulary indices
        token_ids = processor.tokenizer.convert_tokens_to_ids(labels)

        def get_trellis(emission, token_ids, blank_id=0):
           num_frame = emission.size(0)
           num_tokens = len(token_ids)

           trellis = torch.zeros((num_frame, num_tokens))
           trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
           trellis[0, 1:] = -float("inf")
           trellis[-num_tokens + 1 :, 0] = float("inf")

           for t in range(num_frame - 1):
              trellis[t + 1, 1:] = torch.maximum(
                 trellis[t, 1:] + emission[t, blank_id],
                 trellis[t, :-1] + emission[t, token_ids[1:]],
           )
           return trellis

        trellis = get_trellis(emissions[0], token_ids)

        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float

        def backtrack(trellis, emission, token_ids, blank_id=0):
            t, j = trellis.size(0) - 1, trellis.size(1) - 1
            path = [Point(j, t, emission[t, blank_id].exp().item())]
    
            while j > 0 and t > 0:
               p_stay = emission[t - 1, blank_id]
               p_change = emission[t - 1, token_ids[j]]
        
               stayed = trellis[t - 1, j] + p_stay
               changed = trellis[t - 1, j - 1] + p_change

               t -= 1
               if changed > stayed:
                  j -= 1

               prob = (p_change if changed > stayed else p_stay).exp().item()
               path.append(Point(j, t, prob))

            while t > 0:
               prob = emission[t - 1, blank_id].exp().item()
               path.append(Point(j, t - 1, prob))
               t -= 1

            return path[::-1]

        path = backtrack(trellis, emissions[0], token_ids)

        @dataclass
        class Segment:
           label: str
           start: int
           end: int
           score: float

        def merge_repeats(path):
            i1, i2 = 0, 0
            segments = []

            while i1 < len(path):
               while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                  i2 += 1

               avg_score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
               segments.append(Segment(
                   processor.tokenizer.convert_ids_to_tokens(token_ids[path[i1].token_index]),
                   path[i1].time_index,
                   path[i2 - 1].time_index + 1,
                   avg_score
               ))
        
               i1 = i2
            return segments

        segments = merge_repeats(path)

        for seg in segments:
           print(seg)
        merged_transcript = "".join(seg.label for seg in segments if seg.label not in ["_", "<pad>"])

        print("Final Merged Transcript:\n", merged_transcript)
        transcripts.append(merged_transcript)

    return transcripts

def create_database_new(batch_data, csv_file="new_hindi_wav2vec_transcripts.csv"):
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
            writer.writerow([next_id] + [row])
            next_id += 1

    # Read the CSV back into batch_data-style list
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = [(row["id"], row["w2v"]) for row in reader]

    return results

def main():
    ds = iter(load_dataset("ai4bharat/Lahaja",split="test", streaming=True))
    #audio_list = return_database()
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
