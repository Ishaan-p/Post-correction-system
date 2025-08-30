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


BATCH_SIZE = 5
TOTAL_SIZE = 100


ds = iter(load_dataset("ai4bharat/Lahaja",split="test", streaming=True))

iconformer = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")
whisper_pipeline = FlaxWhisperPipline('parthiv11/indic_whisper_hi_multi_gpu', dtype=jnp.bfloat16)
DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ai4bharat/indicwav2vec-hindi"
device = torch.device(DEVICE_ID)

iwav2vec = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE_ID)
processor_w2v = AutoProcessor.from_pretrained(MODEL_ID)

def run_asr_batch(dataset):
    whisper_transcripts = []
    file_paths = []
    wav2vec_transcripts = []
    results = []
    references = []

    for i in range(BATCH_SIZE):
        sample = next(dataset)
        print("Current sample:", sample['verbatim'])

        #IndicWhisper

        audio_file = sample["audio_filepath"]["array"]
        whisper_transcripts.append(whisper_pipeline(audio_file)['text'])

        #IndicConformer

        audio_np = sample["audio_filepath"]["array"].astype(np.float32)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, audio_np, samplerate=16000)
        file_paths.append(tmp_wav.name)

        #IndicWav2Vec

        resampled_audio = F.resample(torch.tensor(sample["audio_filepath"]["array"]), 48000, 16000).numpy()

        input_values = processor_w2v(resampled_audio, sampling_rate=16000, return_tensors="pt").input_values

        with torch.no_grad():
            logits = iwav2vec(input_values.to(DEVICE_ID)).logits.cpu()
        
        output_str = processor_w2v.batch_decode(logits.numpy()).text
        wav2vec_transcripts.append(output_str[0])


        references.append(sample["verbatim"])

        print("File paths:", file_paths)


    #IndicConformer

    iconformer.freeze() # inference mode
    iconf = iconformer.to(device) # transfer model to device

    iconf.cur_decoder = "ctc"
    ctc_text = iconf.transcribe(file_paths, batch_size=BATCH_SIZE, logprobs=False, language_id='hi')[0]

    print("CTC Text:", ctc_text, "| Length:", len(ctc_text))

    for _ in range(BATCH_SIZE):
        h1 = ctc_text[_]
        h2 = wav2vec_transcripts[_]
        h3 = whisper_transcripts[_]
        results.append((h1, h2, h3, references[_]))

    return results

def create_database(batch_data):
    # Connect to your postgres DB
    conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="9545", port=5432)

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Execute a query
    cur.execute('''CREATE TABLE IF NOT EXISTS transcripts (
        id SERIAL PRIMARY KEY,
        conf TEXT,
        w2v TEXT,
        wh TEXT,
        reference TEXT
    );
    ''')

    cur.executemany(
        "INSERT INTO transcripts (conf, w2v, wh, reference) VALUES (%s, %s, %s, %s)",
        batch_data
    )

    conn.commit()

    cur.execute("""SELECT conf, w2v, wh, reference FROM transcripts""")
    results = cur.fetchall()

    cur.close()
    conn.close()

    return results

def main():
    ds = iter(load_dataset("ai4bharat/Lahaja",split="test", streaming=True))

    all_asr = []
    for i in range(0, TOTAL_SIZE, BATCH_SIZE):
        results_arr = run_asr_batch(ds)
        all_asr.extend(results_arr)
        print(f"ASR batch {i//BATCH_SIZE + 1}/{TOTAL_SIZE//BATCH_SIZE}: done")
    
    print("Pre ASR:", all_asr)

    final_database = create_database(all_asr)

    print('ALL ASR:', final_database)

if __name__ == '__main__':
    main()