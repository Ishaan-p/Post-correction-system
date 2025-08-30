from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp
from datasets import load_dataset
import numpy as np
import soundfile as sf
import tempfile
from jiwer import wer

BATCH_SIZE = 1

transcripts = []
direct_transcripts = []
verbatims = []

total_wer_filepath = 0
total_wer_direct = 0

pipeline = FlaxWhisperPipline('parthiv11/indic_whisper_hi_multi_gpu', dtype=jnp.bfloat16, batch_size=BATCH_SIZE)
ds = iter(load_dataset("ai4bharat/IndicVoices", "hindi", split="valid", streaming=True))

for i in range(BATCH_SIZE):
    sample = next(ds)
    audio_np = sample["audio_filepath"]["array"].astype(np.float32)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_wav.name, audio_np, samplerate=16000)
    
    transcripts.append(pipeline(tmp_wav.name)['text'])
    direct_transcripts.append(pipeline(sample["audio_filepath"]["array"])['text'])
    verbatims.append(sample['verbatim'])

#Compare WER
for _ in range(BATCH_SIZE):
    ref = verbatims[_]
    total_wer_filepath = total_wer_filepath + wer(ref, transcripts[_])
    total_wer_direct = total_wer_direct + wer(ref, direct_transcripts[_])

print(f"Transcript:", transcripts)

print("Using file paths:", total_wer_filepath/BATCH_SIZE)
print("Using audio files directly", total_wer_direct/BATCH_SIZE)
