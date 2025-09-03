from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp

pipeline = FlaxWhisperPipline('parthiv11/indic_whisper_nodcil', dtype=jnp.bfloat16)
transcript= pipeline('./sample.wav')
