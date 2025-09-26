- The main file is batch.py, which gets all three ASR transcripts and runs the GPT-OSS model.
  The final result is the generated CSV for the gpt output batch, along with some metrics (these metrics don't matter for now)
- oss_run is where I have played with Gpt-oss 20b, and how I have loaded the model
- The files: conformer_generator.py, whisper_generator.py, wav2vec generator.py, are used to generate the CSV files for all ASR outputs on the Hindi Lahaja dataset (this has already been done)
- The ASR transcript files for Hindi are conformer_transcripts.csv, new_wav2vec_transcripts.csv, whisper_transcripts.csv
- oov-generator.py is where I have created the OOV incontext dataset, which is then loaded into the ASR generator files to create the OOV batch for each model

NOTE: To run GPT OSS, the package versions need to be the following:
huggingface_hub=0.34.5
transformers=4.56.1

For running the other ASR models, the package versions needed are on requirements.txt
