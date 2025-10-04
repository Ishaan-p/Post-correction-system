import os, certifi

import pandas as pd
import asyncio
import random
from openai import OpenAI, AsyncOpenAI, RateLimitError, InternalServerError, APIError
from jiwer import wer, compute_measures
from transformers import pipeline
import torch
# from sarvamai import SarvamAI
# from langchain.chat_models.openai import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.memory import ConversationSummaryBufferMemory
# from langchain.utilities import GoogleSearchAPIWrapper
# from langchain.retrievers.web_research import WebResearchRetriever
# from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
import csv
import os
from datetime import datetime


N_CONCURRENCY = 1  # Not used anymore since we're doing true batching
BATCH_SIZE = 16  # Increased batch size for GPU processing
TOTAL_SIZE = 16
API_KEY = ""

# global iconformer, whisper_pipeline, device, iwav2vec, processor_w2v

# ds = iter(load_dataset("ai4bharat/IndicVoices", "hindi", split="valid", streaming=True))

start_time = datetime.now()

client = AsyncOpenAI()
# client = SarvamAI(
#     api_subscription_key="",
# )

# model_id = "openai/gpt-oss-20b"

# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype="auto",
#     device_map="auto",
# )

# client = AsyncOpenAI(
   #  base_url="https://router.huggingface.co/v1",
   #  api_key=os.getenv("HF_TOKEN"),
#)

reference = pd.read_csv("reference.csv")
conformer = pd.read_csv("multi_hindi_conformer_transcripts.csv")
whisper = pd.read_csv("hindi_whisper_transcripts.csv")
wav2vec = pd.read_csv("hindi_wav2vec_transcripts.csv")



def collect_database():
    results = []
    for _ in range(TOTAL_SIZE):
        conformer_row = conformer[conformer["id"] == _+1]
        whisper_row = whisper[whisper["id"] == _+1]
        wav2vec_row = wav2vec[wav2vec["id"] == _+1]
        reference_row = reference[reference["id"] == _+1]

        results.append((conformer_row["conf"].values[0], wav2vec_row["w2v"].values[0], whisper_row["wh"].values[0], reference_row["ref"].values[0]))

    return results


def create_prompt(h1, h2, h3):
    """Helper function to create prompt for a single sample"""
    language = "Hindi"
    prompt = f'''Language: {language}
    You are given outputs from three different Automatic Speech Recognition (ASR) systems. Each may contain errors.
    Your task is to generate the corrected transcription of the utterance by selecting and fixing words only from the given hypotheses. 
    Do not introduce new words or rephrase.
    
    ASR hypotheses:
        <hypothesis1>{h1}</hypothesis1>
        <hypothesis2>{h2}</hypothesis2>
        <hypothesis3>{h3}</hypothesis3>
    Return only the corrected transcription, with no explanations or additional text.
'''
    return prompt


async def correct_one(h1, h2, h3, max_retries=5):
    """Keep this for compatibility but it will be called less frequently"""
    prompt = create_prompt(h1, h2, h3)
    
    retries = 0
    while True:
        try:
            resp = await client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role":"user","content":prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        
        except (RateLimitError, InternalServerError, APIError) as e:
            retries += 1
            if retries > max_retries:
                raise  # give up after too many retries
            
            # exponential backoff with jitter
            wait_time = min(2 ** retries + random.random(), 60)
            print(f"Rate limit hit. Waiting {wait_time:.1f}s before retrying...")
            await asyncio.sleep(wait_time)


async def correct_batch_sync(batch_data):
    """Synchronous batch processing - more efficient for GPU"""
    # Create all prompts for the batch
    prompts = [create_prompt(h1, h2, h3) for h1, h2, h3, _ in batch_data]
    
    # Create messages for all prompts
    all_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    
    #print(f"Processing batch of {len(all_messages)} samples...")
    
    # Process entire batch at once
    # outputs = pipe(
    #     all_messages,
    #     max_new_tokens=4096,
    #     batch_size=len(all_messages),  # Process all at once
    #     padding=True,
    #     truncation=True
    # )

    sem = asyncio.Semaphore(len(all_messages))
    async def guarded(r):
        print("Launching correction for:", r[:3])
        async with sem:
            return await correct_one(*r[:3])
        
    tasks = [asyncio.create_task(guarded(r)) for r in batch_data]
    corrections = await asyncio.gather(*tasks)
    
    # Extract results
    #results = []
    # for output in outputs:
    #     raw = output[0]["generated_text"][-1]["content"]
        
    #     if "assistantfinal" in raw:
    #         final_answer = raw.split("assistantfinal", 1)[1].strip()
    #     else:
    #         final_answer = raw.strip()
        
    #     results.append(final_answer)
    
    return corrections


async def correct_batch(asr_results):
    """Modified to use synchronous batch processing in chunks"""
    all_corrections = []
    
    # Process in larger batches for better GPU utilization
    # Adjust this based on your GPU memory (L40S has 48GB, should handle 16-32 easily)
    GPU_BATCH_SIZE = 16  # Can increase this if GPU memory allows
    
    for i in range(0, len(asr_results), GPU_BATCH_SIZE):
        batch = asr_results[i:i+GPU_BATCH_SIZE]
        print(f"Processing batch {i//GPU_BATCH_SIZE + 1}/{(len(asr_results) + GPU_BATCH_SIZE - 1)//GPU_BATCH_SIZE}")
        
        # Run synchronous batch processing
        batch_corrections = await correct_batch_sync(batch)
        all_corrections.extend(batch_corrections)
        
        print(f"Completed {len(all_corrections)}/{len(asr_results)} samples")
    
    return all_corrections


def create_database_new(batch_data, csv_file="dump.csv"):
    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    # Append new rows
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header if file didn't exist
        if not file_exists:
            writer.writerow(["id", "gpt"])

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
        results = [(row["id"], row["gpt"]) for row in reader]

    return results

def main():
    # load into memory or stream; here we list
    # samples = [s for s in ds]
    # ds = iter(load_dataset("ai4bharat/Lahaja",split="test", streaming=True))
    


    all_asr = collect_database()
    # for i in range(0, TOTAL_SIZE, BATCH_SIZE):
    #     results_arr = run_asr_batch(ds)
    #     all_asr.extend(results_arr)
    #     print(f"ASR batch {i//BATCH_SIZE + 1}/{TOTAL_SIZE//BATCH_SIZE}: done")

    print(f"Starting correction of {len(all_asr)} samples...")
    
    # Async correction with batching
    corrections = asyncio.run(correct_batch(all_asr))

    create_database_new(corrections)

    def safe_str(x):
        return str(x) if isinstance(x, str) else "" if x is None or (isinstance(x, float) and x != x) else str(x)


    # Compute WERs
    metrics = {'conf':0, 'w2v':0, 'wh':0, 'gpt':0}
    for (h1, h2, h3, ref), corr in zip(all_asr, corrections):
        h1, h2, h3, ref, corr = map(safe_str, (h1, h2, h3, ref, corr))
        metrics['conf'] += wer(ref, h1)
        metrics['w2v'] += wer(ref, h2)
        metrics['wh']  += wer(ref, h3)
        metrics['gpt'] += wer(ref, corr)
        print("h1:", h1, "h2:", h2, "h3:", h3, "corr:", corr, "ref:", ref)

    print("Average WERs:", {k: metrics[k]/TOTAL_SIZE for k in metrics})

if __name__ == '__main__':
    main()
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"Took {elapsed.total_seconds() / 60:.2f} minutes")