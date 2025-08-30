import os, certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

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
import asyncio
from openai import OpenAI, AsyncOpenAI
from jiwer import wer, compute_measures
from sarvamai import SarvamAI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

N_CONCURRENCY = 1
BATCH_SIZE = 5
TOTAL_SIZE = 20
API_KEY = "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"

# global iconformer, whisper_pipeline, device, iwav2vec, processor_w2v

# ds = iter(load_dataset("ai4bharat/IndicVoices", "hindi", split="valid", streaming=True))

iconformer = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")
whisper_pipeline = FlaxWhisperPipline('parthiv11/indic_whisper_hi_multi_gpu', dtype=jnp.bfloat16)
DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ai4bharat/indicwav2vec-hindi"
device = torch.device(DEVICE_ID)

iwav2vec = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE_ID)
processor_w2v = AutoProcessor.from_pretrained(MODEL_ID)
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()
# client = SarvamAI(
#     api_subscription_key="sk_75a9n21l_DCAYvYeiDVtN7eLSCsb6srKw",
# )

def create_vector_db(dataset):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "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"

    texts = []
    for _ in range(1000):
        sample = next(dataset)
        texts.append(sample['verbatim'])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = InMemoryVectorStore(embeddings)

    docs = [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(texts)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    return vector_store
    

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

        input_values = processor_w2v(resampled_audio, sampling_rate=16_000, return_tensors="pt").input_values

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

async def correct_one(h1, h2, h3, vector):
    class State(TypedDict):
        h1: str
        h2: str
        h3: str
        context: List[Document]
        answer: str


    def retrieve(state: State):
        retrieved_docs = vector.similarity_search(state["h1"])
        return {"context": retrieved_docs}
    
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = f'''Perform error correction based on the top3 outputs generated by an Automatic Speech Recognition (ASR) system. The ASR hypotheses are listed in order of their ASR posterior score. Please provide the corrected top1 ASR transcription of the given utterance only.  Do not add any explanations or other words. 

            Here are a few in-context examples: 

            Example 1:
            <hypothesis1>जी नमस्ते जी बोलिए</hypothesis1> 
            <hypothesis2>क प</hypothesis2> 
            <hypothesis3>जी नमस्ते जी बोल्गे</hypothesis3> 

            Your output: जी नमस्ते जी बोलिए

            Example 2:
            <hypothesis1>जी</hypothesis1> 
            <hypothesis2></hypothesis2> 
            <hypothesis3>धन्यवाद</hypothesis3>

            Your output: जी

            Example 3:
            <hypothesis1>अगर आपका बच्चा चला जाएगा तो फिर उनका लॉस हो जाएगा ना ब अगर हम आपको छुट्टी दे भी रहे हैं तो मैम आपका चार्ज ज़्यादा लगेगा</hypothesis1> 
            <hypothesis2></hypothesis2> 
            <hypothesis3>अगर आपका बच्चा चला जायेगा तो फिर उनका लॉस हो जायेगा न बस अगर हम आपको छुट्टी दे भी रहे तो माम आपका चार्ज ज्यादा लगेगा</hypothesis3> 

            Your output: अगर आपका बच्चा चला जाएगा तो फिर उनका लॉस हो जाएगा ना बट अगर हम आपको छुट्टी दे भी रहे हैं तो मैम आपका साल ज्यादा लगेगा

            Example 4:
            <hypothesis1>जी बिल्कुल</hypothesis1> 
            <hypothesis2></hypothesis2> 
            <hypothesis3>तुझी भूल कर</hypothesis3>

            Your output: जी बिल्कुल

            Example 5:
            <hypothesis1>जी जी वैसे दो ढाई जन लग रहा है</hypothesis1> 
            <hypothesis2>ा का क</hypothesis2> 
            <hypothesis3>जीजी वैसे दो ढाई जल लग रहा </hypothesis3>
    
            Your output: जी जी वैसे दो ढाई हजार लग रहा है

            Example 6:
            <hypothesis1>ढाई हजार लग रहा है तो ठीक है हम चार्ज तो नहीं कम कर सकते हैं आप बीच में भी बीच में जा ही रही हैं और बच्चे का लॉस भी होगा</hypothesis1> 
            <hypothesis2>क क क र</hypothesis2> 
            <hypothesis3>ढाई अज़र लग रहा है तू थी एक है चार्ज तो नहीं कम कर सकते है आपको बीच में भी बीच में जाए ही नहीं है और बच्चे का लॉस भी होगा</hypothesis3>

            Your output: ढाई हजार लग रहा है दो तो ठीक है हम चार्ज तो नहीं कम कर सकते हैं आप बीच में भी बीच में जा ही रही हैं और बच्चे का लॉस भी होगा 

            Example 7:
            <hypothesis1>और आप तो ये भी बोल रहे हो कि मतलब बच्चा है तो अकेले नहीं छोड़ेंगे आपकी भी बात सही है किसी बच्चे को तो भई माँ के बिना तो आप जा रही हो तो आपकी तो चिंता बनी ही रहेगी सही बात है</hypothesis1> 
            <hypothesis2>क क क क क</hypothesis2> 
            <hypothesis3>और आप तो यदि बोल रहे हो की मतलब बच्चा है तो अकेले नहीं छोड़ेंगे आपकी भी बात सही है किसी बच्चे को तो माँ के बिना तो आप जा रहे हैं तो आपकी तो चिंता बनी ही रहेगी सही बात है</hypothesis3>

            Your output: और आप तो ये भी बोल रहे हो कि मतलब बच्चा है तो अकेले नहीं छोडेंगे आपकी भी बात सही हैं किसी बच्चे को तो भाई माँ के बिना तो आप जा रही हैं सो आपकी तो चिंता बनी ही रहेगी सही बात है

            Example 8:
            <hypothesis1>बट आप क रही हैं तो ठीक है हम आपको छुट्टी दे दे रहे हैं बट जो भी इनकी तैयारी है आप वहाँ पे कराते रहिएगा क्योंकि भई बच्चे</hypothesis1> 
            <hypothesis2>प क पर क ा</hypothesis2> 
            <hypothesis3>बस आप कह रही है तो ठीक है हम आपको छुट्टी दे दे रहा है जो भी इनकी तैयारी है आप वहाँ ऐसी करते रहियेगा क्यूँकी भाई बच्चे</hypothesis3>

            Your output: बट आप क केह रही हैं तो ठीक है हम आपको छुट्टी दे दे रहे हैं बट जो भी इनकी तैयारी है आप वहाँ पे कराते रहिएगा क्योंकि भाई बच्चे

            Example 9:
            <hypothesis1>जी बिल्कुल बिल्कुल ठीक</hypothesis1> 
            <hypothesis2>ा र के प</hypothesis2> 
            <hypothesis3>जी बिल्कुल बिल्कुल ठीक</hypothesis3>

            Your output: जी बिलकुल बिलकुल ठीक

            Example 10:
            <hypothesis1>जी तो कितने दिन की चाहिए डेस बता दीजिए</hypothesis1> 
            <hypothesis2>ा क क क</hypothesis2> 
            <hypothesis3>जी तो कितने दिन के चाहिए ढेर बता दीजिए</hypothesis3>

            Your output: जी तो कितने दिन की चाहिए डे बता दीजिए

            Example 11:
            <hypothesis1>ओके ऐसे ही आपको जाना कौन सी जगह पर है</hypothesis1> 
            <hypothesis2>क क का क</hypothesis2> 
            <hypothesis3>एक्सपेनेम ऐसे आपको जाना कौन सी जगह पर है</hypothesis3> 

            Your output: ओके ऐसे आपको जाना कौनसी जगह पर है

            Example 12:
            <hypothesis1>तो ताजमहल घुमाने जा रही हैं बच्चे को</hypothesis1> 
            <hypothesis2>ा</hypothesis2> 
            <hypothesis3>तो साजना घुमाने जा रही है बच्चे को</hypothesis3> 

            Your output: तो ताजमहल घुमाने जा रही हैं बच्चे को

            Example 13:
            <hypothesis1>ओके मैम ठींक है</hypothesis1> 
            <hypothesis2>ा</hypothesis2> 
            <hypothesis3>ओके माम थाने क्या</hypothesis3>

            Your output: ओके मैम ठीक है
            

            Please feel free to refer to this example. Do not output any placeholder tokens or angle‑bracketed markers (e.g. <output>).
            Provide only the corrected transcription, with no extra words, symbols, or tags.
            
            Here are some additional context/phrases that are semantically similar to the corrected transcript that you could refer to: {docs_content}

            Please start: 
            <hypothesis1>{state["h1"]}</hypothesis1> 
            <hypothesis2>{state["h2"]}</hypothesis2> 
            <hypothesis3>{state["h3"]}</hypothesis3>
        '''
        print("Context", docs_content)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            # top_p=1,
            # max_tokens=4096,
        )
        return {"answer": response.choices[0].message.content.strip()}
    
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({"h1": h1, "h2": h2, "h3": h3})

    return response["answer"]

async def correct_batch(asr_results, vector):
    sem = asyncio.Semaphore(N_CONCURRENCY)
    async def guarded(r):
        print("Launching correction for:", r[:3])
        async with sem:
            return await correct_one(*r[:3], vector)

    tasks = [asyncio.create_task(guarded(r)) for r in asr_results]
    corrections = await asyncio.gather(*tasks)
    return corrections


def main():
    # load into memory or stream; here we list
    # samples = [s for s in ds]
    ds = iter(load_dataset("ai4bharat/Lahaja",split="test", streaming=True))
    training_ds = iter(load_dataset("ai4bharat/IndicVoices", "hindi", split="train", streaming=True))

    vector = create_vector_db(training_ds)

    all_asr = []
    for i in range(0, TOTAL_SIZE, BATCH_SIZE):
        results_arr = run_asr_batch(ds)
        all_asr.extend(results_arr)
        print(f"ASR batch {i//BATCH_SIZE + 1}/{TOTAL_SIZE//BATCH_SIZE}: done")

    print("All ASR:", all_asr)
    # Async correction
    corrections = asyncio.run(correct_batch(all_asr, vector))

    # Compute WERs
    metrics = {'conf':0, 'w2v':0, 'wh':0, 'gpt':0}
    for (h1, h2, h3, ref), corr in zip(all_asr, corrections):
        metrics['conf'] += wer(ref, h1)
        metrics['w2v'] += wer(ref, h2)
        metrics['wh']  += wer(ref, h3)
        metrics['gpt'] += wer(ref, corr)
        print("h1:", h1, "h2:", h2, "h3:", h3, "corr:", corr, "ref:", ref)

    print("Average WERs:", {k: metrics[k]/TOTAL_SIZE for k in metrics})

if __name__ == '__main__':
    main()