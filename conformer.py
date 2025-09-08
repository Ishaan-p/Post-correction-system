import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained("ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.freeze() # inference mode
model = model.to(device) # transfer model to device

model.cur_decoder = "ctc"
ctc_text = model.transcribe(['./sample.wav'], batch_size=1,logprobs=False, language_id='hi')[0]
print(ctc_text)
