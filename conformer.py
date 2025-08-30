import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.freeze() # inference mode
model = model.to(device) # transfer model to device