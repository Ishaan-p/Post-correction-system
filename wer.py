from jiwer import wer, compute_measures

# ground truth (verbatim) and three model outputs and chatgpt
reference = "किम ने उन्नीस सौ तिरानवे में फ़िल्म उद्योग से संन्यास ले लिया उनकी आखिरी फ़िल्में प्रतिकार हनीमून बलवान मुस्कुराहट और बुलंद में अतिथि भूमिकाएँ थीं"
hyp1 = "क्या आप चेक कर सकते हैं कि बनारस हिंदू विश्वविद्यालय शैक्षणिक वर्ष दो हज़ार उन्नीस दो हज़ार बीस में ए आई सी टी ई अनुमोदित संस्थान था या नहीं"  
hyp2 = "क्या आप चेक कर सकते हैं कि बनारस हिंदू विश्वविद्यालय शैक्षणिक वर्ष 201920 में एआईसीटीई अनुमोदित संस्थान था या नहीं"   
hyp3 = "पहर म तरवीन ने कमरम पते ने को मिशनपीकतनतपरेन म मे भिन पनी कषय ना नेट पपभल दे हो"
hyp4 = "किम ने उन्नीस सौ तिरानवे में फ़िल्म उद्योग से संन्यास ले लिया उनकी आखिरी फिल्में प्रतिकार हनीमून बलवान मुस्कहट और बुलंद में अतिथि भूमिकाएँ थीं"  

# simple WER
print("Model 1 WER:", wer(reference, hyp1))
print("Model 2 WER:", wer(reference, hyp2))
print("Model 3 WER:", wer(reference, hyp3))
print("ChatGPT WER:", wer(reference, hyp4))

# detailed counts
measures = compute_measures(reference, hyp4)
print(measures)   # {'wer': ..., 'substitutions': X, 'deletions': Y, 'insertions': Z, ...}
