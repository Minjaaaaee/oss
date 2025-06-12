# 1. 설치 (최초 1회만)
'''
!pip install transformers
!pip install torch
'''

# 2. 라이브러리 로딩

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 3. 입력: 영어 뉴스 기사

text = """
Apple has announced a new line of products during their Worldwide Developers Conference,
including a new generation of M-series chips and the release date for the long-awaited Vision Pro headset.
Industry analysts believe this marks a major step forward for Apple in the competitive tech landscape.
"""

# 4. 요약 모델: T5-small

sum_tokenizer = AutoTokenizer.from_pretrained("t5-small")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

input_text = "summarize: " + text
input_ids = sum_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = sum_model.generate(
    input_ids,
    max_length=80,
    min_length=20,
    num_beams=2,
    early_stopping=True
)
summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 5. 감정 분석 모델 - distilbert-base-uncased (가벼운 버전)
sent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sent_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

sent_inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
sent_outputs = sent_model(**sent_inputs)
probs = F.softmax(sent_outputs.logits, dim=1)

label_index = torch.argmax(probs).item()
labels = ["NEGATIVE", "POSITIVE"]
sentiment_label = labels[label_index]
sentiment_score = probs[0][label_index].item()

# 6. 출력

print("원문 기사:\n")
print(text)

print("\n요약문:\n")
print(summary)

print("\n감정 분석 결과:")
print(f"Label: {sentiment_label} (Confidence: {sentiment_score:.2f})")
