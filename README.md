# 프로젝트 소개
이 소프트웨어는 **영어 뉴스 기사**를 입력하면,

T5-small 모델을 이용해 기사를 자동으로 요약하고

DistilBERT 기반 모델로 감정(긍정/부정) 분석을 수행합니다.

# 주요 기능
**1. 영어 뉴스 기사 입력**

사용자가 입력한 영어 기사를 처리합니다.

**2. 자동 요약**

Hugging Face의 t5-small 모델로 기사의 핵심 내용을 1~2문장으로 요약합니다.

**3. 감정 분석**

distilbert-base-uncased 모델을 사용해 입력 기사의 감정을 "POSITIVE" 또는 "NEGATIVE"로 분류하고, 신뢰도 점수(Confidence)도 제공합니다. 이로 인해 기사의 작성자에 기분을 파악해 객관적인지 확인합니다.

**4. 결과 출력**

원문, 요약문, 감정 분석 결과(라벨/신뢰도)를 출력합니다.

# 설치 방법
```
pip install transformers
pip install torch
```

# 사용 예시
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

******************************************
# 입력: 영어 뉴스 기사(여기 코드에다가 뉴스 원문 입력)

text = """
Apple has announced a new line of products during their Worldwide Developers Conference,
including a new generation of M-series chips and the release date for the long-awaited Vision Pro headset.
Industry analysts believe this marks a major step forward for Apple in the competitive tech landscape.
"""
******************************************

# 요약 모델: T5-small

sum_tokenizer = AutoTokenizer.from_pretrained("t5-small")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

input_text = "summarize: " + text
input_ids = sum_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = sum_model.generate(input_ids, max_length=80, min_length=20, num_beams=2, early_stopping=True)
summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 감정 분석 모델: distilbert-base-uncased

sent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sent_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

sent_inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
sent_outputs = sent_model(**sent_inputs)
probs = torch.nn.functional.softmax(sent_outputs.logits, dim=1)

label_index = torch.argmax(probs).item()
labels = ["NEGATIVE", "POSITIVE"]
sentiment_label = labels[label_index]
sentiment_score = probs[0][label_index].item()

print("원문 기사:\n", text)
print("\n요약문:\n", summary)
print("\n감정 분석 결과:")
print(f"Label: {sentiment_label} (Confidence: {sentiment_score:.2f})")
```

# 입출력 예시
```
원문 기사:
Apple has announced a new line of products during their Worldwide Developers Conference...

요약문:
Apple announced new products including M-series chips and Vision Pro headset at WWDC.

감정 분석 결과:
Label: POSITIVE (Confidence: 0.85)
```

# 참고사항
Python 3.8 이상 권장

최초 실행 시 모델 자동 다운로드

추가 데이터 파일 불필요

# 문의
이슈나 문의사항은 GitHub Issues로 남겨주세요.
