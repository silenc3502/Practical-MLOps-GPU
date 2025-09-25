import os
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ✅ 현재 스크립트 기준 상대 경로로 모델 위치 지정
model_path = os.path.join(os.path.dirname(__file__), "results")
print(f"📂 Loading model from: {model_path}")

# 모델 폴더 존재 여부 체크
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model path not found: {model_path}")

# 토크나이저와 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# LabelEncoder 로드
label_encoder_path = os.path.join(model_path, "label_encoder.pkl")
with open(label_encoder_path, "rb") as f:
    le = pickle.load(f)

# 파이프라인 생성
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 테스트 문장
test_texts = [
    "좋아요",
    "별로예요",
    "보통이에요",
    "정말 좋아요",
    "싫어요",
    "괜찮아요"
]

# 추론 및 라벨 복원
for text in test_texts:
    result = clf(text)
    label_id = int(result[0]["label"].split("_")[-1])  # LABEL_숫자
    label_str = le.inverse_transform([label_id])[0]  # 원래 sentiment 문자열로 변환
    print(f"{text} -> {label_str} (score: {result[0]['score']:.4f})")
