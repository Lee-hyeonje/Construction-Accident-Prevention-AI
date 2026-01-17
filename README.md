# Construction Accident Prevention AI

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Ko--Gemma--2--9b--Safety--FineTuned-blue)](https://huggingface.co/lee124/Ko-Gemma-2-9b-Safety-FineTuned)

본 프로젝트에서 Fine-tuning한 모델의 가중치(Weights)는 Hugging Face에서 다운로드할 수 있습니다.
> **Note**: 위 링크에서 `adapter_model.safetensors` 및 설정 파일을 다운로드하여 베이스 모델과 병합하거나, 아래 코드로 바로 불러올 수 있습니다.
## 🚀 Quick Start
Hugging Face에 등록된 모델을 `transformers` 라이브러리를 통해 바로 로드하여 사용할 수 있습니다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Hugging Face Model ID
model_id = "lee124/Ko-Gemma-2-9b-Safety-FineTuned"

# 1. Load Tokenizer & Model
# 어댑터(LoRA) 가중치가 자동으로 베이스 모델과 병합되어 로드됩니다.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16  # GPU 환경에 따라 float16 사용 가능
)

# 2. Inference Example
input_text = "건설 현장에서 추락 사고를 예방하려면 어떤 조치가 필요한가요?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 3. Generate Output
outputs = model.generate(
    **inputs, 
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(outputs, skip_special_tokens=True))
```

## 프로젝트 개요

건설 현장 사고 데이터를 기반으로 재발 방지 대책 및 향후 조치 계획을 자동으로 생성하는 AI 시스템입니다. 
RAG(Retrieval-Augmented Generation) 기반으로 104개의 건설안전지침 문서를 활용하여 
현장 맞춤형 안전 대책을 제공합니다.

## 주요 과제

- **데이터 증강**: BERT 기반 Random Masking Replacement로 훈련 데이터 확장
- **LLM 파인튜닝**: Ko-Gemma-2-9B 모델을 QLoRA로 효율적 학습
- **RAG 시스템**: FAISS 벡터 스토어 + Cross-Encoder 재순위화
- **자동 답변 생성**: 건설 현장 사고에 대한 맞춤형 안전 대책 생성

### 사용한 모델 
- **LLM**: `rtzr/ko-gemma-2-9b-it` (4-bit 양자화)
- **Embeddings**: `jhgan/ko-sbert-sts`
- **Cross-Encoder**: `bongsoo/albert-small-kor-cross-encoder-v1`
- **Data Augmentation**: `monologg/koelectra-base-v3-generator`

## Data Engineering & 전처리

### 2.1 PDF 데이터 정제 (Knowledge Base)

검색 성능과 LLM 처리 효율성을 극대화하기 위해 PDF 원문 데이터를 정밀하게 전처리했습니다.

- **불필요 정보 제거**: 정규표현식을 사용하여 문서 상단 반복 문구('KOSHA GUIDE'), 그림 캡션, 비표준 유니코드(PUA) 문자 등을 제거

- **문서 분할(Chunking)**:
  - `chunk_size`: 500자 단위 분할
  - `overlap`: 문맥 보존을 위해 10%(50자) 중첩 허용

### 2.2 정형 데이터 전처리 및 증강

- **컬럼 세분화**: '공사종류', '공종', '사고객체' 등 계층형 데이터를 대/중분류로 파싱하여 특징 명확화

- **항목 통합**: '인적사고' 컬럼의 유사 항목을 통합하여 학습 효율 증대

- **RMR(Random Masking Replacement) 증강**:
  - 중분류 기준 데이터 불균형 해결을 위해 BERT 기반 RMR 기법 적용
  - 의미가 자연스러운 토큰으로 대체된 증강 데이터를 생성하여 모델의 일반화 성능 향상

## Modeling & RAG 아키텍처

### 3.1 Base Model 선정

- Llama3, Polyglot, Qwen2.5 등 다양한 모델 비교 실험 수행
- 한국어 이해도 및 지시 이행 능력이 가장 뛰어난 Ko-Gemma2 9B를 최종 모델로 선정

### 3.2 RAG 고도화: Cross Encoder 리랭킹

단순 Vector Search의 노이즈를 줄이기 위해 2단계 필터링 구조를 도입했습니다.

- **유사도 재평가**: 검색된 문서 중 질문과 의미적 유사도가 낮은 문서를 다시 계산
- **임계치(Threshold) 적용**: 유사도 점수 0.55 이상의 문서만 컨텍스트로 활용하여 답변 왜곡 방지

### 3.3 학습 파라미터 및 프롬프트 최적화

- **LoRA(Low-Rank Adaptation) 학습**:
  - 초기 설정: `r=16`, `alpha=32`
  - 최종 최적화: `r=8`, `alpha=16` 설정을 통해 효율적인 파인튜닝 수행

- **질문 조합 최적화**: 실험 결과, [공종(중분류) + 인적사고 + 사고원인] 조합의 템플릿이 답변 생성에 가장 효과적임을 확인

## 성능 향상 전략 및 후처리

### 4.1 자체 평가 지표 구축

- **Validation Set**: 훈련 데이터의 10%를 검증 데이터로 분리
- **Metric**: 생성 답변과 정답 간의 Cosine Similarity를 활용하여 의미론적 유사도를 정량적으로 모니터링하며 모델 개선

### 4.2 생성 답변 후처리 (Post-processing)

- **중복 제거**: 답변 내에 질문 내용이 반복 포함된 경우 필터링
- **가독성 개선**: 번호 나열 방식 대신 쉼표(",")를 사용하여 자연스러운 문장으로 변환
- **텍스트 정제**: 불필요한 공백, 중복 쉼표, 특수 문자 제거로 최종 출력 품질 향상
