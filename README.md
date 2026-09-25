<p align="center">
    <img align="top" width="30%" src="docs/images/Astralopithecus.png" alt="Astralopithecus"/>
</p>

<div align="center">

# Astralopithecus (Astra + Australopithecus)

**SSAFY 16기 2회차 AI 챌린지: 한국어 이미지 객관식 질의응답**

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-5.16-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-0.20-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/peft)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com)
[![Public](https://img.shields.io/badge/Public-0.97616-success?style=flat-square)](docs/submission_report.md)

_이미지 속 글자를 읽어서 네 개의 선택지 가운데 정답 하나를 고르는 한국어 VQA 모델_

</div>

---

## 소개

SSAFY 16기 2회차 AI 챌린지에 팀 Astralopithecus가 제출한 결과물 저장소입니다.

2026년 9월 26일 기준 Public 0.97616을 기록했습니다.

팀 이름 Astralopithecus는 ChatGPT의 Astra와 Australopithecus(오스트랄로피테쿠스)를 합친 이름입니다.

## 팀

| 이름   | GitHub                                                         | 역할                                                               |
| ------ | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| 이준혁 | [@bnbong](https://github.com/bnbong)                           | 팀장, 파이프라인 설계, Colab 27B 학습, 전체 통합 |
| 유수정 | [@ysj120808](https://github.com/ysj120808)                     | 평가, 제출 파일 검사, 확률 결합                     |
| 김창혁 | [@rlckdur123](https://github.com/rlckdur123)                   | 프롬프트 비교, OCR 실험, 오답 유형 분류             |
| 조수빈 | [@217user217](https://github.com/217user217)                   | 데이터 감사, 고정 분할 확정, 오답 라벨 점검         |
| 이유림 | [@steaksteak25-design](https://github.com/steaksteak25-design) | 공통 학습 코드, 9B 실험, OCR 엔진 비교              |

## 결과

아래 값은 [`docs/submission_report.md`](docs/submission_report.md)의 3.5절에서 발췌했습니다.

fold 0은 train을 이미지 기준으로 나눈 고정 검증셋 1,344문항에서 측정한 정확도이고, Public은 Kaggle의 공개 리더보드 점수입니다.

| 후보                | 설정                         | 학습 데이터              | fold 0     | Public      |
| ------------------- | ---------------------------- | ------------------------ | ---------- | ----------- |
| 3B zero-shot        | Qwen2.5-VL-3B, 프롬프트 p2   | 없음                     | 0.9144     | 0.89871     |
| 9B zero-shot        | Qwen3.5-9B, 프롬프트 p2      | 없음                     | 0.9464     | 미제출      |
| 27B zero-shot       | Qwen3.5-27B, 프롬프트 v1     | 없음                     | 0.9539     | 0.95889     |
| LoRA p2             | Qwen3.5-27B, 기본 해상도, p2 | fold 1~4 (5,358)         | 0.9583     | 0.96544     |
| 확대 LoRA p2        | Qwen3.5-27B, 이미지 확대, p2 | fold 1~4 (5,358)         | **0.9688** | 0.96931     |
| 확대 전체 데이터 p2 | Qwen3.5-27B, 이미지 확대, p2 | 전체 (6,690)             | 검증 불가  | 0.96901     |
| Gemma 4 zero-shot   | Gemma-4-31B-it, 시각 토큰 1,120, p2 | 없음              | 0.9695     | 0.96484     |
| 두 모델 결합        | 위 27B 두 모델의 확률 균등 평균 | 위 두 모델의 학습 데이터 | 검증 불가  | 0.97110     |
| 세 모델 가중 결합   | 확대 LoRA 0.3 + 확대 전체 데이터 0.3 + Gemma 4 0.4 | 위 세 모델의 학습 데이터 | 검증 불가 | **0.97616** |

최종 제출로는 세 모델 가중 결합(0.97616)과, 같은 세 모델을 균등하게 평균한 결합(0.97348) 두 개를
골랐습니다.

Gemma 4의 가중치 0.4는 fold 0에서 그린 곡선을 보고 고른 값입니다. 그 선택이 test에서 어긋날
가능성에 대비해, 가중치를 고르지 않은 균등 결합을 함께 남겨 두었습니다.

## 방법

**모델과 학습.** Qwen3.5-27B를 bf16으로 올리고, 언어 경로에만 LoRA를 붙여서 정답 글자 한 개에만 손실을 거는 answer-only 방식으로 학습했습니다. 비전 인코더는 얼리고 학습하지 않았습니다.

**프롬프트.** 이미지 속 글자를 먼저 읽으라고 지시하는 프롬프트 p2(`v2_read_text`)를 fold 0에서 비교해 골랐습니다. 프롬프트는 system turn 없이 user turn 하나로 구성하고 이미지를 텍스트 앞에
둡니다.

**이미지 확대.** 작은 글자를 틀리는 문항이 많았기 때문에 processor의 `min_pixels`를 올려서 작은 이미지를 확대해 넣었습니다. zero-shot 기준으로 fold 0 정확도가 0.9591에서 0.9635로 올랐고, 작은 글자 때문에 틀렸던 8문항 가운데 6문항을 회복했습니다.

**확률 결합.** fold 1~4로 학습한 모델과 전체 데이터로 학습한 모델이 낸 선택지별 확률을 0.5 대 0.5로 평균 냈습니다. 가중치는 검증셋에서 탐색하지 않고 미리 정한 값을 그대로 썼습니다.

**다른 계열 모델 결합.** 같은 Qwen3.5-27B 계열 안에서는 배깅과 dev 세트 추가, epoch 연장 가운데 무엇으로도 두 모델 결합(0.97110)을 넘지 못했기 때문에, 시각 인코더 자체가 다른 Gemma-4-31B-it를 학습 없이 추론만 시켜서 세 번째 확률로 더했습니다. Gemma는 틀린 문항에서도 확률을 1.0에 가깝게 실어 주는 과신 모델이라서 가중치를 절반 아래인 0.4로 낮추어 섞었고, 그 결과 Public 점수가 0.97616까지 올랐습니다.

**검증.** 이미지 해시를 그룹 키로 삼은 고정 fold 0에서만 모든 비교를 수행했고, 두 예측이 갈린 문항에는 McNemar 검정을 적용했습니다. fold 0의 오답은 전수로 라벨을 점검해서 라벨 자체가 잘못된 문항 35건을 확인했고, 이 값에서 달성 가능한 상한을 약 0.974로 추정했습니다.

## 구조

```mermaid
graph LR
    D[train.csv / test.csv<br/>이미지 6,714 + 6,714] --> S[고정 fold 0 분할<br/>이미지 해시 그룹]
    S --> P[preflight<br/>속도·VRAM·설정 점검]
    P --> T[answer-only LoRA 학습<br/>Colab A100 80GB bf16]
    T --> I[추론<br/>선택지별 확률 CSV]
    I --> B[세 모델 확률 가중 평균 결합]
    B --> U[제출 파일 생성과 형식 검사]
```

| 디렉터리       | 내용                                                                                    |
| -------------- | --------------------------------------------------------------------------------------- |
| `vqa/`         | 학습과 추론, 평가, 제출을 담당하는 공통 파이썬 패키지입니다.                            |
| `configs/`     | 모델과 정밀도, 해상도별 학습 설정 YAML 파일입니다.                                      |
| `scripts/`     | 분할 생성과 예측 비교, 확률 결합, 제출 검사를 수행하는 실행 스크립트입니다.             |
| `notebooks/`   | Colab과 실습실 PC에서 여는 진입점 노트북이며 `scripts/build_notebooks.py`가 생성합니다. |
| `results/`     | 실제로 실행을 마친 노트북 사본과 실측 기록입니다.                                       |
| `docs/`        | 제출 설명서와 발표 개요, 팀 운영 문서, 환경 구축 안내입니다.                            |
| `experiments/` | 최종 파이프라인에 들어가지 않은 팀원들의 실험 코드입니다.                               |
| `tests/`       | 프롬프트와 라벨 마스킹, 평가와 제출 경로를 검증하는 pytest 모음입니다.                  |

## 실행

라이선스 이유로 대회 데이터셋은 이 저장소에 들어 있지 않습니다. 

`train/`과 `test/` 폴더, `train.csv`, `test.csv`, `sample_submission.csv`를 프로젝트 루트에 놓고 시작하십시오.

```
pip install -q -r requirements/colab.txt
pip install -q --no-deps -r requirements/colab-nodeps.txt   # fla 가속 커널. 빠지면 약 네 배 느려집니다
python scripts/folds_to_split.py                            # 고정 fold 0 분할 생성
python -m vqa.preflight --config configs/colab_qwen35_27b_bf16_upscale.yaml
python -m vqa.train --config configs/colab_qwen35_27b_bf16_upscale.yaml --run-id ft27_upscale \
    --split artifacts/data/split_fold0.json --set prompt_id=v2_read_text
python -m vqa.predict --config configs/colab_qwen35_27b_bf16_upscale.yaml --run-id ft27_upscale \
    --csv test.csv --adapter <RUN_ROOT>/ft27_upscale/final_adapter --out-name test_ft --set prompt_id=v2_read_text
python -m vqa.predict --config configs/colab_gemma4_31b_bf16.yaml --run-id gemma4_base \
    --csv test.csv --out-name test_base --set prompt_id=v2_read_text
python scripts/blend_preds.py --a <모델 A의 예측 CSV> --b <모델 B의 예측 CSV> --c <Gemma의 예측 CSV> \
    --w-c 0.4 --sample sample_submission.csv --submission submission.csv
```

`--w-c`는 Gemma의 가중치만 정하고 나머지를 A와 B가 반씩 나누므로, `--w-c 0.4`가 곧 A 0.3과 B 0.3과 Gemma 0.4입니다. `--c`를 빼면 A와 B만 `--w-a`로 섞습니다.

환경 설치와 run-id 규칙, 제출 파일 sha256을 포함한 상세 절차는 [`docs/submission_report.md`](docs/submission_report.md)의 5절에 있습니다.


## 문서

| 문서                                                             | 내용                                                                                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [`docs/submission_report.md`](docs/submission_report.md)         | 대회 제출용 설명서입니다. 데이터 처리와 모델 개발 과정, 재현 절차를 한 파일에 정리했습니다. |
| [`docs/presentation_outline.md`](docs/presentation_outline.md)   | 발표 개요와 후보별 비교 표입니다.                                                           |
| [`results/README.md`](results/README.md)                         | Colab A100에서 실측한 속도와 VRAM, 정확도 기록입니다.                                       |
| [`docs/README_team.md`](docs/README_team.md)                     | 팀 내부 운영 문서이며 담당과 게이트 조건, 자주 쓰는 명령이 들어 있습니다.                   |
| [`docs/discussion_fla_kernel.md`](docs/discussion_fla_kernel.md) | fla 가속 커널이 빠졌을 때 학습이 느려지는 현상을 추적한 기록입니다.                         |
| [`docs/history_cleanup.md`](docs/history_cleanup.md)             | 공개 전환 전 이력 정리 절차입니다.                                                          |
| [`docs/plan_initial.md`](docs/plan_initial.md)                   | D1 계획서입니다.                                                                            |

## 기술 스택

| 영역      | 사용 기술                                                                 |
| --------- | ------------------------------------------------------------------------- |
| 모델      | Qwen3.5-27B(최종 채택), Gemma-4-31B-it(결합에 추가), Qwen3.5-9B, Qwen2.5-VL-3B       |
| 학습      | PyTorch 2.11, Transformers 5.16, PEFT 0.20 (LoRA), bitsandbytes, fla-core |
| 실행 환경 | Google Colab A100 80GB, Windows RTX 5060 Ti 16GB                          |
| 데이터    | pandas, Pillow, ImageHash, scikit-learn                                   |
| 검증      | pytest, McNemar 검정, 고정 fold 0 분할                                    |
