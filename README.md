<p align="center"><img align="top" width="30%" src="docs/images/Astralopithecus.png" alt="Astralopithecus"/></p>

<div align="center">

# Astralopithecus (Astra + Australopithecus)

**SSAFY 16기 2회차 AI 챌린지: 한국어 이미지 객관식 질의응답**

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-5.16-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-0.20-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/peft)
[![Public](https://img.shields.io/badge/Public-0.97765-success?style=flat-square)](docs/submission_report.md)

_이미지 속 글자를 읽어서 네 개의 선택지 가운데 정답 하나를 고르는 한국어 VQA 모델_

</div>

## 소개

SSAFY 16기 2회차 AI 챌린지는 이미지와 한국어 질문을 보고 네 개의 선택지 가운데 정답을 고르는 객관식 VQA 대회였습니다. 팀 Astralopithecus는 2026년 9월 26일 기준 Public 0.97765를 기록했고, 최종 1순위와 2순위 제출물이 이 점수로 동점이었습니다. 이 저장소는 팀이 만든 코드와 실험 기록을 남겨 두려고 보존하는 저장소입니다.

## 팀

| 이름   | GitHub                                                         | 역할                                             |
| ------ | -------------------------------------------------------------- | ------------------------------------------------ |
| 이준혁 | [@bnbong](https://github.com/bnbong)                           | 팀장, 파이프라인 설계, Colab 27B 학습, 전체 통합 |
| 유수정 | [@ysj120808](https://github.com/ysj120808)                     | 평가, 제출 파일 검사, 확률 결합                  |
| 김창혁 | [@rlckdur123](https://github.com/rlckdur123)                   | 프롬프트 비교, OCR 실험, 오답 유형 분류          |
| 조수빈 | [@217user217](https://github.com/217user217)                   | 데이터 감사, 고정 분할 확정, 오답 라벨 점검      |
| 이유림 | [@steaksteak25-design](https://github.com/steaksteak25-design) | 공통 학습 코드, 9B 실험, OCR 엔진 비교           |

## 결과

fold 0은 train을 이미지 기준으로 나눈 고정 검증셋 1,344문항에서 잰 정확도이고, Public은 Kaggle 공개 리더보드 점수입니다. 전체 후보와 판단 근거는 [`docs/submission_report.md`](docs/submission_report.md)에 정리되어 있습니다.

| 후보                      | 설정                                                                     | fold 0     | Public                  |
| ------------------------- | ------------------------------------------------------------------------ | ---------- | ----------------------- |
| 3B zero-shot              | Qwen2.5-VL-3B, 프롬프트 p2                                               | 0.9144     | 0.89871                 |
| 27B zero-shot             | Qwen3.5-27B, 프롬프트 v1                                                 | 0.9539     | 0.95889                 |
| 확대 LoRA p2 (A)          | Qwen3.5-27B, 이미지 확대, p2, fold 1~4 (5,358)                           | 0.9688     | 0.96931                 |
| 확대 전체 데이터 p2 (B)   | Qwen3.5-27B, 이미지 확대, p2, 전체 (6,690)                               | 검증 불가  | 0.96901                 |
| Gemma 4 zero-shot         | Gemma-4-31B-it, 시각 토큰 1,120, p2                                      | 0.9695     | 0.96484                 |
| Gemma 4 QLoRA             | Gemma-4-31B-it, nf4 QLoRA, 시각 토큰 1,120, p2, fold 1~4 (5,358)         | **0.9747** | 단독 미제출             |
| 세 모델 가중 결합 (QLoRA) | A 0.3 + B 0.3 + Gemma 4 QLoRA 0.4                                        | 검증 불가  | **0.97765** (최종 1순위) |
| 네 모델 가중 결합         | A 0.3 + B 0.3 + Gemma 4 QLoRA 0.2 + Gemma 4 전체 QLoRA 0.2               | 검증 불가  | 0.97765 (최종 2순위)    |

## 방법 요약

- **모델과 학습**: Qwen3.5-27B를 bf16으로 올리고 언어 경로에만 LoRA를 붙였습니다. 정답 글자 한 개에만 손실을 거는 answer-only 방식으로 학습했고, 비전 인코더는 학습하지 않았습니다.
- **프롬프트 p2**: 이미지 속 글자를 먼저 읽으라고 지시하는 프롬프트 `v2_read_text`를 fold 0에서 비교해 골랐습니다.
- **이미지 확대**: processor의 `min_pixels`를 올려 작은 이미지를 확대해 넣었습니다. zero-shot 기준으로 fold 0 정확도가 0.9591에서 0.9635로 올랐습니다.
- **확률 가중 결합**: 모델마다 낸 선택지별 확률을 가중 평균했습니다. 과신이 심한 Gemma의 가중치 0.4는 fold 0 곡선을 보고 미리 고정한 값이며, Public 점수를 보고 고른 값이 아닙니다.
- **Gemma nf4 QLoRA**: Gemma-4-31B-it는 bf16으로는 A100 80GB 한 장에서 학습할 수 없어서, 가중치를 nf4로 양자화하고 LoRA adapter만 학습했습니다. 학습 peak VRAM은 약 32GB였습니다.

## 저장소 구성

| 경로              | 내용                                                                 |
| ----------------- | -------------------------------------------------------------------- |
| `vqa/`            | 학습과 추론, 평가, 제출을 담당하는 공통 파이썬 패키지입니다.         |
| `configs/`        | 모델과 정밀도, 해상도별 학습 설정 YAML 파일입니다.                   |
| `scripts/`        | 분할 생성과 예측 비교, 확률 결합, 제출 검사를 수행하는 스크립트입니다. |
| `notebooks/`      | Colab과 실습실 PC에서 쓴 진입점 노트북입니다.                        |
| `results/`        | 실행을 마친 노트북 사본과 날짜별 실험 기록입니다.                    |
| `docs/`           | 제출 보고서와 참고 기록입니다.                                       |
| `experiments/`    | 최종 파이프라인에 들어가지 않은 팀원들의 실험 코드입니다.            |
| `tests/`          | 프롬프트와 라벨 마스킹, 평가와 제출 경로를 검증하는 pytest 모음입니다. |
| `artifacts/data/` | 고정 분할 파일과 id 목록, 라벨 점검 결과입니다.                      |
| `fold_csv_files/` | 팀이 확정한 fold 분할을 id 단위로 담은 CSV입니다. |

## 이 저장소에 포함하지 않은 것

대회 원본 데이터(이미지와 CSV), 모델 checkpoint와 LoRA adapter 가중치, 제출 CSV는 라이선스와 용량 문제 때문에 이 저장소에 포함하지 않았습니다. 분할 파일과 id 목록은 이미지나 질문 본문을 담고 있지 않아서 포함했습니다. 다만 dev 문항의 다수결 라벨 파일(`fold_csv_files/dev_aux.csv`)은 정답에 준하는 값을 담고 있어서 함께 제외했습니다.

## 문서

| 문서                                                             | 내용                                                         |
| ---------------------------------------------------------------- | ------------------------------------------------------------ |
| [`docs/submission_report.md`](docs/submission_report.md)         | 대회에 제출한 보고서입니다.                                  |
| [`results/README.md`](results/README.md)                         | 날짜별 실험의 설정과 측정값, 채택 여부를 정리한 기록입니다.  |
| [`docs/discussion_fla_kernel.md`](docs/discussion_fla_kernel.md) | fla 가속 커널이 빠졌을 때 학습이 느려지는 현상을 추적한 기록입니다. |
