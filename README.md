# 텍스트 이미지 기반 질의응답(VQA) 챌린지

이미지 한 장과 질문 하나, a부터 d까지 네 개의 선지를 받아서 정답 하나를 고릅니다. 평가 지표는 정확도입니다. Colab에서 Qwen3.5-27B를, 실습실 PC에서 Qwen3.5-9B를 answer-only LoRA로 학습합니다. 각 모델이 낸 선지별 확률을 고정 validation에서 비교해 결합합니다. 설계 근거는 `vqa_5day_implementation_plan.md`에, 인계 순서와 게이트 조건은 `docs/README_team.md`에 있습니다.

## 데이터

대회 데이터셋은 이 저장소에 들어 있지 않습니다. 팀 내부 경로로만 받습니다. `train/`, `test/`, `dev/` 폴더와 `train.csv`, `test.csv`, `dev.csv`, `sample_submission.csv`를 프로젝트 루트에 놓거나, 다른 위치에 두었다면 환경 변수 `VQA_DATA_ROOT`에 그 경로를 지정하십시오. 데이터 파일은 어떤 경우에도 commit 하지 마십시오. `.gitignore`가 데이터와 파생 산출물을 제외합니다. 새 파일을 추가할 때에는 `git status`를 확인하십시오.

## 설치

실습실 PC는 Windows입니다. GPU는 Blackwell 세대인 RTX 5060 Ti 16GB입니다. 따라서 torch는 반드시 cu128 빌드를 설치해야 합니다. 실습실에서 인터넷이 됩니다. 따라서 pip 설치와 Hugging Face 다운로드가 모두 동작합니다.

```
git clone git@github.com:bnbong/Astralopithecus.git
cd Astralopithecus
conda create -n vqa-tf5 python=3.11 -y && conda activate vqa-tf5   # 파이썬 버전은 docs/lab_setup.md 참고
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
pip install -r requirements/lab_windows_tf5.txt
```

Qwen3.5는 transformers 5.x에서만 로드됩니다. 그래서 환경을 두 개로 나눕니다. `vqa-tf4`는 transformers 4.x로 Qwen2.5-VL-3B baseline을 돌립니다. `vqa-tf5`는 transformers 5.x로 Qwen3.5 계열을 돌립니다. Colab에서는 `requirements/colab.txt`를 씁니다. 설치가 끝나면 무엇보다 먼저 `notebooks/00_lab_env_check.ipynb`를 실행하십시오. 이 노트북이 통과해야 나머지가 동작합니다. 상세한 절차는 `docs/lab_setup.md`에 있습니다.

## 노트북과 담당

| 노트북 | 담당 | 목적 |
|---|---|---|
| `00_lab_env_check.ipynb` | 전원 | GPU와 패키지 설치 확인 |
| `01_data_audit_split.ipynb` | 수빈 | 데이터 검사와 고정 split 생성 |
| `10_lab_baseline_zero_shot.ipynb` | 누구나 | 학습 없는 첫 정확도와 첫 제출 후보 |
| `11_lab_prompt_ocr_compare.ipynb` | 창혁 | 프롬프트와 OCR, 해상도 동결 |
| `12_lab_train_9b_qlora.ipynb` | 유림 | 실습실 9B NF4 QLoRA 학습 |
| `20_colab_27b_setup_preflight.ipynb` | 준혁 | Colab 자원 준비와 27B 실측 |
| `21_colab_27b_run.ipynb` | 준혁 | Colab 27B 추론과 학습 |
| `30_eval_submit.ipynb` | 수정 | 평가와 결합, 제출 파일 생성 |

첫 정확도 숫자를 빨리 보고 싶은 사람은 노트북 10번을 3B 모델로 열면 됩니다.

## 자주 쓰는 명령

```
python -m vqa.train --config configs/lab_qwen35_9b.yaml --run-id ft9_v1 --split artifacts/data/split_v1.json
python -m vqa.jobs start --name ft9_v1 --auto-restart 5 -- python -m vqa.train --config configs/lab_qwen35_9b.yaml --run-id ft9_v1
python -m vqa.jobs status --name ft9_v1
python -m vqa.predict --config configs/lab_qwen35_9b.yaml --run-id ft9_v1 --csv test.csv --out-name ft9_test
python -m vqa.submit --pred artifacts/preds/combined_test.csv --sample sample_submission.csv --out artifacts/submission/final.csv
python scripts/check_submission.py --submission artifacts/submission/final.csv --sample sample_submission.csv
```

`vqa.train` 명령을 똑같이 다시 실행하면 마지막으로 완결된 checkpoint에서 이어집니다. `--resume auto`가 기본값입니다. 따라서 명령을 고칠 필요가 없습니다. `vqa.jobs start`로 띄운 작업은 노트북 셀을 멈추거나 브라우저를 닫아도 계속 돌아갑니다. 다만 Colab 가상 머신이 회수되면 함께 죽습니다. 그때에는 원래 실행했던 시작 셀을 그대로 다시 실행하면 됩니다. 상황별 대응은 `docs/colab_survival.md`에 정리되어 있습니다.

## 팀 규칙

- validation split은 고정합니다. 내용 해시는 `c929ac2059dccfa127c67d5f738f1bbc648c031abe73ec6bbedc18f48b54d5cb`입니다. train은 6214문항, validation은 500문항입니다. 값이 다르면 즉시 팀 채널에 알리십시오.
- test로는 어떤 조정도 하지 않습니다. 프롬프트와 하이퍼파라미터는 고정 validation에서만 비교합니다.
- 예측 확률 파일의 컬럼은 `id,p_a,p_b,p_c,p_d,run_id` 순서를 지킵니다.
- 노트북은 직접 고치지 마십시오. `python scripts/build_notebooks.py`로 다시 생성합니다.

## 현재 상태 (2026-09-21)

지금까지 확인된 것은 CPU 단위 테스트가 통과한다는 사실입니다. GPU와 Colab, Windows 실습실에서는 아직 한 번도 실행해 보지 않았습니다. 처음 GPU에서 preflight를 돌릴 때 다음 세 가지를 반드시 확인해야 합니다.

- NF4 4비트 경로의 순전파와 역전파가 정상적으로 동작하는지 확인해야 합니다.
- Qwen3.5의 LoRA 대상 모듈이 올바른지 확인해야 합니다.
- 학습과 추론의 prefix가 일치하는지 확인해야 합니다.

저장소 루트의 `check_submission.py`와 `eval_val.py`는 확률 컬럼으로 `prob_a`부터 `prob_d`까지를, `vqa` 패키지와 `scripts/`는 `p_a`부터 `p_d`까지를 씁니다.
