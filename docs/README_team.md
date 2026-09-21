# 팀 운영 안내

이 문서는 다섯 명이 각자 오늘 무엇을 열고 무엇을 넘겨야 하는지를 정리합니다. 상세한 실행 절차는
노트북 안의 설명 셀에 들어 있으므로, 이 문서에서는 순서와 인계 조건만 다룹니다.

노트북은 모두 얇은 진입점입니다. 모델을 다루는 코드는 `vqa/` 패키지와 `scripts/` 안에 있고,
노트북은 그 명령을 호출하고 결과를 보여 주는 역할만 맡습니다. 그래서 같은 셀을 여러 번 실행해도
대체로 안전하며, 중단된 작업은 같은 셀을 다시 실행해서 이어 갈 수 있습니다.

## 1. 담당과 오늘 열어야 할 노트북

| 담당 | 역할 | 오늘 여는 노트북 |
|---|---|---|
| 준혁 | 전체 통합과 Colab 27B 실행 | `20_colab_27b_setup_preflight.ipynb`, 이어서 `21_colab_27b_run.ipynb` |
| 수빈 | 데이터 검사와 공식 분할 확정 | `01_data_audit_split.ipynb` |
| 유림 | 공통 학습 코드와 실습실 9B 학습 | `12_lab_train_9b_qlora.ipynb` |
| 창혁 | 프롬프트와 OCR, 해상도 결정 | `11_lab_prompt_ocr_compare.ipynb` |
| 수정 | 평가와 결합, 제출 검사 | `30_eval_submit.ipynb` |

시작하기 전에 전원이 `00_lab_env_check.ipynb` 를 각자의 실습실 PC에서 한 번씩 실행합니다.
이 노트북이 끝나야 나머지 노트북이 동작합니다. 실습실 PC는 인터넷에 연결되어 있으므로 온라인
설치를 기본으로 하며, torch 2.11.0과 짝이 되는 `torchvision==0.26.0` 을 같은 cu128 인덱스에서
함께 받아야 합니다. 수빈이 실행하는 데이터 검사는 ImageHash가 없으면 오류로 중단되므로
requirements를 통째로 설치하십시오. 파이썬 버전은 우리가 아직 검증한 값이 없으므로, 노트북
00번이 통과한 버전을 그대로 쓰고 그 값을 팀 채널에 기록합니다. 자세한 내용은
`docs/lab_setup.md` 에 있습니다.

`10_lab_baseline_zero_shot.ipynb` 는 담당자가 정해져 있지 않습니다. 손이 비는 사람이 열어서
학습 없이 첫 정확도 숫자와 첫 제출 후보를 확보하는 용도로 사용합니다. 여러 대의 PC로 test를
나누고 싶을 때는 각자 `SHARD` 값만 다르게 지정하면 됩니다.

## 2. D1 시간대별 진행

아래 시각은 시작 시점을 기준으로 한 목표입니다. 다운로드나 검사에 시간이 더 걸리면 시각보다
단계별 완료 조건을 우선합니다.

| 시간대 | 준혁 | 수빈 | 유림 | 창혁 | 수정 |
|---|---|---|---|---|---|
| 0~1시간 | 노트북 20번으로 자원과 가중치를 준비합니다. | 노트북 00번을 마치고 01번으로 CSV와 이미지를 검사합니다. | 노트북 00번을 마치고 9B 로드와 학습 코드를 확인합니다. | 노트북 00번을 마치고 대표 이미지 20장을 살펴봅니다. | 대회 규칙을 확인하고 노트북 30번의 가짜 확률 경로를 완성합니다. |
| 1~4시간 | 입력 설정이 정해지면 27B preflight로 20 step과 50문항을 실측합니다. | split을 확정해서 해시와 함께 공유합니다. | 노트북 12번으로 preflight와 20-step smoke, 중단과 재개 시험을 마칩니다. | split이 나온 뒤 노트북 11번으로 프롬프트와 해상도를 비교합니다. | 소량의 실제 확률로 검사기를 연결하고 시간 예산을 계산합니다. |
| 4시간 지점 | 27B가 예산을 통과하는지 판단합니다. | 필수 데이터 준비 상태를 보고합니다. | 공통 코드와 9B 준비 상태를 보고합니다. | 프롬프트와 OCR, 해상도를 동결합니다. | 남은 시간과 CU, 미완료 조건을 정리합니다. |
| 이후 | 노트북 21번으로 base validation과 전체 test를 실행합니다. | dev 후보를 검수하고 학습 manifest를 동결합니다. | 조건이 갖춰지면 9B 본 학습을 시작합니다. | 필요한 OCR 캐시를 마무리하고 검수를 돕습니다. | base 결과를 검사하고 제출을 준비합니다. |
| 퇴실 전 | 학습 job을 띄웠는지, 아니면 다음 실행 계획을 공유했는지 확인합니다. | 데이터 버전과 출처를 인계합니다. | 완결된 checkpoint와 재개 명령을 확인합니다. | 확정한 입력 설정을 인계합니다. | 진행 상태와 중단 시각, 다음 담당자를 기록합니다. |

## 3. 게이트와 선행 조건

작업 사이에는 순서가 있습니다. 앞 단계가 끝나지 않았는데 다음 단계를 시작하면, 나중에 결과를
비교할 수 없게 됩니다.

- **G0 형식 합의**: 입력 컬럼과 확률 출력 형식을 합의해야 각자 코드를 쓸 수 있습니다. 예측 확률
  파일은 `id,p_a,p_b,p_c,p_d,run_id` 컬럼을 이 순서로 가져야 하며, `vqa/predict.py` 와
  `vqa/submit.py`, `scripts/evaluate_run.py` 가 모두 이 형식을 전제합니다. 저장소에는 별도의
  계약 문서를 두지 않고 `README.md` 의 팀 규칙 절을 기준으로 삼습니다.
- **G1 데이터 확정**: 수빈의 `artifacts/data/split_fold0.json` 이 있어야 창혁과 유림이 시작할 수
  있습니다. 이 파일은 `scripts/folds_to_split.py` 가 팀 공식 fold CSV 두 개를 합쳐서 만듭니다.
  창혁의 개발 표본은 반드시 `train_fold1234` 의 학습 ID에서만 뽑으며, fold 0에서 뽑으면 이후의
  모든 정확도 비교가 오염됩니다.
- **G2 입력과 환경 확정**: 창혁의 `artifacts/input_config_frozen.json` 이 나와야 27B base 캐시를
  만들 수 있습니다. 이 게이트는 모델별로 따로 통과하므로, 9B는 27B 점검이 끝나기를 기다리지
  않습니다.
- **G3 학습 준비 완료**: 학습 manifest를 동결하고 checkpoint 저장과 재개를 확인한 뒤에 본 학습을
  시작합니다. 학습이 시작된 다음에는 데이터를 추가하지 않습니다. 이 게이트에는 preflight의
  `train_infer_prefix_match` 통과도 포함됩니다. 학습 텍스트는 추론용 프롬프트에 정답 letter와
  end-of-turn 접미사를 이어 붙여서 만들기 때문에, 두 경로의 prefix가 어긋나면 학습한 내용이
  추론에 그대로 쓰이지 않습니다.

### 운영진에게 확인할 규칙

아래 두 가지는 우리가 임의로 정할 수 없는 사항이며, 답을 받아야 일정과 후보 구성이 확정됩니다.
D1 오전에 수정이 대표로 질문하고, 받은 답을 팀 채널에 그대로 공유합니다.

**zero-shot 제출을 허용하는지 물어야 합니다.** zero-shot 제출은 미세조정을 하지 않은 base 모델의
예측을 그대로 제출하는 방식입니다. 대회 공지는 모델을 train 데이터로 학습시킨다고 적고 있으므로,
학습을 거치지 않은 예측이 리더보드 확인용으로 허용되는지, 그리고 최종 선택 제출로도 허용되는지를
따로 확인해야 합니다. 이 답에 따라 첫 제출을 언제 올릴 수 있는지가 정해지고, base 모델 단독 결과를
네 개의 결합 후보 안에 남길지도 결정됩니다.

**TTA4를 허용하는지 물어야 합니다.** TTA4는 이미지를 건드리지 않고 선지 순서만 네 가지로 바꿔서
네 번 추론한 다음, 확률을 원래 순서로 되돌려 평균 내는 방식입니다. 선지 위치에 따른 편향을
상쇄하는 효과가 있지만, 규정이 증강을 train과 dev에만 허용하고 있어서 test 추론에 적용하려면
명시적인 허가가 필요합니다. 추론 시간이 네 배로 늘어나는 비용도 있으므로, 허가를 받기 전까지는
코드 기본값인 `--tta 1` 을 유지합니다.

질문은 다음 문구를 그대로 사용하면 됩니다.

> 안녕하세요, [팀명]입니다. 제출 규정에 대해 두 가지 확인 부탁드립니다.
>
> 1. 사전학습 모델을 추가 학습 없이 그대로 추론한 결과(zero-shot)를 제출해도 되는지 궁금합니다.
>    공지에는 모델을 train 데이터로 학습시킨다고 되어 있는데, 학습 파이프라인을 점검하는
>    중간 확인용 제출과 최종 선택 제출 각각에 대해 허용 여부를 알려주시면 감사하겠습니다.
> 2. test 추론 시 이미지는 그대로 두고 선지(a~d)의 순서만 바꿔 여러 번 추론한 뒤 확률을 평균 내는
>    방식(순서 permutation TTA)을 사용해도 되는지 궁금합니다. 규정에 증강은 train과 dev에만
>    허용된다고 되어 있어서, 이 방식이 증강에 해당하는지 판단이 서지 않습니다.

## 4. 공유 폴더 구조

모든 산출물은 아래 구조를 따릅니다. 실습실 PC와 Colab에서 위치만 다르고 구조는 같습니다.

```
프로젝트 루트/
├─ train.csv, test.csv, dev.csv, sample_submission.csv
├─ train/, test/, dev/                  이미지 폴더
├─ vqa/                                 공통 파이썬 코드 (유림이 관리)
├─ scripts/                             실행 스크립트
├─ configs/                             모델과 환경별 설정 파일
├─ requirements/                        환경별 의존성 목록
├─ notebooks/                           이 문서가 안내하는 노트북
├─ docs/                                이 문서와 환경별 안내
└─ artifacts/
   ├─ data/                             split과 검사 결과
   │  ├─ split_fold0.json               공식 분할입니다. git으로 추적합니다.
   │  ├─ val_fold0_ids.txt              `--val-ids` 와 `--ids-file` 에 넘기는 목록입니다.
   │  ├─ README.md                      split_v1 폐기 안내
   │  ├─ split_v1.json, train_ids.txt, val_ids.txt   폐기한 옛 분할입니다.
   │  ├─ split_leakage.csv
   │  ├─ audit_report.json, audit_report.md
   │  └─ near_duplicate_pairs.csv, image_hashes.csv
   ├─ input_config_frozen.json          창혁이 동결한 입력 설정
   ├─ eval/                             평가 결과 JSON과 오답 CSV
   ├─ preds/                            결합용으로 모아 둔 예측 확률 CSV
   ├─ submission/                       제출 후보 CSV
   ├─ experiments.jsonl                 실험 기록 (수정이 관리)
   └─ runs/                             `VQA_RUN_ROOT` 의 기본값입니다.
      ├─ preflight/<시각>.json          preflight 점검 결과
      ├─ env_lock.txt                   노트북 20번이 남기는 설치 버전 고정 기록
      ├─ budget.json                    노트북 20번이 남기는 compute unit 예산 계산
      └─ <run_id>/                      실행별 산출물
         ├─ run_manifest.json
         ├─ train_log.jsonl
         ├─ heartbeat.json
         ├─ checkpoints/step_000123/    COMPLETE 표시가 있어야 완결된 저장입니다.
         ├─ checkpoints/latest.json
         ├─ final_adapter/              학습이 끝난 adapter
         ├─ final_adapter/processor/    재현용 processor 사본. 추론은 읽지 않습니다.
         └─ preds/                      예측 확률 CSV와 manifest
```

예측 파일은 `python -m vqa.predict` 가 `<RUN_ROOT>/<run_id>/preds/` 아래에 만들고, 여러 run의
결과를 결합할 때에만 `artifacts/preds/` 로 모읍니다. 평가 기록은 `scripts/evaluate_run.py` 가
`artifacts/experiments.jsonl` 에 한 줄씩 덧붙입니다. CSV가 아니라 JSONL이므로 표 계산기로 열지
말고 `--json-out` 으로 저장한 요약 JSON을 보십시오.

Colab에서는 `artifacts/runs` 에 해당하는 위치를 Drive에 두고, 이미지와 압축을 푼 데이터는 반드시
런타임 로컬 디스크에 둡니다. Drive에서 작은 이미지 파일을 반복해서 읽으면 속도가 크게 떨어집니다.

## 5. 인계 규칙

작업을 넘길 때는 네 가지를 함께 적습니다. 이 네 가지가 빠지면 받는 사람이 다시 물어봐야 하고,
그동안 GPU가 놀게 됩니다.

1. 결과물이 어디에 있는지 경로를 적습니다.
2. 버전이나 해시를 적습니다. split은 내용 해시를, 제출 파일은 sha256을 함께 적습니다.
3. 어떤 검사를 통과했는지 적습니다.
4. 받는 사람이 실행할 다음 명령을 적습니다.

각 노트북의 마지막 셀이 이 네 가지를 포함한 공유 블록을 출력합니다. 그 출력을 그대로 팀 채널에
붙여 넣으면 됩니다. 직접 문장을 만들 필요는 없습니다.

특히 중요한 인계 항목은 다음과 같습니다.

- 수빈이 넘기는 split 해시는 전원에게서 같은 값이 나와야 합니다. 값이 다르면 서로 다른 데이터로
  평가하고 있는 것이므로 즉시 알려야 합니다.
- 창혁이 동결한 입력 설정은 학습이 시작된 뒤에 바꾸지 않습니다. 바꿔야 할 사정이 생기면 별도
  run으로 기록하고, 같은 조건의 비교라고 보고하지 않습니다.
- 준혁이 공유하는 실행 상태는 다른 사람이 확인할 수 있는 유일한 수단입니다. 노트북을 공유해도
  실행 중인 가상 머신은 공유되지 않기 때문입니다.

## 6. 실습실과 Colab 사이에서 코드를 옮기는 방법

실습실 PC는 인터넷에 연결되어 있으므로 git 원격 저장소를 기본 경로로 씁니다. 저장소는
`git@github.com:bnbong/Astralopithecus.git` 이며 비공개로 운영합니다. 실습실에서는 이 저장소를
clone 해서 쓰고, Colab에서는 노트북 맨 앞에서 같은 저장소를 clone 합니다. 코드가 한 곳에서만
관리되므로 버전이 어긋날 일이 없습니다.

대회 데이터셋은 이 저장소로 옮기지 않습니다. `.gitignore` 가 `train/`, `test/`, `dev/` 와 네 개의
CSV, 그리고 이미지 해시와 예측 결과 같은 파생 산출물을 제외하고 있으며, 데이터는 팀 내부 경로로만
받습니다. 저장소에 새 파일을 추가할 때마다 `git status` 로 데이터가 섞여 들어가지 않았는지
확인하십시오.

git을 쓸 수 없는 상황에서만 Drive나 USB 저장 장치를 대비책으로 씁니다. 데이터 폴더를 뺀 코드만
zip으로 묶고 파일 이름에 날짜와 시각을 넣어서 어느 버전인지 알 수 있게 합니다. 오프라인 PC에
필요한 wheel 파일과 모델 가중치를 준비하는 절차는 `docs/lab_setup.md` 에 정리되어 있습니다.

어느 방법을 쓰든 다음 두 가지는 지켜야 합니다. 첫째, `vqa/` 와 `scripts/`, `configs/` 는 유림이
관리하는 한 곳에서만 고칩니다. 각자 자기 PC에서 고치면 나중에 어느 코드로 만든 결과인지 알 수
없게 됩니다. 둘째, 학습 중간 산출물과 예측 파일은 zip으로 옮기지 말고 각자의 `artifacts/runs`
아래에 두었다가, 필요한 파일만 이름을 정해서 공유합니다.

## 7. 실행 중에 문제가 생겼을 때

- **노트북 셀이 CLI 인자 오류로 실패하는 경우**가 있습니다. 노트북은 합의한 형식에 없는 보조
  인자를 일부 가정해서 작성했습니다. 해당 명령에 `--help` 를 붙여서 실제 인자 이름을 확인하고, 셀 안의
  문자열만 고치면 됩니다. 고친 내용은 팀 채널에 알려서 다른 사람이 같은 곳에서 막히지 않게
  합니다.
- **추론이 중간에 멈춘 경우**에는 같은 셀을 다시 실행합니다. 이미 처리한 ID는 건너뛰고 이어서
  진행합니다.
- **학습이 중단된 경우**에도 같은 시작 셀을 그대로 다시 실행합니다. 저장된 checkpoint에서
  이어집니다. 명령을 고칠 필요가 없습니다.
- **Colab 세션이 끊긴 경우**의 대응은 `docs/colab_survival.md` 에 따로 정리했습니다.
- **GPU가 모자란 경우**에는 한 PC에서 큰 작업을 두 개 동시에 돌리지 않습니다. PC 다섯 대의 VRAM은
  하나로 합쳐지지 않으므로, 같은 학습을 여러 대에서 반복하는 대신 추론 대상 ID를 나눠야 합니다.

## 8. 공식 분할과 기준선 (2026-09-21 확정)

공식 검증셋은 train의 fold 0이며 1,344문항입니다. 파일은 `fold_csv_files/val_fold0.csv` 이고,
`id` 컬럼 하나만 들어 있습니다. 학습에는 `fold_csv_files/train_fold1234.csv` 의 5,358문항을
씁니다. 이 파일에서는 test와 이미지가 완전히 같은 train 24문항을 이미 빼 두었습니다. 분할은
이미지 MD5를 그룹 키로 쓰는 StratifiedGroupKFold 5-fold 결과이며, 근거는
`fold_csv_files/데이터처리_검증분할.md` 에 정리되어 있습니다.

두 CSV를 합쳐서 `vqa.train --split` 에 넘길 JSON을 만드는 명령은 다음과 같습니다.

```
python scripts/folds_to_split.py
```

결과 파일은 `artifacts/data/split_fold0.json` 이고 내용 해시는
`ea2c8572129342f138c551e6e32a3427902c730db349c45d0e5342f03dfccae0` 입니다. 검증 500문항을 쓰던
`artifacts/data/split_v1.json` 은 폐기했습니다.

MD5 그룹만으로 묶었기 때문에 픽셀이 조금 다른 근접 중복은 잡히지 않았습니다. fold 0과 학습
분할에 걸쳐 있는 쌍이 하나 남아 있으며 `train_0395.jpg` 와 `train_4458.jpg` 입니다. perceptual
hash로만 일치하고 문항 수로 보면 무시할 만한 수준이라서 그대로 둡니다.

넘어야 할 기준선은 Qwen2.5-VL-3B를 zero-shot으로 돌리고 프롬프트 p2를 쓴 fold 0 정확도
0.9159입니다. 같은 모델의 p1 정확도는 0.8862였고, Kaggle public 점수는 0.89871이었습니다.

## 9. 27B에서 프롬프트를 비교하는 방법

팀원이 측정한 p2는 `vqa/prompt.py` 에 `v2_read_text` 라는 prompt_id로 들어가 있습니다. 문구와
선택지 표기, 그리고 system turn을 쓰지 않는다는 점까지 `infer_val.py` 의 원본과 같습니다.
기본 prompt_id는 바꾸지 않았으므로, 비교할 때에만 `--set prompt_id=...` 로 지정하십시오.

fold 0에서 두 번 추론한 다음 각각 평가하면 됩니다.

```
python -m vqa.predict --config configs/colab_qwen35_27b_bf16.yaml \
    --run-id colab27b_base_bf16 --csv train.csv \
    --ids-file fold_csv_files/val_fold0.csv --adapter none \
    --set prompt_id=v1_image_only --out-name fold0_base_v1

python -m vqa.predict --config configs/colab_qwen35_27b_bf16.yaml \
    --run-id colab27b_base_bf16 --csv train.csv \
    --ids-file fold_csv_files/val_fold0.csv --adapter none \
    --set prompt_id=v2_read_text --out-name fold0_base_v2

python scripts/evaluate_run.py --pred <RUN_ROOT>/colab27b_base_bf16/preds/fold0_base_v1.csv \
    --gold train.csv --val-ids fold_csv_files/val_fold0.csv --note "27B zero-shot v1"

python scripts/evaluate_run.py --pred <RUN_ROOT>/colab27b_base_bf16/preds/fold0_base_v2.csv \
    --gold train.csv --val-ids fold_csv_files/val_fold0.csv --note "27B zero-shot v2_read_text"
```

`--out-name` 을 서로 다르게 지정해야 두 결과가 덮어써지지 않습니다. 두 평가 결과는
`artifacts/experiments.jsonl` 에 한 줄씩 쌓이므로 나중에 비교할 수 있습니다.

## 10. 예측 파일의 컬럼 이름이 다를 때

팀원 도구인 `infer_val.py` 와 `eval_val.py` 는 `id, pred, prob_a~prob_d` 를 쓰고, `vqa` 패키지는
`id, p_a~p_d, run_id` 를 씁니다. `vqa.predictions.load_predictions` 가 두 이름을 모두 읽으므로,
`scripts/combine.py` 와 `scripts/evaluate_run.py` 에는 팀원 파일을 그대로 넘겨도 됩니다.
`run_id` 가 없는 파일은 `unknown` 으로 채워집니다.

파일 자체를 바꿔야 한다면 다음 명령을 쓰십시오.

```
python scripts/convert_preds.py --to teammate --in <vqa 형식 CSV> --out <팀원 형식 CSV>
python scripts/convert_preds.py --to vqa --in <팀원 형식 CSV> --out <vqa 형식 CSV> --run-id <run_id>
```
