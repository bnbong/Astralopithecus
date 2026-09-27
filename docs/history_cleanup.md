# 저장소를 공개로 전환하기 전에 수행할 이력 정리 절차

이 문서는 지금 비공개 상태인 이 저장소를 나중에 공개로 전환할 때, 그 전에 반드시 끝내야 하는
git 이력 정리 작업을 정리한 것입니다. 작업 순서와 명령, 그리고 팀원에게 알려야 하는 사항을
함께 적었습니다. 이 문서의 내용을 실행하면 커밋 해시가 전부 바뀌므로, 대회가 끝나고 공개
전환을 실제로 결정한 다음에 한 번에 수행하십시오.

## 1. 왜 이력 정리가 필요한가

현재 작업 트리에서는 대회 데이터에서 파생된 파일들을 추적에서 제외했고 일부 컬럼도
제거했지만, git은 과거 커밋의 내용을 그대로 보관합니다. 그래서 저장소를 공개로 전환하면
과거 커밋을 내려받는 방법으로 아래의 내용을 누구나 열람할 수 있습니다.

첫째, 대회 데이터의 질문 원문과 정답, 그리고 train 이미지에서 OCR로 읽어 낸 글자가 담긴
파일 세 개가 과거 커밋에 남아 있습니다. 이 파일들은 폴더 구조를 정리하기 전에
`ocr_files/` 아래에 있었기 때문에, 옛 경로와 새 경로 양쪽을 모두 지워야 합니다.

둘째, `artifacts/data/fold0_label_check_27b.csv`의 과거 버전에는 대회 데이터의 정답 글자가
담긴 `answer` 컬럼이 들어 있습니다. 현재 버전에서는 이 컬럼을 제거했습니다.

셋째, Colab 실행본 노트북 네 개의 과거 버전에는 압축 파일 비밀번호가
`ARCHIVE_PASSWORD = "..."` 형태의 평문으로 들어 있었습니다. 해당 문자열을 포함하는 커밋은
아래 일곱 개입니다.

| 커밋 | 날짜 | 커밋 메시지 | 문자열이 들어 있는 파일 |
|---|---|---|---|
| `24121de` | 2026-09-22 | 문서 정리 | `results/20_colab_27b_setup_preflight_20260921_A100-40GB_nf4.ipynb`, `results/21_colab_27b_run_20260921_A100-80GB_bf16_v1_image_only.ipynb`, `results/21_colab_27b_run_20260922_A100-80GB_bf16_v2_read_text.ipynb` |
| `520265a` | 2026-09-22 | 백업 및 학습 로직 수정 | `notebooks/21_colab_27b_run.ipynb`, `results/21_colab_27b_run_20260921_A100-80GB_bf16_v1_image_only.ipynb` |
| `3005c85` | 2026-09-22 | read_text_v2 실행 결과 | `notebooks/21_colab_27b_run_20260922_A100-80GB_bf16_v2_read_text.ipynb` |
| `5ba97d9` | 2026-09-21 | Colab을 통해 생성됨 | `notebooks/21_colab_27b_run.ipynb` |
| `3f8e1bb` | 2026-09-21 | bf16 전환 | `notebooks/20_colab_27b_setup_preflight.ipynb`, `results/20_colab_27b_setup_preflight_20260921_A100-40GB_nf4.ipynb` |
| `413d69e` | 2026-09-21 | Colab을 통해 생성됨 - 결과 | `notebooks/20_colab_27b_setup_preflight.ipynb` |
| `a1ae1ef` | 2026-09-27 | Colab을 통해 생성됨 | `results/21_colab_internvl_38b_run_full_20260926_A100-80GB_nf4_qlora_v2_read_text_personal.ipynb` |

이 목록은 아래 명령으로 다시 뽑을 수 있습니다. `<압축 비밀번호>` 자리에는 실제 비밀번호
문자열을 넣으십시오. 이 문서에는 비밀번호를 적지 않습니다.

```bash
git log -S "<압축 비밀번호>" --oneline --all
git log -S "<압축 비밀번호>" --all --name-only --pretty=format:"COMMIT %h %ad %s" --date=short
```

현재 작업 트리와 `HEAD` 시점의 파일에는 이 문자열이 남아 있지 않습니다. 따라서 정리해야
하는 대상은 과거 커밋뿐입니다.

## 2. 이력에서 지울 대상 목록

### 2.1 파일 경로

아래 세 파일은 옛 경로와 새 경로를 모두 지정해야 합니다. 폴더 구조를 정리하면서 파일을
옮겼기 때문에, 한쪽 경로만 지우면 다른 쪽 경로에 내용이 그대로 남습니다.

| 파일 | 옛 경로 | 새 경로 | 무엇이 들어 있는가 |
|---|---|---|---|
| OCR 캐시 | `ocr_files/ocr_cache.json` | `experiments/ocr/ocr_cache.json` | train 이미지에서 읽어 낸 OCR 텍스트 |
| 상품명·가격·번호 추출 결과 | `ocr_files/fold0_name_price_number.csv` | `experiments/ocr/fold0_name_price_number.csv` | train 질문 원문 |
| OCR 검증 결과 | `ocr_files/test_result/ocr_verification_result.csv` | `experiments/ocr/test_result/ocr_verification_result.csv` | train 정답과 OCR 텍스트 |

`artifacts/data/fold0_label_check_27b.csv`는 파일 자체를 지우지 않습니다. 이 파일은 현재
버전에서 정답 컬럼을 제거했으므로, 과거 버전만 문제가 됩니다. 이력 전체에서 이 파일을 지울
것인지, 아니면 정답 컬럼이 들어 있던 과거 버전만 치환할 것인지는 아래 4절의 결정 항목과
함께 판단하십시오. 파일 전체를 지우기로 결정한다면 경로 목록에 이 경로를 추가하면 됩니다.

### 2.2 문자열

압축 비밀번호 문자열을 이력 전체에서 치환해야 합니다. 이 문자열은 위 1절의 표에 적은 일곱
커밋의 노트북 파일 안에 평문으로 들어 있습니다. 노트북 파일 자체는 실행 기록으로 남겨야
하므로, 파일을 지우는 대신 문자열만 치환하는 방식을 사용합니다.

## 3. 정리 절차

### 3.1 git filter-repo 설치

`git filter-branch`는 느리고 오류가 생기기 쉬우므로 `git filter-repo`를 사용합니다. macOS와
리눅스에서는 아래 중 하나로 설치할 수 있습니다.

```bash
# Homebrew를 쓰는 경우
brew install git-filter-repo

# pip를 쓰는 경우
pip install git-filter-repo
```

설치가 끝났는지 `git filter-repo --version`으로 확인하십시오.

### 3.2 백업 클론 만들기

이력을 고쳐 쓰는 작업은 되돌릴 수 없습니다. 작업을 시작하기 전에 원격 저장소를 통째로
백업해 두십시오.

```bash
cd ~/backup
git clone --mirror git@github.com:bnbong/Astralopithecus.git Astralopithecus-backup.git
```

`--mirror`를 붙이면 모든 브랜치와 태그, 원격 추적 정보까지 함께 복제합니다. 백업 클론은
정리 작업이 끝나고 결과를 충분히 확인할 때까지 지우지 마십시오.

### 3.3 작업용 클론 준비

`git filter-repo`는 작업 트리에 커밋하지 않은 변경이 남아 있으면 실행을 거부합니다. 따라서
평소에 쓰던 작업 폴더가 아니라 새로 받은 클론에서 실행하십시오.

```bash
cd ~/work
git clone git@github.com:bnbong/Astralopithecus.git Astralopithecus-cleanup
cd Astralopithecus-cleanup
```

### 3.4 파일 경로 제거

`--path`로 지울 경로를 지정하고 `--invert-paths`를 붙이면, 지정한 경로만 이력에서
제거합니다.

```bash
git filter-repo \
  --path ocr_files/ocr_cache.json \
  --path ocr_files/fold0_name_price_number.csv \
  --path ocr_files/test_result/ocr_verification_result.csv \
  --path experiments/ocr/ocr_cache.json \
  --path experiments/ocr/fold0_name_price_number.csv \
  --path experiments/ocr/test_result/ocr_verification_result.csv \
  --invert-paths
```

명령이 끝나면 아래 명령으로 해당 경로가 이력에서 사라졌는지 확인하십시오. 출력이 비어
있어야 합니다.

```bash
git log --all --name-only --pretty=format: | sort -u | grep -Ei "ocr_cache|name_price|ocr_verification"
```

### 3.5 비밀번호 문자열 치환

치환 규칙을 담은 파일을 먼저 만듭니다. 이 파일은 저장소 바깥에 두십시오.

```bash
cat > ~/work/replace-rules.txt <<'RULES'
<압축 비밀번호>==>REMOVED_ARCHIVE_PASSWORD
RULES
```

`==>` 왼쪽에 실제 비밀번호 문자열을 적고, 오른쪽에는 치환할 문구를 적습니다. 그다음
`--replace-text`로 규칙 파일을 넘깁니다.

```bash
git filter-repo --replace-text ~/work/replace-rules.txt
```

치환이 끝나면 규칙 파일을 지우고, 이력에 문자열이 남아 있지 않은지 확인하십시오. 아래
명령의 출력이 비어 있어야 합니다.

```bash
git log -S "<압축 비밀번호>" --oneline --all
rm ~/work/replace-rules.txt
```

### 3.6 원격에 강제로 반영하기

`git filter-repo`는 안전을 위해 `origin` 설정을 지웁니다. 원격을 다시 등록한 다음에 모든
브랜치와 태그를 강제로 밀어 넣으십시오.

```bash
git remote add origin git@github.com:bnbong/Astralopithecus.git
git push --force --all origin
git push --force --tags origin
```

GitHub 저장소에 브랜치 보호 규칙이 걸려 있으면 강제 push가 거부됩니다. 규칙을 잠시 해제하고
작업한 다음에 다시 설정하십시오.

### 3.7 팀원에게 알릴 사항

이력을 고쳐 쓰면 모든 커밋 해시가 바뀝니다. 기존에 받아 둔 클론에서 `git pull`을 하면
이력이 뒤섞이므로, 팀원 전원에게 아래 내용을 알려야 합니다.

1. 정리 작업을 하는 동안에는 push를 하지 말아야 합니다.
2. 작업이 끝나면 기존 클론을 지우고 저장소를 새로 클론해야 합니다.
3. 기존 클론에 아직 올리지 않은 작업이 남아 있다면, 새로 클론한 저장소에 그 변경 내용만
   따로 옮겨야 합니다.

### 3.8 비밀번호 변경

GitHub는 강제 push 이후에도 pull request의 diff와 fork된 저장소에 옛 커밋을 얼마간
보관합니다. 그러므로 이력에서 문자열을 지웠더라도 그 비밀번호가 완전히 사라졌다고 볼 수
없습니다. **압축 파일의 비밀번호는 반드시 새 값으로 변경하십시오.** GitHub 지원팀에 연락해서
남아 있는 캐시를 지워 달라고 요청할 수도 있지만, 비밀번호 변경이 더 확실한 조치입니다.

## 4. 대회 종료 후에 결정할 항목

실행본 노트북의 출력 셀에는 fold 0 오답 점검 과정에서 출력한 train id와 정답 글자가 최대
61건 남아 있습니다. 이 내용을 어떻게 처리할지는 **대회가 끝난 다음에 결정합니다.** 결정하기
전에는 해당 노트북에 손대지 마십시오. 실행본 노트북은 제출물의 재현 근거이므로, 대회가
진행되는 동안 내용을 바꾸면 근거로서의 가치가 떨어지기 때문입니다.

선택지는 다음 두 가지입니다.

- **(a) 그대로 둔다.** 노출되는 정보가 대회 데이터의 이미지나 질문 본문이 아니라 id와 정답
  글자뿐이며 건수도 61건에 그치므로, 실질적인 위험이 낮다고 판단하는 선택입니다.
- **(b) 출력에서 정답 글자 컬럼을 지운 사본으로 교체한다.** 노트북의 코드 셀은 그대로 두고
  출력 셀에서 정답 글자 컬럼만 제거한 사본을 만들어서 교체하는 선택입니다. 이 경우에도 과거
  커밋에는 원본이 남으므로, 3절의 절차를 적용해서 이력까지 함께 정리해야 합니다.

## 5. 로컬에만 보관하는 데이터 파일

아래 파일들은 저장소에서 제외했으며 개인 PC와 Drive에만 보관합니다. 저장소를 새로 클론한
사람은 이 파일들을 직접 확보해야 코드를 실행할 수 있습니다.

| 대상 | 무엇인가 | 어디에 있는가 |
|---|---|---|
| `train/`, `test/` | 대회 원본 이미지 폴더입니다. | 대회 사이트에서 내려받거나 팀 공용 Drive에서 받습니다. |
| `train.csv`, `test.csv`, `sample_submission.csv` | 대회 원본 CSV입니다. | 위와 같습니다. |
| `artifacts/data/image_hashes.csv`, `artifacts/data/near_duplicate_pairs.csv`, `artifacts/data/split_leakage.csv`, `artifacts/data/audit_report.json`, `artifacts/data/audit_report.md` | 데이터 감사 단계에서 만든 산출물이며 이미지 해시와 질문 정보가 들어 있습니다. | `notebooks/01_data_audit_split.ipynb`를 다시 실행하면 같은 파일을 만들 수 있습니다. |
| `artifacts/demo/`, `artifacts/experiments.jsonl` | 파이프라인 점검용 가짜 예측 결과와 실험 기록입니다. | 데모 스크립트를 다시 실행하면 만들 수 있습니다. |
| `experiments/ocr/ocr_cache.json`, `experiments/ocr/fold0_name_price_number.csv`, `experiments/ocr/test_result/ocr_verification_result.csv` | OCR 실험 산출물이며 질문 원문과 정답, OCR 텍스트가 들어 있습니다. | 담당자 개인 PC에 보관하며, 필요하면 `experiments/ocr/`의 코드를 다시 실행해서 만듭니다. |

반면에 `fold_csv_files/dev_aux.csv`와 id 목록 CSV, split JSON은 저장소에 그대로 둡니다.
`dev_aux.csv`는 원본 라벨이 아니라 모델들이 투표한 결과이고, id 목록과 split 정보에는
이미지나 질문 본문이 들어 있지 않기 때문입니다. 팀원 전원이 동일한 split을 써야 하므로 이
파일들은 오히려 저장소에 있어야 합니다.

## 6. 체크리스트

공개 전환을 실행할 때 아래 순서대로 확인하십시오.

- [ ] 대회가 끝났고 공개 전환을 팀에서 합의했습니다.
- [ ] 4절의 실행본 노트북 출력 처리 방침을 (a)와 (b) 중에서 결정했습니다.
- [ ] 정리 작업이 끝날 때까지 push를 멈추라고 팀원 전원에게 알렸습니다.
- [ ] `git clone --mirror`로 백업 클론을 만들었습니다.
- [ ] 작업용 클론을 새로 받았습니다.
- [ ] `git filter-repo --invert-paths`로 2.1절의 여섯 경로를 제거했습니다.
- [ ] `git filter-repo --replace-text`로 압축 비밀번호 문자열을 치환했습니다.
- [ ] `git log --all --name-only`와 `git log -S`로 대상이 남아 있지 않은지 확인했습니다.
- [ ] 치환 규칙 파일을 지웠습니다.
- [ ] 브랜치 보호 규칙을 잠시 해제하고 `--force`로 push한 다음에 규칙을 되돌렸습니다.
- [ ] 압축 파일의 비밀번호를 새 값으로 변경했습니다.
- [ ] 팀원 전원이 기존 클론을 지우고 저장소를 새로 클론했습니다.
- [ ] 저장소를 공개로 전환하기 직전에 `.gitignore`와 `git ls-files` 목록을 한 번 더
      점검했습니다.
