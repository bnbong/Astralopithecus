# artifacts/data 안내

공식 분할은 `split_fold0.json` 하나입니다. 이 파일은 `scripts/folds_to_split.py`가
`fold_csv_files/val_fold0.csv`(검증 1,344문항)와 `fold_csv_files/train_fold1234.csv`(학습
5,358문항)를 읽어서 만듭니다. 검증 id 목록은 같은 스크립트가 함께 쓰는
`val_fold0_ids.txt`에 들어 있습니다.

전체 학습용 분할인 `split_full.json`도 같은 스크립트로 만들었습니다.
fold 1~4의 5,358문항과 fold 0의 1,344문항을 합친 다음,
`fold_csv_files/exclude_train_overlap_test.csv`에 적힌 문항 가운데 fold 0에 남아 있던 12문항을
빼서 6,690문항이 됩니다. 이 12문항은 test와 이미지가 완전히 같은 train 문항입니다. 같은
이름의 `split_full_v1_6702.json`은 이 12문항을 빼기 전의 6,702문항짜리 보존본이며, 그 분할로
이미 시작한 학습만 참조합니다.

`split_v1.json`과 `train_ids.txt`, `val_ids.txt`는 폐기했습니다. 이 세 파일은 검증 500문항을
쓰던 초기 분할이며, 팀이 fold 0을 공식 검증셋으로 정하기 전에 만든 것입니다. 파일 자체는
과거 측정값을 다시 확인할 때를 위해 남겨 두었지만, 새로 시작하는 학습과 평가에서는 쓰지
않습니다.

`audit_report.*`와 `image_hashes.csv`, `near_duplicate_pairs.csv`, `split_leakage.csv`는
데이터에서 파생된 산출물이라서 저장소에 추적하지 않습니다. 이 파일들은
`scripts/audit_data.py`가 만듭니다.

`train_folds.csv`는 train 6,714문항 전체의 fold 번호와 test 중복 여부를 담은 표입니다. 열은
`id`, `fold`(0~4), `overlap_test`(불리언) 세 개뿐이고 질문과 정답은 넣지 않았습니다. 팀원이
배포한 `extract_split_csvs.py`의 `build()` 로직을 그대로 재현해서 만들었습니다. train, dev,
test 이미지 파일 바이트의 MD5를 그룹 키로 삼고, answer를 층화 기준으로
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`를 적용한 결과입니다.
같은 로직으로 다시 만든 결과는 pandas 3.0.2와 scikit-learn
1.8.0, pandas 3.0.6과 scikit-learn 1.9.1 두 조합 모두에서 `fold_csv_files`의 다섯 파일과
내용이 같았습니다. 저장소의 원본 파일은 줄바꿈이 CRLF라서 줄바꿈만 맞추면 바이트까지
동일합니다. fold별 문항 수는 0번 1,344문항, 1번과 2번이 각각 1,343문항, 3번과 4번이 각각
1,342문항이고 `overlap_test`가 참인 문항은 24건입니다.
