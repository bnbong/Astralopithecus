# artifacts/data 안내

공식 분할은 `split_fold0.json` 하나입니다. 이 파일은 `scripts/folds_to_split.py`가
`fold_csv_files/val_fold0.csv`(검증 1,344문항)와 `fold_csv_files/train_fold1234.csv`(학습
5,358문항)를 읽어서 만듭니다. 검증 id 목록만 따로 필요하면 같은 스크립트가 함께 쓰는
`val_fold0_ids.txt`를 사용하시면 됩니다. 분할 방식은 `fold_csv_files/데이터처리_검증분할.md`에
정리되어 있습니다.

`split_v1.json`과 `train_ids.txt`, `val_ids.txt`는 폐기했습니다. 이 세 파일은 검증 500문항을
쓰던 초기 분할이며, 팀이 fold 0을 공식 검증셋으로 정하기 전에 만든 것입니다. 파일 자체는
과거 측정값을 다시 확인할 때를 위해 남겨 두었지만, 새로 시작하는 학습과 평가에서는 쓰지
않습니다.

`audit_report.*`와 `image_hashes.csv`, `near_duplicate_pairs.csv`, `split_leakage.csv`는
데이터에서 파생된 산출물이라서 저장소에 추적하지 않습니다. 필요하면
`scripts/audit_data.py`를 다시 실행해서 만드십시오.
