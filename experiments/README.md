# 실험 기록

최종 파이프라인에 들어가지 않은 팀원들의 실험 코드를 모아 둔 폴더입니다. 어떤 시도를 했고 왜
채택하지 않았는지를 남기려고 보관하며, 최종 제출물을 재현할 때에는 이 폴더의 코드를 실행하지
않습니다. 최종 방법과 판단 근거는 [`docs/submission_report.md`](../docs/submission_report.md)에 있습니다.

| 폴더 | 무엇을 했는지 | 담당 | 결과 |
|---|---|---|---|
| [`ocr/`](ocr/) | 외부 OCR로 읽은 글자를 프롬프트에 덧붙이는 방식을 비교하고, PaddleOCR과 EasyOCR의 인식 품질을 검증했습니다. | 김창혁 | OCR을 붙여도 정확도가 오르지 않아서 기각했으며, 최종 모델에 쓰지 않았습니다. |
| [`augmentation/`](augmentation/) | 이미지 증강 함수를 작성하고 노트북에서 동작을 확인했습니다. | 팀 공용 | 학습 경로에 연결하지 않았고, 최종 모델에 쓰지 않았습니다. |
| [`eval_tools/`](eval_tools/) | 3B와 9B 모델을 추론하고 채점하는 도구를 만들어서 초기 후보를 비교했습니다. | 유수정 | 초기 비교에만 썼으며, 최종 파이프라인은 `vqa` 패키지와 `scripts/`를 씁니다. |

`ocr/` 폴더가 만들어 낸 산출물 가운데 `ocr/ocr_cache.json`, `ocr/fold0_name_price_number.csv`,
`ocr/test_result/ocr_verification_result.csv` 세 파일은 대회 데이터의 질문 원문과 정답, 그리고
OCR이 읽어 낸 글자를 담고 있습니다. 이 세 파일은 저장소에는 포함하지 않으며 팀 내부에서만
보관하므로, `ocr/` 폴더의 코드를 다시 실행하려면 대회 데이터를 내려받아 이 파일들을 새로
만들어야 합니다.

`eval_tools/check_submission.py`는 유수정이 작성한 초기 검사기입니다. 최종 파이프라인이 쓰는
검사기는 이준혁이 작성한 `scripts/check_submission.py`이며, 두 파일은 이름만 같고 서로 다른
구현입니다. 확률 컬럼 이름도 달라서 `eval_tools/`의 도구는 `prob_a`부터 `prob_d`까지를 쓰고
`vqa` 패키지와 `scripts/`는 `p_a`부터 `p_d`까지를 씁니다. 두 형식 사이의 변환은
`scripts/convert_preds.py`가 담당합니다.
