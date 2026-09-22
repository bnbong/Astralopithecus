# 텍스트 이미지 VQA — 5일 구현·개발 계획

이미지와 질문·선택지를 입력받아 a~d 중 정답을 고르는 모델을 개발한다. 목표는 5일 안에 정확도·재현성·효율성을 확보하고 제출을 완료하는 것이다.

팀은 5명이다. 실습실 PC는 각각 i7-14700·RAM 32GB·RTX 5060 Ti 16GB이며, Colab Pro+ 1계정을 함께 사용한다. 실습실에서도 Colab에 접속해 D1부터 병행한다. 외부·공공·직접 수집 데이터는 출처 링크를 기록해 사용한다. 모델 실행 설정과 소요 시간은 D1 실측으로 확정한다.

이 문서는 모델 구성, 데이터 처리, 역할과 일정을 정리한다. 담당자별 첫날 작업은 `vqa_D1_team_checklist.md`를 따른다.

## 1. 운영 전략

**Colab Qwen3.5-27B + 실습실 Qwen3.5-9B + answer-only LoRA + 확률 캐시 기반 결합**으로 진행한다.

- 27B: 80GB GPU에서는 BF16 LoRA, 40GB급에서는 NF4 QLoRA를 기본으로 한다.
- 실습실: 16GB GPU에서 9B NF4 QLoRA를 기본으로 한다.
- D1: 대회 규칙 확인, Colab 환경·커널·메모리·속도 검증을 끝낸다.
- 학습 전 27B base로 validation과 전체 test를 추론한다. 첫 유효 제출과 확률 캐시를 확보한다.
- 27B와 9B를 각각 1회 학습한다. 모델별 선행 조건이 충족되면 별도 GPU에서 병행한다. 전체 train 재학습과 하이퍼파라미터 탐색은 기본 생략한다.
- dev는 학생 응답과 9B base 예측이 일치한 후보를 검수한 뒤 제한적으로 활용한다.
- base와 미세조정 모델의 전체 예측 확률을 저장한다. 결합 후보는 CPU에서 비교한다.
- 보조 9B의 선택지 순열 TTA4는 공식 허용 확인·검증 이득·시간 여유가 있을 때만 적용한다.

실습실 작업도 GPU 가용 시간과 검수 시간을 기준으로 배정한다. 모델과 기법의 채택은 고정 validation 결과로 결정한다.

## 2. 장비별 모델과 정밀도

| 환경 | 기본 설정 | 처리 원칙 |
|---|---|---|
| Colab A100 80GB급 | 27B BF16 LoRA | 가장 긴 입력 batch로 역전파까지 확인 |
| Colab A100 40GB급 또는 유사 용량 | 27B NF4 4-bit QLoRA | batch1, checkpointing. precision·pixel·속도를 실측 |
| Colab 16~24GB급만 가용 | 9B 주력 전환 | 27B CPU offload 학습을 일정의 필수 경로로 만들지 않음 |
| 실습실 5대, 각각 16GB | 9B NF4 4-bit QLoRA | 같은 확정 모델을 배포해 데이터 분할 추론 |

양자화 호환 문제로 9B 실행이 실패하면 **4B BF16 LoRA**를 비양자화 fallback으로 점검하되, 가중치 약 8GB라는 계산만으로 16GB 학습 적합성을 보장하지 않는다.

80GB BF16에서도 OOM이 나면 먼저 메모리 누수·불필요한 logits·입력 길이를 확인한다. 8-bit 로딩을 한 번 점검하거나 pixel budget을 낮춘다. 40GB NF4에서 시간·메모리 예산을 통과하지 못하면 9B 경로로 전환한다. 여러 정밀도로 전체 학습을 반복하지 않는다.

주력 정밀도는 D1에 결정하고 base/FT 캐시에서 유지한다. 중간에 변경하면 별도 run으로 기록하며 서로 같은 조건의 before/after라고 보고하지 않는다. PC 5대의 VRAM은 하나의 GPU처럼 합쳐지지 않는다.

bitsandbytes 공식 설치표에는 CUDA 12.8 계열 등의 Linux·Windows sm120 대상이 있다. Blackwell을 무조건 미지원으로 간주하지 않는다. 실제 wheel·드라이버·PyTorch·연산 조합의 성공 여부를 확인한다.[S4]

## 3. D1 Colab 사전 점검과 커널 확인

D1 시작 시 D는 제출 규칙을 확인하고, 준혁은 Colab 자원·환경을 점검한다. 다른 팀원은 데이터와 로컬 코드를 병행 준비한다. 규칙에 의존하는 실행은 확인 후 진행한다. Colab 런타임 조작자는 한 명으로 유지한다.

최종 데이터 전체가 준비되기 전에도 설치·가중치 다운로드·소규모 입출력 확인은 가능하다. 이때의 임시 실행을 성능 평가로 사용하지 않는다. 본 학습용 20-step 점검은 분할 확정 후 train ID로 수행하고 임시 adapter는 폐기한다.

1. 실제 GPU·VRAM·CPU RAM·디스크·Drive 여유·잔여 CU·시간당 CU를 기록한다.
2. Qwen3.5를 지원하는 Transformers 5.x의 **검증된 버전 또는 commit**과 PEFT/torch/accelerate를 고정한다. 단순히 최신 또는 임의의 5.x를 설치했다고 통과시키지 않는다.
3. Gated DeltaNet과 causal convolution의 가속 경로를 확인한다. 고정 버전에 맞는 FLA·causal-conv1d 또는 동등한 kernel-hub 경로를 준비한다.
4. 설치 성공·import 성공과 실제 가속 커널 사용을 구분한다. forward와 backward, 경고·프로파일 또는 호출 경로, latency를 확인한다. FlashAttention 설정만으로 linear attention 가속을 확인했다고 판단하지 않는다.
5. 27B 원본 snapshot을 revision과 함께 지속 캐시한다. GPU 없이 가능한 다운로드는 미리 수행한다. Drive 용량을 확인하고 실제 실행 파일은 런타임 로컬 디스크에 staging한다. 직접 Drive에서 반복적인 작은 파일 읽기를 피한다.
6. 작은 글자·영수증·메뉴판 등 대표 20장을 processor 적용 후 확인한다. 긴 질문과 큰 시각 토큰을 포함한 batch에서 역전파를 검사한다.
7. 준비·JIT warm-up과 분리해 약 20 **optimizer step** 및 50문항 추론 시간을 측정한다. accumulation8이면 20 microbatch가 아니라 최대 160 microbatch 규모임을 기록한다.
8. preflight에서 갱신한 임시 adapter·optimizer는 버린다. base 캐시에는 adapter를 적용하지 않고, 본 학습은 seed를 재설정한 새 adapter로 시작한다.

Transformers에는 kernel-hub와 torch fallback 경로가 있다. **고정한 버전에서 가속 경로가 실제 쓰이고 처리량이 예산에 맞아야** 사전 점검을 통과한다.[S2][S3]

실습실에서도 bitsandbytes의 4-bit forward/backward를 확인한다. FLA/Triton 등은 Colab Linux와 Windows 설치 조건이 다를 수 있다. 이미 사용 가능한 허용 환경과 wheel을 우선하고, 대회 중 OS 재구축에 시간을 쓰지 않는다. 최적화 경로가 없으면 실측 처리량으로 작업을 줄이거나 배치를 바꾼다.

D1에 적합 Colab GPU가 없으면 준비 가능한 캐시·코드·로컬 점검을 끝낸다. 실제 GPU 적합성 확인이 미완료임을 기록하고 D2 초반에 즉시 완료하거나 9B 경로로 전환한다. 확보하지 못한 환경을 완료했다고 표시하지 않는다.

## 4. 모델·학습·출력 구조

27B와 9B 모두 이미지 encoder와 hybrid language backbone을 이용한다. 이미지·질문·선택지를 공식 chat template에 넣고 non-thinking 답변 위치를 맞춘다. `enable_thinking=False` 등 해당 버전이 지원하는 방식을 사용한다. `/nothink` 문자열만으로 처리하지 않는다.

| 항목 | 27B | 9B |
|---|---|---|
| 학습 정밀도 | 80GB BF16 / 40GB NF4 | NF4 |
| LoRA | r16, alpha32, dropout0.05 | 동일 |
| 학습률·epoch | 1e-4, 1 epoch | 동일 |
| microbatch·누적 | 1·8 | 동일 |
| scheduler | warmup3% + linear decay | 동일 |
| loss | answer-only causal LM | **동일한 구현** |
| 학습 대상 | 언어모델의 attention·linear attention·FFN adapter | 동일한 원칙 |
| 동결 | base·vision encoder | 동일 |
| 메모리 | gradient checkpointing, train use_cache=False | 동일 |
| train/infer max_pixels | 동일 1024²부터 확인 | 동일 768²부터 확인 |

27B는 1024²에서 작은 글자가 판독되지 않으면 1280²를 D1에 한 번 확인한다. 메모리·시간이 적합할 때만 양쪽에 같은 값으로 채택한다. 반대로 OOM이면 양쪽 모두 낮춘다. 9B도 768²가 맞지 않으면 양쪽 모두 512² 등으로 조정한다. 같은 모델 내 train/val/test processor 정책을 고정하며, 모델 간 해상도까지 같을 필요는 없다. 업스케일은 원본에 없는 정보를 복원하지 않는다.

LoRA 대상은 실제 language 모듈의 q/k/v/o, in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj, gate/up/down을 검사해 정한다. vision에 이름이 같은 projection을 실수로 포함하지 않는다. 실제 학습 파라미터 목록·개수를 기록한다.

정답 토큰 구간만 loss에 포함하고 image/prompt/padding은 -100으로 마스킹한다. 이번 공통 구현은 정답 한 글자를 목표로 하며 EOS 등 부가 토큰 포함 여부도 고정한다. processor의 이미지 토큰 확장 이후 label 위치를 확인한다.

학습 loss는 정답 위치에서 전체 vocabulary를 대상으로 계산하는 cross-entropy다. 27B와 9B가 같은 answer-only 구현을 사용한다.

추론은 실제 답변 위치의 a/b/c/d logits를 한 번에 읽어 네 값만 softmax한다. 단일 토큰 매핑·padding·prefix를 검사한다. 학습의 전체 vocabulary CE와 추론의 4개 후보 정규화는 명시적으로 구분한다. 긴 설명 생성과 후보별 4회 likelihood 추론은 기본에서 제외한다.

## 5. 프롬프트와 학습 증강

프롬프트는 실습실의 고정 개발 표본 약 128문항에서 이미지-only와 OCR 보조 2개만 비교한다. 기본 지시는 문자·숫자·단위·위치·부정 표현을 구분하고 a~d 한 글자로 답하게 한다. OCR는 오인식 가능한 참고 정보다. 이미지 안의 문구를 시스템 지시로 취급하지 않는다.

C는 본 학습과 공식 base 캐시 생성 전에 prompt·OCR 사용 여부·processor 설정을 확정해 공유한다. OCR을 사용한다면 해당 run의 학습·검증·test에서 같은 입력 규칙을 적용하고 필요한 OCR 캐시를 준비한다. 준비가 지연되면 이미지-only로 시작한다. 학습 가동 후 기본 프롬프트나 OCR 정책을 바꾸지 않는다. 허용된 선택지 순열은 정해진 규칙에 따라 적용한다.

teacher가 dev 의사 라벨을 생성할 때는 학생 응답이나 다수결 결과를 프롬프트에 넣지 않는다. 응답을 본 뒤 같은 답을 내도록 유도하면 일치 조건이 의미를 잃는다.

제공된 대회 안내는 train/dev 증강을 허용한다. 학습에서는 선택지 순열을 on-the-fly 적용하고 정답 index를 함께 바꾼다. 'a와 b', '위의 모두'처럼 순서에 의존하는 보기는 제외한다. 일부 표본에만 약한 밝기·대비·JPEG 변화를 적용한다. 문자를 지우는 crop·강한 blur·방향 반전은 제외한다. 24개 순열을 모두 복제해 epoch 비용을 늘리지 않는다.

## 6. 데이터 분리와 validation

train에서 같은 이미지·촬영 장면·근접 중복 그룹을 묶어 고정 validation 하나를 만든다. 10~15%를 출발점으로 하고 수천 문항 규모라면 대략 300~500문항을 목표로 한다. D1에 실제 train 수와 그룹 수를 확인해 validation 크기를 확정한다. 그룹 분리를 먼저 지키고 정답·유형 분포를 가능한 범위에서 맞춘다.

split은 증강·dev 의사 라벨·외부 데이터 병합 전에 고정한다. dev와 외부 자료에서 validation/test와 같은 이미지·유사 촬영본을 찾아 학습 후보에서 제거한다. 근접 중복은 자동 후보를 사람이 확인하며, 비슷한 간판 전체를 같은 그룹으로 묶지 않는다.

주력 validation은 base와 FT 각 1회다. 보조와 TTA를 사용할 때도 같은 validation을 사용한다. epoch마다 전체 평가나 k-fold·다중 seed·대형 grid를 추가하지 않는다. 저장된 validation 확률로 제한된 결합 후보를 CPU에서 비교한다. CPU 계산이 싸더라도 수많은 가중치·규칙 탐색으로 validation에 과적합하지 않는다.

## 7. dev 데이터와 의사 라벨

분석 대상 `dev.csv`의 SHA256: `ffc92bc4dfccaad566547128a864e63090f50f4f0b208ae4d2474bc760fb269a`.

| 항목 | 실측 |
|---|---|
| 크기·실제 응답 컬럼 | 2,683행·12컬럼, answer1~answer5 |
| 응답 값 | 빈칸 또는 소문자 a/b/c/d. 자유 텍스트 없음 |
| 유효 응답 5개/4개/3개/0개 | 2,287 / 364 / 30 / 2행 |
| 같은 답 최다 3표/2표/1표/0표 | 2,026 / 649 / 6 / 2행 |
| 4표·5표 일치 | 0행 |
| 최다 응답 동률 | 465행 |
| 빈 응답 | 434칸, 396행에 존재 |

응답 예: dev_0001.jpg=`d,d,빈칸,d,a`; dev_0002.jpg=`b,b,d,c,d`; dev_0005.jpg=`b,b,b,c,c`. 질문·선택지는 별도 컬럼이다. 실제 응답 컬럼명은 **answer1~answer5**다.

4표 이상 일치하는 행은 없다. 원인은 CSV만으로 확인할 수 없다. 학생 응답은 정답 라벨이 아니므로 모델 예측과 이미지 검수를 함께 사용한다.

의사 라벨 절차:

1. 필요 스키마와 split을 고정하고 dev 이미지·중복을 검사한다.
2. 미세조정 전 9B base로 dev를 추론한다. 가용 PC 최대 4대에 ID를 분할하고 동일 revision·prompt·processor로 실행한다. OCR 등 다른 작업과 GPU 슬롯을 겹치지 않게 배정한다.
3. 유효 응답 중 같은 선택지가 3표 이상이고, teacher argmax와 같은 행을 후보로 한다. 모델 일치 전 최대 후보는 **2,026행**이다. 최종 사용 수는 teacher 추론·중복 제거·이미지 검수 후 확정한다.
4. 9B 확률·margin·응답 개수·표수를 기록한다. 모델 confidence는 보정된 정확도가 아니므로 일치만으로 정답이라고 단정하지 않는다.
5. 후보 최대 100개를 검수한다. 무작위 80개와 낮은 margin·어려운 숫자/위치 등 20개를 포함하고, 이미지가 모호하거나 틀린 행을 분리한다. 후보가 적으면 전부 본다.
6. 시작 운영 기준은 표본의 95% 이상이 이미지로 명확히 뒷받침되는 경우다. 이는 전체 집합의 95% 정확도를 보장하는 통계적 주장이 아니다. 기준 미달·검수 불가·준비 지연 시 의사 라벨은 제외한다.
7. 채택 시 공식 정답과 분리해 label_source/teacher_revision/검수 정보를 저장한다. 초기 loss weight는 공식1.0, 의사 라벨0.25로 둔다. 전체 학습에서 의사 라벨 노출은 20% 이하로 제한한다.
8. D1 종료~D2 본 학습 시작 전에 manifest를 동결한다. teacher 추론을 기다리며 27B 학습을 미루지 않는다. 준비가 안 되면 official-only로 예정대로 시작한다.

가중치 0.25·노출 20%는 노이즈 영향을 제한하기 위한 초기 설정이다. dev는 정답 기반 validation으로 사용하지 않는다. 의사 라벨의 효과는 별도 비교한 경우에만 성능 기여로 보고한다.

## 8. base 전체 test와 최종 결합

27B 실행 순서: **입력 설정 확정·preflight → base validation + 전체 test → 캐시 검사 → 새 adapter로 본 학습 → FT validation + 전체 test**.

base 전체 캐시를 확보하면 D가 첫 제출을 준비한다. 제출 준비와 학습은 병행하며, Kaggle 점수 회신을 기다리느라 GPU를 대기시키지 않는다. 이 순서는 조기 제출본 확보를 위한 운영 방식이다. 같은 Colab GPU의 27B base 추론과 학습은 차례로 실행한다. 로컬 9B 학습은 독립적인 준비가 끝나면 먼저 시작할 수 있다.

- base/FT 각각 독립적인 submission을 보존한다. zero-shot 제출 가능 여부는 D1 규칙 확인 결과를 따른다.
- base cache와 FT cache는 model revision·정밀도·prompt·pixel budget·이미지/CSV hash를 기록한다.
- 저장한 전체 확률을 재사용해 결합한다. 결합 가중치 비교에는 추가 GPU 추론이 필요 없다.
- test는 정답이 없으므로 결합 가중치를 고르는 데이터로 쓰지 않는다. 같은 ID의 validation 캐시를 사용한다.

결합 후보는 사전에 아래 네 개로 제한한다: base27 단독, FT27 단독, FT27 0.8+base27 0.2, FT27 0.8+FT9 0.2. local-only 제출은 자원 실패 시 대체본으로 보유한다. 동률·극소 차이는 단순한 후보와 비용을 우선한다. final answer와 margin은 같은 최종 확률에서 계산한다.

2N 추론은 기본 예산에 먼저 반영한다. base 전체 추론이 학습+최종 FT 추론 시간을 위협한다면, 시작 전에 9B 전체 base를 조기 제출로 사용하고 27B는 validation+FT 전체 추론으로 줄이는 예산 예외를 기록한다. 중간에 일부 base 결과를 전체 안전망으로 오인하지 않는다.

base 모델에서 test를 추론해도 test를 학습에 넣지 않는다. test 예측으로 pseudo-label 학습을 하지 않는다. FT가 base보다 나빠지면 무조건 FT를 제출하지 않되, 대회가 학습 모델 예측을 요구하면 해당 규칙을 지킨다.

## 9. 보조 9B의 선택지 TTA4

선택지 TTA4는 **운영진의 허용을 확인한 경우에만** 적용한다. 원본 이미지를 유지하고 a~d 선택지 순서만 바꾸는 추론 방식임을 명확히 질문한다. 허용 범위와 답변을 규칙표에 기록한다.

허용되면 로컬 9B에서 abcd/bcda/cdab/dabc 네 순서를 사용한다. 각 결과의 선택지 확률을 원래 a~d 순서로 복원한 뒤 평균한다. 다수결 답과 평균 확률 margin을 섞지 않는다. 순서 의존 보기와 remap되지 않은 OCR/검색 문맥의 선택지 문자는 별도 처리한다.

먼저 고정 validation에서 single-view 대비 확인한다. 이득과 실습실 시간이 있을 때만 전체 test에 적용한다. single-view 캐시가 있으면 추가 비용은 3N forward다. 5대는 벽시계 시간을 나눌 뿐 총 계산량을 없애지 않는다. 27B TTA4·multi-crop·다중해상도 test 결합은 이번 기본 예산에 넣지 않는다.

## 10. 계산 예산과 실행 시간

공식 Colab 요금 페이지에서 Pro+는 현재 월 600 CU로 안내된다.[S1] 지급 계정의 실제 잔여 CU·만료일·GPU별 시간당 소모는 별도 확인한다. 아래 7.5 CU/h와 5.4 CU/h는 계산용 가정이다. 실제 계획에는 계정 화면의 소모율을 사용한다.

| 가정: 잔여 600 CU | 단순 환산 | 20% 예비 CU 제외 |
|---|---|---|
| 7.5 CU/h | 80 GPU시간 | 64 GPU시간 |
| 5.4 CU/h | 약 111.1 GPU시간 | 약 88.9 GPU시간 |

실행 가능 시간은 CU 환산 시간과 실제 GPU 이용 가능 시간 중 작은 값으로 잡는다. 접근 시간·세션 제한·마감을 함께 고려한다. Pro+의 연속 실행 최대 24시간은 충분한 CU 등의 조건이 있다.[S8]

N=5,000을 단순 예시로 하면, 문항당1초는 한 pass 약1.39시간, 2초는 약2.78시간이다. base+FT 두 pass는 약2.78~5.56시간이며 학습·validation·다운로드·warm-up·I/O는 별도다. 이 예시는 실제 test 수량이나 27B 속도의 측정값이 아니다.

D1에는 cold start/JIT와 warm 실행을 나누고 50문항 평균·느린 구간·peak VRAM을 기록한다. 대표 이미지 크기와 질문 길이를 섞는다. 예상 필요 CU는 실제 소모율×(환경 준비+base test+학습+validation+FT test+복구 여유)로 계산한다. dev와 외부 자료를 추가한 실제 학습 step 수를 사용한다.

잔여 CU의 20%를 복구·최종 추론 예비분으로 두고 계획한다. 마감 여유는 최소4시간과 실제 재추론·패키징 요구 시간 중 큰 값으로 정한다. 27B를 로드한 상태에서 사람이 데이터 준비를 마칠 때까지 기다리지 않는다. 시간이 부족하면 외부 데이터·dev 추가·보조 TTA를 먼저 생략한다.

## 11. 분업·인계 조건과 5일 일정

분업은 담당 영역과 결과물의 인계 조건을 함께 정한다. 데이터 정리, 실행 환경 준비, 학습 코드, 입력 설계, 평가 코드는 동시에 준비할 수 있다. 본 학습은 데이터·입력·코드가 합쳐진 뒤 시작한다. 의존 작업이 끝나기 전에는 공통 샘플로 기능을 확인하고, 정식 정확도 비교는 고정 validation에서만 수행한다.

### 담당과 인계 결과물

| 담당 | 처음부터 할 수 있는 일 | 다음 작업에 넘길 결과물 | 받는 사람 |
|---|---|---|---|
| 준혁 | Colab 자원·환경·가중치 준비, 실행 순서 관리 | 환경 lock, 27B 실행 설정·실측 예산, base/FT 캐시 | B·D |
| A | CSV·이미지 검사, 그룹 분리, 출처 정리 | train/val ID 목록, 데이터 manifest, 경로·중복 검사 결과 | 전원 |
| B | 9B 로드, 공통 collator·학습·추론 코드, 저장·재개 구현 | 공통 Python 코드, 실행 명령, checkpoint 재개 결과 | 준혁·C·D |
| C | 대표 이미지 판독, 프롬프트·OCR 후보 준비 | prompt, OCR 사용 여부, 모델별 processor 설정 | 준혁·B |
| D | 대회 규칙 정리, 가짜 확률로 평가·병합·CSV 검사 구현 | 예측 파일 형식, 평가·제출 검사기, 실험 기록 | 전원 |

B가 두 모델에 쓰는 공통 코드를 관리한다. 준혁이 Colab 27B를 실행하고, B가 로컬 9B를 실행한다. A는 학습 데이터, C는 입력 설정, D는 결과 검사를 맡는다. 5명이 각자 모델 학습 코드를 만들지 않는다.

### 공통 형식과 시작 조건

첫 30분을 목표로 CSV 입력과 확률 출력 형식을 합의한다. 시간은 목표이며, 조건을 충족하지 못하면 다음 단계로 넘기지 않는다.

- 공통 입력: id, path, question, a~d. train만 정답 answer를 가진다. dev 응답은 answer1~answer5로 별도 보존한다.
- 분할: 원본 ID를 담은 train_ids와 val_ids. 행 순서에 의존하지 않는다.
- 예측: id, p_a~p_d, run_id. 모델·입력 설정과 파일 hash는 run manifest로 연결한다.
- 인계: 결과물 위치, 버전/hash, 통과한 검사, 다음 실행 명령을 남긴다.

| 단계 | 완료 조건 | 시작할 수 있는 작업 |
|---|---|---|
| G0: 형식 합의 | 샘플 행·이미지, 입력/출력 스키마, 담당 확정 | 각자 코드 작성, 임시 입출력 확인 |
| G1: 데이터 확정 | 필수 데이터의 ID·경로·이미지 검사, 그룹 분할·val ID 고정 | 정식 validation 비교, train 표본으로 학습 점검 |
| G2: 입력·환경 확정 | prompt/OCR/해상도, 커널·메모리·속도, 정답 mask·scoring 확인 | 27B base 공식 캐시, 확정 teacher 설정의 dev 추론 |
| G3: 학습 준비 완료 | 학습 manifest 동결, 증강·설정 확정, checkpoint 저장·재개 확인, 실행·중단 시간 확보 | 새 adapter로 9B 학습. 27B는 base 캐시 확보 후 학습 |

G2는 모델별로 통과한다. 9B 입력과 환경이 준비되면 27B 점검 완료를 기다리지 않고 teacher 추론을 시작할 수 있다. 27B base 추론이 진행되는 동안 A가 의사 라벨 검수를 마무리할 수 있다. G3 시점까지 준비되지 않은 dev·외부 데이터는 제외한다. 학습 중 데이터를 추가하지 않는다. 두 모델이 같은 학습 manifest를 사용하도록 먼저 확정하고, 9B는 27B base 추론 완료를 기다리지 않는다.

### D1 시간대별 작업

아래 시간은 시작 시각 기준의 목표다. 다운로드·검사 시간이 길어지면 단계별 완료 조건을 기준으로 이동한다.

| 시간대 | 준혁 | A | B | C | D |
|---|---|---|---|---|---|
| 0~1h | Colab 자원·가중치·환경 준비 | 공통 샘플 공유, CSV·이미지 검사·split | 9B 로드, 학습 코드 뼈대 | 대표 20장·입력 후보 검토 | 규칙 확인, 평가·병합·제출 검사기 |
| 1~4h | 분할·입력 준비 후 27B 커널·20 step·50문항 실측 | split 공유, 누수·출처 검사 | 9B 역전파·mask·저장/재개 확인 | split 이후 로컬 9B로 프롬프트 2개·OCR 비교 | 실제 소량 확률로 검사기 연결, 시간 예산 계산 |
| 약 4h 점검 | G1/G2와 27B 실행 가능성 판단 | 필수 데이터 준비 보고 | 공통 코드·9B 준비 보고 | prompt/OCR/해상도 동결 | 남은 시간·CU·미완료 조건 정리 |
| 이후 | G2 통과 후 base val·전체 test | 준비된 dev 후보 검수, G3 manifest 동결 | teacher 작업 후 G3 통과 시 9B 학습 | 필요 OCR 캐시 완료, 검수 지원 | base 결과 검사·허용 시 제출 준비 |
| 퇴실 전 | base 완료·G3 통과 시 27B 학습, 아니면 다음 실행 계획 공유 | 데이터 버전·출처 인계 | 완결 checkpoint·재개 명령 확인 | 확정 입력 설정 인계 | 진행 상태·중단 시각·다음 담당 기록 |

3B 베이스라인은 이미 실행 가능한 경우 소규모 입출력 확인에만 선택적으로 사용한다. 새로운 모델 설치나 전체 validation 반복을 필수 작업으로 넣지 않는다. 프롬프트 후보는 실제 보조 9B에서 좁히고, 주력 기준 정확도와 처리 시간은 27B base에서 확보한다.

주력 모델 판단은 27B가 실제 Colab 메모리·시간·CU 예산을 통과하는지로 한다. 통과하면 27B를 유지한다. 실패하면 제2절의 대체 경로를 적용한다. 로컬 9B의 실측만으로 Colab 27B 사용 여부를 결정하지 않는다.

### GPU 배정과 야간 실행

Colab은 27B 점검 → base 추론 → 학습 → FT 추론 순서로 사용한다. 로컬 B PC는 9B 점검·공통 코드 확인에, C PC는 입력 비교에 우선 배정한다. 남는 PC만 teacher 추론에 투입한다. B·C 작업이 끝나면 teacher용 GPU를 최대 4대까지 늘릴 수 있다. 각 PC는 한 번에 한 개의 대형 GPU 작업만 실행한다.

9B 본 학습은 1대에서 1회 수행한다. 이후 완성된 adapter를 필요한 PC에 배포해 test ID를 분할 추론한다. 5대의 GPU를 한 학습 작업의 VRAM처럼 합치거나, 같은 학습을 5번 반복하지 않는다. GPU를 사용하지 않는 팀원은 데이터·검수·평가·문서를 진행한다.

야간 실행은 시작 전에 실습실 퇴실·전원 정책, 사용 가능한 환경, 중단 시각을 확인한다. 지속 실행이 불가능하면 optimizer step 경계에서 저장하고 다음 환경에서 재개한다. Colab과 실습실의 종료 조건을 각각 관리한다. Colab 조작자는 준혁 한 명이며, 다른 팀원은 저장된 로그를 확인한다.

### 5일 일정과 완료 기준

| 날짜 | 작업 | 완료 기준 |
|---|---|---|
| D1 | G0~G2, base 추론 시작, G3 준비. 조건을 충족한 모델부터 학습 | split·입력·환경·예산 확정, base 진행 상태, 학습 manifest, 다음 실행 명령 |
| D2 | base 완료·제출, 미시작 본 학습 실행, 27B/9B 각 1회 학습·FT validation | base 유효 제출, 학습 checkpoint, validation 확률 |
| D3 | 27B FT 전체 test, 로컬 9B 분할 추론, 조건부 TTA4, CPU 결합 | FT/보조 캐시, 최종 후보 선정 |
| D4 | 실패·미완료 추론 복구, 최종 검사·제출, 재현·발표 준비 | 최종 유효 제출과 대체본, 설정·모델 보존 |
| D5 | 접수·최종본 지정 확인, 재현 smoke test, 개별 발표 | 마감 전 제출 완료 확인 |

D1 학습 가동은 준비 조건과 base 추론 시간이 맞을 때의 목표다. 기본 계획에서는 D2 시작도 허용한다. 학습 시작 시각을 맞추기 위해 split이나 입력 설정을 미완성 상태로 두지 않는다. base 전체 추론이 학습·최종 추론 예산을 위협하면 제8절의 9B 조기 제출 경로를 실행 전에 결정한다.

날짜별 환경을 배타적으로 묶지 않는다. Colab은 적합 GPU와 준비된 작업이 있을 때 사용한다. 잘 동작하는 세션은 후속 작업으로 이어 쓰고, 할 일이 없는 GPU의 idle 비용은 피한다.

## 12. 저장·인계·재현

DATA_ROOT 아래 train/dev/test 이미지 구조를 유지한다. CSV 상대 path를 그대로 조합하고 .jpg가 포함된 ID도 원문 보존한다. MODEL_ROOT/RUN_ROOT만 환경별 설정으로 분리한다.

코드·환경 lock·모델 snapshot revision·processor·prompt·pixel budget·split·source manifest·학습 recipe를 함께 기록한다. 27B와9B adapter는 교환하지 않는다. 장비별 실제 커널 경로·정밀도 변경도 run에 남긴다.

학습 checkpoint는 adapter·processor·base revision·optimizer·scheduler·scaler·RNG·epoch/step·sampler 진행을 포함한다. optimizer update 경계에서 저장하고 COMPLETE/hash 확인 후 latest를 갱신한다. best·latest를 분리하고 최근 완결본을 보존한다.

추론은 ID별 partial을 주기적으로 원자 저장한다. 5대는 서로 다른 shard 파일에 쓰고, 고정 manifest 행번호 mod5 등 안정적인 방법으로 분할한다. 각 ID가 정확히 한 번인지 검사한다. TTA에는 permutation과 역매핑 정보를 추가한다.

캐시 key에는 ID뿐 아니라 모델/adapter/precision/prompt/processor/pixel budget/image/question/options/OCR hash를 포함한다. base와 FT 캐시는 분리한다. GPU 종류가 달라지는 재개에서는 동일성 범위와 수치 차이를 기록한다.

환경 전환 전 완결 저장·동기화·checksum·다음 명령을 남긴다. 새 환경에서는 해당 모델의 소규모 재로딩/추론을 확인한다. 실습실에서27B를 로드할 필요는 없다. notebook 공유와 실행 중인 VM 공유는 다르므로 진행 상황은 저장 로그로 확인한다.

## 13. 외부 데이터·제출·구현 산출물

source registry에는 실제 사용 자료의 원본 URL·항목 ID·제공자·버전·이용 조건·파일 hash·QA 변환·검수자·사용 건수를 기록한다. 직접 수집 자료도 주최 측이 접근할 수 있는 자료 링크를 준비한다. dev 의사 라벨은 대회 원본 출처, 원본 ID, teacher와 생성 조건을 연결한다. 출처 없는 외부 행은 학습에 넣지 않는다.

외부 수집은 D2 학습 시작을 미루지 않는 검수된 소량만 사용한다. 시험 데이터나 validation의 라벨을 학습에 넣지 않는다. 마지막 날 새로운 모델·데이터 전략을 추가하지 않는다.

필수 모듈: preflight, data/split, collator/train, checkpoint, OCR, teacher_dev/pseudo_filter, predict(shard/resume/TTA), evaluate/combine, submit. 모듈은 공통 Python 코드로 두고 실습실/Colab notebook은 얇은 진입점으로 구성한다.

중요 검사는 선택지 label 재매핑, 정답 mask·padding, image-token 확장, group 누수, teacher prompt에서 학생 응답 제외, shard/TTA 역매핑, 캐시 fingerprint, 제출 ID 일대일 대응이다. 매 PR마다 전체 GPU 테스트를 돌리지 않는다.

최종 제출 CSV는 sample_submission과 ID 집합·행수·순서를 맞추고 answer∈a~d, 결측·중복·추가 index가 없는지 검사한다. 파싱 실패를 a로 채우지 않는다. 파일 hash·run ID·접수 상태·최종 선택을 기록한다. 일일 제출 제한과 최종 선택 개수는 D1에 확인한 공식 규칙을 따른다.

최종 산출물: submission 후보, 코드·notebook, base/adapter 전달 방법, dependency lock, config, split, 데이터·모델 출처, 실행·오류 분석 기록, 재현 README, 발표 자료. Discussion과 발표는 실제 확인한 결과만 근거로 작성한다.

## 참고 자료

- 대회 안내와 장비 사양. 제공된 baseline 노트북 3개와 27B 노트북 2개.
- `dev.csv`: 2,683행. train/test 수량과 이미지 상태는 D1 데이터 검사에서 확인한다.
- 참고 저장소: [27B 저장소](https://github.com/kimkihyun1/ssafy-ai-challenge-vqa/tree/2abe09e316ba994ffc5842cc19d209f5be6245d3), [15기2회차 저장소](https://github.com/MinjuJangg/ai_chall/tree/38b6f380648cc2c4fdca306f0094da31e993bf7c).
- [S1 Colab 공식 요금](https://colab.research.google.com/signup): 월600 CU 확인. 지급 계정의 실제 잔여량과 소모율은 별도.
- [S2 Transformers Qwen3.5 구현](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py): kernel-hub와 fallback 경로. 실제 사용할 commit 고정 필요.
- [S3 FLA](https://github.com/fla-org/flash-linear-attention), [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d).
- [S4 bitsandbytes 설치표](https://huggingface.co/docs/bitsandbytes/main/en/installation): CUDA/OS별 sm120 대상.
- [S5 Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), [S6 Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
- [S7 PEFT checkpoint](https://huggingface.co/docs/peft/developer_guides/checkpoint).
- [S8 Colab FAQ](https://research.google.com/colaboratory/faq.html): 자원·세션 한도와 Drive I/O.
