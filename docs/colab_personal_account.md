# 개인 Colab Pro 계정에서 21번 노트북 돌리기

지급받은 Colab Pro+ 계정이 아니라 개인 Colab Pro 계정에서 `notebooks/21_colab_27b_run.ipynb`를
처음 실행할 때 필요한 준비물과 절차를 정리했습니다. 목표는 학습이 아니라 zero-shot 추론 비교
실험입니다. fold 0의 1,344문항을 원본 27B로 두 번 추론해서, 기본 해상도와 확대본의 차이를
봅니다.

개인 계정에서는 80GB A100이 거의 배정되지 않습니다. 40GB A100이나 24GB L4를 받게 되므로
정밀도는 nf4로 내려갑니다. 21번 노트북은 이 상황을 스스로 판정해서 학습을 막고 추론만
허용하므로, 아래 준비물만 갖추면 별도로 코드를 고칠 일은 없습니다.

## 1. Drive에 올릴 파일

| 파일 | 기본 경로 | 크기 | 설명 |
|---|---|---|---|
| 데이터 압축 파일 | `MyDrive/ssafy-16-2-ai.zip` | 약 2GB | 암호가 걸린 대회 데이터입니다. |
| 코드 압축 파일 | 임의 경로 | 수 MB | `GITHUB_TOKEN`을 등록하지 않을 때에만 필요합니다. |

산출물이 쌓이는 폴더는 노트북이 직접 만듭니다. `MyDrive/vqa/runs` 아래에 예측 파일과 로그가
남고, 가중치 캐시를 Drive에 두기로 했다면 `MyDrive/vqa/hf_cache` 아래에 약 54GB가 쌓입니다.

## 2. Colab 보안 비밀

화면 왼쪽의 열쇠 아이콘 패널에서 등록하고, 현재 노트북의 접근을 허용해야 합니다.

| 이름 | 필수 여부 | 용도 |
|---|---|---|
| `GITHUB_TOKEN` | 사실상 필수 | 비공개 저장소를 복제합니다. 이 저장소에만 Contents 읽기 권한을 준 fine-grained personal access token을 등록하십시오. 등록하지 않으려면 코드 zip을 Drive에 올리고 `CODE_ZIP`에 경로를 적으면 됩니다. |
| `HF_TOKEN` | 선택 | Hugging Face 읽기 권한 토큰입니다. Qwen3.5는 승인이 필요한 모델이 아니므로 없어도 받을 수 있지만, 익명 요청은 속도 제한에 걸릴 수 있습니다. |

압축 파일의 비밀번호는 비밀로 등록하지 않습니다. 파라미터 셀의 `ARCHIVE_PASSWORD`를 비워 두면
6번 절에서 `getpass`로 입력받고, 입력한 값은 화면에 남지 않습니다. 암호가 걸려 있지 않은
압축 파일이라면 `ARCHIVE_PASSWORD`에 `none`이라고 적으십시오.

## 3. 용량 판단

nf4로 돌리더라도 내려받는 파일은 원본 bf16 스냅샷입니다. 양자화는 스냅샷을 읽어서 GPU에
올리는 시점에 수행하므로, 캐시에는 언제나 약 54GB가 필요합니다. 개인 Drive의 무료 용량은
15GB이므로, 요금제를 올리지 않았다면 Drive 캐시는 쓸 수 없습니다.

| 상황 | `MODEL_CACHE` 값 | 결과 |
|---|---|---|
| Drive 여유가 60GB 이상입니다. | `"auto"` 또는 `"drive"` | `MyDrive/vqa/hf_cache`에 받습니다. 런타임이 바뀌어도 다시 받지 않습니다. |
| Drive 여유가 부족합니다. | `"local"` | 가상 머신의 `/content/hf_cache`에 받습니다. 런타임이 회수되면 사라지므로 접속할 때마다 다시 받아야 합니다. |

`"auto"`로 두어도 Drive 여유가 `MODEL_CACHE_NEED_GB`(기본값 60GB)에 못 미치면 노트북이 로컬
디스크로 자동으로 내려갑니다. 다만 이 판정은 5번 절에서 한 번만 이루어지므로, Drive 여유가
애매하게 걸쳐 있다면 `"local"`로 직접 지정하는 편이 확실합니다. Colab 가상 머신의 로컬 디스크는
100GB 안팎이며, 데이터 압축을 푼 뒤에도 가중치 54GB가 들어갈 만큼은 남습니다.

## 4. 파라미터 셀에서 바꿀 값

21번 노트북의 1번 절에서 아래 값만 확인하면 됩니다. 나머지는 기본값 그대로 두십시오.

```python
DRIVE_DIR = "/content/drive/MyDrive/vqa"        # 개인 Drive의 폴더 이름이 다르면 고칩니다
ARCHIVE_PATH = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
RUN_ROOT = "/content/drive/MyDrive/vqa/runs"
MODEL_CACHE = "local"        # Drive 여유가 60GB에 못 미치면 이 값으로 둡니다
PRECISION_POLICY = "bf16"    # 그대로 둡니다. VRAM이 모자라면 노트북이 nf4로 내려갑니다
CONFIG = ""                  # 그대로 둡니다. 첫 실행은 기본 해상도로 돌립니다
RUN_RESOLUTION_COMPARE = True   # 10-1절의 확대본 비교를 함께 돌립니다
```

`PRECISION_POLICY`를 `"bf16"`으로 두는 까닭은, 정책과 실제 정밀도가 다를 때 run-id 뒤에
`_nf4`가 붙어서 80GB에서 돌린 팀의 기존 결과와 섞이지 않게 하기 위해서입니다. 40GB나 24GB에서
돌리면 원본 추론의 run-id는 `colab27b_base_bf16_nf4`가 되고, 확대본 추론의 run-id는
`colab27b_base_bf16_upscale_nf4`가 됩니다.

학습은 시도하지 마십시오. 13번 절은 VRAM이 모자라면 오류를 내고 멈춥니다. 27B는 nf4로 낮춰도
학습 peak VRAM이 28GB였으므로 24GB L4에서는 들어가지 않고, 40GB에서도 확대본 학습은 허용하지
않습니다.

## 5. 실행 순서

1. 런타임 유형을 GPU로 지정하고 노트북을 엽니다.
2. 1번 절의 파라미터를 위 표대로 고칩니다.
3. 2번 절부터 6번 절까지 차례로 실행합니다. 5번 절의 출력에서 감지된 VRAM과 확정된 run-id,
   학습 허용 여부를 반드시 눈으로 확인하십시오. 5-1절에서 가중치를 내려받습니다.
4. 8번 절의 preflight를 실행합니다. 학습을 허용하지 않는 가상 머신에서는 학습 점검을 건너뛰고
   추론 점검만 수행하므로, `seconds_per_item`과 `pixel_limits` 두 가지를 확인하면 됩니다.
5. 9번 절에서 기본 해상도로 fold 0을 추론하고 평가합니다.
6. 10-1절에서 확대본으로 fold 0을 한 번 더 추론합니다. 이어지는 셀이 McNemar 검정 결과와
   함께, 작은 글자 때문에 틀렸다고 판정한 12문항의 정오 변화를 표로 출력합니다.
7. 11번 절부터는 test 전체 추론과 학습 단계입니다. 이번 실험에서는 실행하지 않습니다.

10번 절의 프롬프트 비교는 `RUN_PROMPT_COMPARE`가 `False`이므로 아무 일도 하지 않고 넘어갑니다.

## 6. 예상 소요 시간

2026-09-21에 A100 40GB에서 nf4로 측정한 값은 문항당 1.06초였습니다. 이 값을 기준으로 잡으면
다음과 같습니다.

| 단계 | A100 40GB | L4 24GB |
|---|---|---|
| 가중치 내려받기 (54GB) | 20분에서 60분 | 20분에서 60분 |
| 환경 설치와 데이터 압축 해제 | 10분 안팎 | 10분 안팎 |
| preflight (추론 점검만) | 10분 안팎 | 15분 안팎 |
| 기본 해상도 fold 0 추론 | 약 25분 | 1시간 안팎 |
| 확대본 fold 0 추론 | 약 35분 | 1시간 30분 안팎 |

L4의 수치는 A100 대비 연산 성능을 놓고 잡은 어림값이므로, 실제 값은 preflight의
`seconds_per_item`으로 대체해서 다시 계산하십시오. 확대본이 더 오래 걸리는 까닭은 평균 시각
토큰이 702개에서 1,027개로 약 1.46배가 되기 때문입니다.

전체로 보면 A100 40GB에서는 두 시간 안팎, L4에서는 네 시간 안팎을 잡아 두시면 됩니다. 가중치를
로컬 디스크에 받았다면 런타임이 회수될 때마다 내려받기 시간이 다시 듭니다.

## 7. 중간에 끊겼을 때

예측은 이미 끝난 문항을 건너뛰고 남은 문항만 다시 추론합니다. 다시 접속했다면 1번 절의
파라미터 셀부터 6번 절까지 다시 실행한 다음, 7번 절의 도우미 셀을 거쳐 중단된 절의 셀을 그대로
다시 실행하십시오. 상황별 대응은 `docs/colab_survival.md`에 더 자세히 적어 두었습니다.

## 8. Qwen3.8-27B 비교 실험

Qwen3.8-27B는 2026년 8월에 공개된 모델이며, HF에 올라온 설정 파일을 확인해 보면 구조가
Qwen3.5-27B와 같습니다. architectures가 `Qwen3_5ForConditionalGeneration`이고 model_type이
`qwen3_5`이며, 64층에 linear attention 세 층마다 full attention 한 층이 오는 배치까지 같습니다.
그래서 지금까지 쓰던 로더와 LoRA target 탐색 규칙, fla 가속 커널을 그대로 재사용합니다. 다만
이 판단은 설정 파일만 읽고 내린 것이므로, 학습을 시작하기 전에 preflight로 실제 동작을 반드시
확인해야 합니다.

이 실험의 목적은 모델 하나만 바꾸었을 때 fold 0 정확도가 얼마나 달라지는지 재는 것입니다.
`configs/colab_qwen38_27b_bf16_upscale.yaml`은 `configs/colab_qwen35_27b_bf16_upscale.yaml`을
복사해서 `model_id`와 `revision`만 바꾼 파일입니다. 해상도와 프롬프트, LoRA 설정, 학습
하이퍼파라미터가 모두 같으므로 두 실행의 차이를 모델 하나로 설명할 수 있습니다.

### 1번 절에서 바꿀 값

```python
CONFIG = QWEN38_UPSCALE_CONFIG   # configs/colab_qwen38_27b_bf16_upscale.yaml
PROMPT_ID = "v2_read_text"       # 3.5 확대본 실험과 같은 프롬프트를 씁니다
TRAIN_SCOPE = "fold0"            # fold 1~4로 학습하고 fold 0을 검증으로 남겨 둡니다
```

`MODEL_ID`와 `MODEL_REVISION`은 고치지 마십시오. 5번 절이 `CONFIG`에서 직접 읽어서 덮어쓰고,
5-1절의 가중치 내려받기도 그 값을 따라갑니다. 같은 값을 두 곳에 적어 두면 한쪽만 고쳤을 때
내려받은 가중치와 추론이 올리는 가중치가 어긋나기 때문에 이렇게 만들었습니다.

run-id는 모델에 따라 앞머리가 달라집니다. 위 설정으로 돌리면 원본 추론은
`colab38_27b_base_bf16_v2_read_text_upscale`이 되고 학습은
`colab38_27b_ft_bf16_v2_read_text_upscale`이 됩니다. Qwen3.5로 돌린 기존 run의 이름은
`colab27b`으로 시작하므로 두 모델의 예측 파일이 같은 폴더에 섞이지 않습니다.

가중치 캐시에는 주의가 필요합니다. Qwen3.5 스냅샷과 Qwen3.8 스냅샷이 각각 약 54GB이므로 두
모델을 번갈아 돌리면 모두 108GB가 쌓입니다. Drive 여유가 그만큼 없다면 `MODEL_CACHE`를
`"local"`로 두어 가상 머신의 로컬 디스크에 받으시고, 그래도 모자라면 다 쓴 쪽의
`models--Qwen--Qwen3.5-27B` 폴더를 직접 지우십시오.

### preflight에서 반드시 볼 항목

8번 절의 preflight는 처음 보는 모델에서 가장 먼저 깨질 만한 곳을 짚어 줍니다. 아래 다섯 가지를
확인하기 전에는 세 시간이 넘는 학습을 시작하지 마십시오.

| 검사 이름 | 확인할 내용 |
|---|---|
| `chat_template_tail` | generation prompt가 빈 `<think></think>`로 끝나는지 봅니다. Qwen3.8은 thinking이 기본으로 켜져 있고 reasoning effort 기본값이 xhigh이므로, 이 검사가 FAIL이면 추론 결과 전체를 믿을 수 없습니다. |
| `label_mask` | 표본마다 감독하는 토큰이 정확히 한 개인지 봅니다. 토크나이저가 달라져서 정답 글자가 여러 토큰으로 쪼개지면 여기에서 걸립니다. |
| `train_infer_prefix_match` | 학습 텍스트의 앞부분과 추론 프롬프트가 바이트 단위로 같은지 봅니다. chat template이 학습 경로와 추론 경로에서 다르게 렌더링되면 여기에서 드러납니다. |
| `kernel_fast_path` | Gated DeltaNet이 fla 가속 커널을 쓰는지 봅니다. 참조 구현으로 내려가면 학습 시간이 두 배 넘게 늘어나므로, 일정을 다시 잡을지 여기에서 판단합니다. |
| `pixel_limits` | 확대본의 `min_pixels` 상향이 실제로 적용되어 시각 토큰이 늘어났는지 봅니다. Qwen3.8의 vision patch 크기가 3.5와 같은지를 실측으로 확인하는 자리이기도 합니다. |

`lora_targets`와 `forward_backward`도 함께 보십시오. LoRA target을 language backbone에서
찾지 못하거나 vision 경로에 LoRA가 붙으면 preflight가 FAIL로 알려 줍니다.

### 예상 소요 시간

A100 80GB에서 bf16으로 돌릴 때를 기준으로 잡은 값입니다.

| 단계 | 예상 시간 |
|---|---|
| 가중치 내려받기 (54GB) | 3분에서 5분 |
| preflight | 10분 안팎 |
| fold 0 추론 (1,344문항) | 25분 안팎 |
| LoRA 학습 (fold 1~4, 5,358문항) | 3시간 30분 안팎 |

이 표는 80GB A100을 배정받았을 때를 전제로 합니다. 3번 절에서 설명한 것처럼 개인 계정에서
40GB A100이나 24GB L4를 받았다면 노트북이 학습을 막으므로, 추론까지만 돌리시고 학습은 80GB를
받은 가상 머신에서 시작하십시오.

내려받기 시간은 Colab 가상 머신과 Hugging Face 사이의 회선이 빠를 때의 값입니다. 익명 요청의
속도 제한에 걸리면 훨씬 오래 걸리므로, Colab 보안 비밀에 `HF_TOKEN`을 등록해 두시는 편이
좋습니다. 학습 시간은 Qwen3.5 확대본의 실측값을 그대로 옮긴 어림값이므로, preflight의
`sec_per_optimizer_step`을 보고 다시 계산하십시오. 3.8에서 가속 커널이 붙지 않으면 이 값이 두
배 넘게 늘어날 수 있습니다.

## 9. Gemma-4-31B 비교 실험

`google/gemma-4-31B-it`는 Qwen 계열과도 InternVL 계열과도 다른 세 번째 계통입니다. 계통이
다르면 같은 문항에서 틀리는 자리도 달라지므로, 단일 모델의 정확도가 조금 낮더라도 27B 확률과
섞었을 때에는 올라갈 여지가 있습니다. 이 설정이 검증한 것은 zero-shot 추론이며, 학습은 아래
"학습을 막는 까닭" 항목에 적은 두 가지 제약을 먼저 풀어야 시작할 수 있습니다.

### 모델과 설정 파일

설정 파일은 `configs/colab_gemma4_31b_bf16.yaml`입니다. 주요 값은 다음과 같습니다.

| 항목 | 값 |
|---|---|
| `model_id` | `google/gemma-4-31B-it` (게이트가 걸려 있지 않아서 HF 토큰 없이 받을 수 있습니다) |
| `revision` | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| `precision` / `attn_implementation` | `bf16` / `sdpa` |
| `prompt_id` | `v2_read_text` |
| `weights_gb` | 62 (bf16 safetensors 두 개로 약 62.5GB입니다) |
| `processor_kwargs.max_soft_tokens` | 1120 |
| `lora.targets` | `auto` (`model.language_model` 아래에서 410개를 찾고 vision 쪽은 한 개도 잡지 않습니다) |

해상도를 정하는 방식이 다른 모델들과 다릅니다. Qwen 계열은 `min_pixels`와 `max_pixels`로
픽셀 수를 직접 정하고 InternVL은 448 타일을 몇 장 자를지로 정하지만, Gemma-4의
`Gemma4ImageProcessor`는 이미지 한 장에 쓸 시각 토큰 예산을 `max_soft_tokens`로 받습니다.
받을 수 있는 값은 70, 140, 280, 560, 1120 다섯 개뿐이고 기본값은 280입니다. 예산은 상한이므로
실제 토큰 수는 원본 종횡비에 따라 그보다 조금 적게 나옵니다. 640x480 이미지를 넣어 본 결과
기본값 280에서 266개가 나왔고, 설정 파일이 쓰는 1120에서는 1,064개가 나왔습니다. 추론이 너무
느리다면 이 값을 560이나 280으로 내리십시오. 설정 파일에 남아 있는 `min_pixels`와
`max_pixels`는 스키마를 맞추기 위한 값이며 이 모델에서는 쓰이지 않습니다.

### 1번 절에서 바꿀 값

```python
CONFIG = GEMMA4_CONFIG           # configs/colab_gemma4_31b_bf16.yaml
PROMPT_ID = "v2_read_text"       # 설정 파일에 적힌 프롬프트와 맞춥니다
TRAIN_SCOPE = "fold0"            # fold 0을 검증으로 남겨 두고 원본 추론만 돌립니다
```

`MODEL_ID`와 `MODEL_REVISION`은 고치지 마십시오. 5번 절이 `CONFIG`에서 직접 읽습니다.
run-id는 `gemma4_31b_base_bf16_v2_read_text`가 되므로 27B run 폴더와 섞이지 않습니다.
학습 절은 건너뛰고 9번 절의 원본 추론만 돌린 다음, 나온 확률을 `scripts/blend_preds.py`로
27B 결과와 섞으십시오.

### 학습을 막는 까닭

가중치가 62GB이므로 5번 절이 bf16 추론 하한을 74GB로, bf16 학습 하한을 82GB로 잡습니다.
80GB A100은 추론 하한을 넘지만 학습 하한에는 못 미치므로, 그 가상 머신에서는 노트북이 학습을
막고 추론만 허용합니다. 학습까지 하려면 gradient checkpointing을 먼저 붙여야 합니다.

한 가지가 더 있습니다. `Gemma4Processor`는 배치를 인코딩할 때 이미지 목록이 표본마다 한 겹
중첩되어 있기를 요구합니다. `vqa/collator.py`가 평평한 목록으로 먼저 부르고 그 호출이 거부되면
중첩해서 다시 부르도록 되어 있어서 `micro_batch`를 2 이상으로 올려도 인코딩 자체는 되지만,
역전파까지 실제로 도는지는 preflight의 `forward_backward`로 확인해야 합니다.

### preflight에서 볼 항목

| 검사 이름 | 확인할 내용 |
|---|---|
| `chat_template_tail` | generation prompt가 `<\|turn>model\n<\|channel>thought\n<channel\|>`로 끝나는지 봅니다. Gemma-4는 `enable_thinking`의 기본값이 거짓이어서 생각 채널을 열자마자 닫아 두며, 그 덕분에 첫 응답 위치가 곧바로 답 한 글자가 됩니다. 이 꼬리가 달라졌다면 추론 결과 전체를 믿을 수 없습니다. |
| `choice_tokens` | a, b, c, d가 각각 단일 토큰인지 봅니다. 기대하는 id는 순서대로 236746, 236763, 236755, 236753입니다. |
| `label_mask` | 표본마다 감독하는 토큰이 정확히 한 개인지 봅니다. |
| `train_infer_prefix_match` | 학습 텍스트의 앞부분과 추론 프롬프트가 토큰 단위로 같은지 봅니다. |
| `lora_targets` | `language_prefix`가 `model.language_model`로 잡히고 target이 410개인지 봅니다. vision 경로에 LoRA가 붙으면 FAIL로 알려 줍니다. |
| `pixel_limits` | 이 설정에서는 WARN으로 빠집니다. `Gemma4ImageProcessor`에 `merge_size` 속성이 없어서 기대 범위를 세울 수 없기 때문이며, WARN은 전체를 막지 않습니다. 해상도가 제대로 올라갔는지는 이 검사가 아니라 추론 로그의 sequence 길이로 확인하십시오. |

### 예상 소요 시간

A100 80GB에서 bf16으로 zero-shot 추론만 돌릴 때를 기준으로 잡은 값입니다.

| 단계 | 예상 시간 |
|---|---|
| 가중치 내려받기 (62GB) | 4분에서 6분 |
| preflight (모델 적재까지) | 10분 안팎 |
| fold 0 추론 (1,344문항) | 40분에서 60분 |

추론 시간은 시각 토큰 예산을 1120으로 두었을 때의 어림값입니다. 이미지 한 장이 1,000토큰을
넘게 차지해서 Qwen3.5 확대본보다 sequence가 길어지므로, 일정이 빠듯하다면
`processor_kwargs.max_soft_tokens`를 560으로 내리고 다시 재 보십시오. 내려받기 시간은 Colab
가상 머신과 Hugging Face 사이의 회선이 빠를 때의 값이며, 익명 요청의 속도 제한에 걸리면 훨씬
오래 걸리므로 Colab 보안 비밀에 `HF_TOKEN`을 등록해 두시는 편이 좋습니다.

### QLoRA 학습

위의 zero-shot 추론은 bf16 가중치 62GB를 그대로 올리기 때문에 학습 활성값이 들어갈 자리가
남지 않습니다. 가중치를 nf4로 양자화해서 올리고 LoRA adapter만 bf16으로 학습하면 같은 80GB
A100에서 학습을 시작할 수 있습니다. 그 설정이 `configs/colab_gemma4_31b_nf4_lora.yaml`이며,
bf16 설정과 다른 곳은 정밀도와 `weights_gb`, 그리고 양자화에서 제외할 모듈 목록 세 가지뿐입니다.
모델 판과 `revision`, 시각 토큰 예산 1120, 프롬프트, LoRA와 학습 설정은 모두 같게 두었으므로
두 결과를 같은 조건에서 견줄 수 있습니다.

| 항목 | 값 |
|---|---|
| `precision` | `nf4` (compute dtype은 bf16이고 double quant를 켭니다) |
| `weights_gb` | 17 (실측값이 아니라 어림값입니다. 아래 "실측으로 확인할 값"을 보십시오) |
| `quant_skip_modules` | `model.vision_tower`, `model.embed_vision`, `lm_head` |

vision 쪽을 양자화에서 제외하는 까닭은 다음과 같습니다. 이미지 특징을 뽑는 계층을 4bit로
내리면 그 단계에서 이미 정보가 손실되고, 그 손실은 language 쪽에만 붙는 LoRA로 되돌릴 수
없습니다. `lm_head`를 함께 적은 까닭은 따로 있습니다. transformers의 quantizer는
`llm_int8_skip_modules`를 넘겨받으면 기본 제외 목록을 아예 쓰지 않으므로, 한 개라도 직접
적으면서 `lm_head`를 빠뜨리면 출력 계층까지 4bit로 내려갑니다. `vqa/modeling.py`가 이 이름을
자동으로 덧붙이기는 하지만, 설정 파일만 읽어도 무엇이 제외되는지 알 수 있도록 함께 적어
두었습니다.

#### 1번 절에서 바꿀 값

```python
CONFIG = GEMMA4_QLORA_CONFIG     # configs/colab_gemma4_31b_nf4_lora.yaml
PROMPT_ID = "v2_read_text"       # 설정 파일에 적힌 프롬프트와 맞춥니다
TRAIN_SCOPE = "fold0"            # 먼저 fold 1~4로 학습하고 fold 0으로 잽니다
```

`weights_gb`가 17이므로 5번 절이 학습 하한을 `max(70 × 17/54, 17 + 20) = 37.0GB`로 잡고,
80GB A100에서 `TRAINING_ALLOWED`가 참이 됩니다. 설정 파일 자체가 nf4이므로 run-id의 기본
이름에 있던 `_bf16`이 `_nf4`로 바뀌어 학습 run-id는 `gemma4_31b_ft_nf4_v2_read_text`가 되고,
이미 결합에 쓰고 있는 bf16 zero-shot의 `gemma4_31b_base_bf16_v2_read_text`를 덮어쓰지
않습니다. fold 0에서 판정을 마친 다음에 `TRAIN_SCOPE`를 `"full"`로 바꾸어 전체 데이터로 다시
학습하십시오. 그때의 run-id는 `gemma4_31b_ft_nf4_full_v2_read_text`가 됩니다.

#### 판정 기준

학습 결과를 nf4 zero-shot과 견주면 안 됩니다. nf4는 가중치를 4bit로 뭉개므로 같은 모델이라도
zero-shot 정확도가 bf16보다 낮게 나올 수 있고, 그렇게 낮아진 값을 기준으로 삼으면 양자화
손실을 메운 것까지 학습의 성과로 계산하게 됩니다. 기준은 bf16 zero-shot이 fold 0에서 기록한
0.9695이며, 이 값보다 0.005 이상, 즉 0.9745 이상이 나와야 결합 후보로 올립니다. 그만큼 오르지
않는다면 LoRA가 양자화 손실을 메우지 못한 것이므로, Gemma 쪽 확률은 지금처럼 bf16 zero-shot을
그대로 쓰는 편이 낫습니다.

#### preflight에서 볼 항목

| 검사 이름 | 확인할 내용 |
|---|---|
| `bitsandbytes_4bit` | 4bit 행렬곱의 순전파와 역전파가 도는지 봅니다. 이 검사가 FAIL이면 아래 항목을 볼 것도 없이 멈추고 bitsandbytes 설치부터 확인하십시오. |
| `model_load` | 기록된 정밀도가 `nf4`인지, 그리고 어느 Auto 클래스가 받았는지 봅니다. Colab이 고정해 둔 transformers 5.16.1에서는 `AutoModelForImageTextToText`가 받습니다. |
| `lora_targets` | `language_prefix`가 `model.language_model`로 잡히고 target이 410개인지 봅니다. bf16에서 잰 값과 같아야 합니다. 양자화된 계층은 `nn.Linear`가 아니라 `Linear4bit`이 되는데, `vqa/modeling.py`가 그 클래스 이름도 Linear로 인정하므로 개수가 줄어들면 안 됩니다. |
| `forward_backward` | peak VRAM과 `sec_per_optimizer_step`을 봅니다. 아래 "실측으로 확인할 값"에 적은 두 가지를 이 검사에서 확인합니다. |
| `label_mask` | 표본마다 감독하는 토큰이 정확히 한 개인지 봅니다. 정밀도를 바꾸어도 달라지지 않아야 하는 값이므로, 여기에서 어긋나면 양자화가 아니라 collator 쪽을 봐야 합니다. |
| `inference` | 20문항을 실제로 추론해서 정확도와 문항당 시간을 봅니다. 이 20문항의 정확도가 bf16으로 잰 값보다 눈에 띄게 낮다면 양자화 손실이 큰 것이므로, 학습을 시작하기 전에 그 사실을 먼저 기록해 두십시오. |

#### 실측으로 확인할 값

`weights_gb`에 적은 17은 어림값입니다. 31B 가운데 vision tower 약 550M과 embedding을 뺀
나머지 선형 계층이 4bit로 내려가면 약 15GB이고, 여기에 double quant의 상수와 4bit로
내려가지 않는 embedding을 더해서 잡은 값입니다. 다만 peft의
`prepare_model_for_kbit_training`이 4bit가 아닌 파라미터를 모두 fp32로 올리므로, 양자화에서
제외한 vision tower와 embedding이 fp32로 상주하면서 실제 사용량은 이보다 몇 GB 더 커집니다.
`forward_backward`가 보고하는 peak VRAM이 80GB에 가깝게 나온다면 `max_soft_tokens`를 560으로
내리거나 `quant_skip_modules`에서 `model.vision_tower`를 빼고 다시 재 보십시오.

#### 예상 소요 시간

| 단계 | 예상 시간 |
|---|---|
| 가중치 내려받기 (62GB) | 4분에서 6분 (bf16 스냅샷을 그대로 받습니다) |
| preflight (모델 적재와 4bit 검사까지) | 15분 안팎 |
| LoRA 학습 (fold 1~4, 5,358문항, 1 epoch) | 4시간 30분에서 6시간 |
| fold 0 추론 (1,344문항) | 50분에서 70분 |

내려받는 파일은 nf4로 돌리더라도 원본 bf16 스냅샷 약 62GB입니다. 양자화는 스냅샷을 읽어서
GPU에 올리는 시점에 수행하므로 캐시에 필요한 용량은 줄지 않습니다. 그런데 5-1절의 여유 공간
검사는 `weights_gb`를 보고 필요한 용량을 계산하므로, 이 설정에서는 실제로 필요한 62GB가 아니라
약 23GB만 있으면 통과합니다. 캐시 위치의 여유가 62GB에 못 미치면 검사를 통과하고도 내려받기가
도중에 실패하므로, Drive 여유가 넉넉하지 않다면 `MODEL_CACHE`를 `"local"`로 적어서 가상 머신
로컬 디스크에 받으십시오.

학습 시간은 실측값이 아니라 두 가지 어림을 곱해서 잡은 값입니다. 27B 확대본이 microbatch
하나에 1.5초가 걸렸는데, Gemma-4는 이미지 한 장이 시각 토큰 1,064개를 차지해서 sequence가 더
길고, 4bit 가중치를 행렬곱마다 되돌리는 비용과 gradient checkpointing의 재계산 비용이 함께
붙습니다. 그래서 microbatch 하나에 3초에서 4초가 걸린다고 보고, `micro_batch`가 1이므로
5,358문항에 그대로 곱했습니다. 실제 값은 preflight의 `sec_per_optimizer_step`을
`grad_accum`인 8로 나누어 확인하시고, 4시간을 크게 넘을 것 같으면
`processor_kwargs.max_soft_tokens`를 560으로 내려서 다시 재 보십시오. 다만 예산을 내리면
bf16 zero-shot과 조건이 달라지므로 위의 판정 기준을 그대로 쓸 수 없게 됩니다.

#### 처음 돌려 보는 경로라는 점

bitsandbytes와 Gemma-4를 함께 돌리는 것은 저희 파이프라인에서 실제 GPU로는 처음 시도하는
경로입니다. nf4 경로 자체는 Qwen3.5-27B로 검증했고 Gemma-4는 bf16으로 검증했지만, 두 가지를
겹친 조합은 아직 실행한 적이 없습니다. 특히 Gemma-4의 하이브리드 어텐션은 sliding window
계층과 full attention 계층을 섞어 쓰고 뒤쪽 계층이 앞 계층의 k/v를 공유하는데, 이 구조가
gradient checkpointing의 재계산과 어떻게 맞물릴지는 실행해 봐야 알 수 있습니다. 그러므로
학습을 걸기 전에 preflight를 반드시 끝까지 돌리시고, `forward_backward`가 PASS인지 눈으로
확인한 다음에 13번 절을 실행하십시오.
