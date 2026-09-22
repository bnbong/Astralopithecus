# Kaggle Discussion 초안: fla 커널 누락으로 인한 속도 저하

이 파일은 대회 Discussion에 올릴 게시글의 초안입니다. 아래 제목 후보 가운데 하나를 고른 다음,
구분선 아래의 본문을 그대로 붙여 넣으시면 됩니다.

**제목 후보 1**: Qwen3.5를 transformers 5.x로 돌리신다면 `fla` 설치 여부를 먼저 확인하십시오
(학습 속도 4배 차이)

**제목 후보 2**: 학습이 step당 58초에서 13.5초가 된 이유: Gated DeltaNet의 조용한 참조 구현
fallback

---

## 본문

Qwen3.5 계열을 transformers 5.x로 학습하거나 추론하시는 분들께 공유드립니다. 저희 팀은 Colab
A100에서 27B 모델을 LoRA로 학습하다가, 학습 속도가 예상보다 네 배 가까이 느렸고, 학습을
시작하고 40분쯤 지나서야 원인을 찾았습니다. 원인은 모델 설정이나 하이퍼파라미터가 아니라
`flash-linear-attention` 패키지가 설치되어 있지 않았다는 것 하나였습니다. 실행이 오류 없이
정상으로 끝나기 때문에 알아채기가 쉽지 않았고, 같은 함정에 빠지시는 분이 계실 것 같아 정리해
둡니다.

### 무슨 일이 일어나는가

Qwen3.5는 Gated DeltaNet 레이어와 일반 attention 레이어를 섞어 쓰는 하이브리드 구조입니다.
이 가운데 Gated DeltaNet 경로는 `chunk_gated_delta_rule` 이라는 함수를 호출하는데,
transformers는 이 함수를 전용 커널로 부르려고 시도합니다. 커널을 불러오지 못하면 예외를
던지지 않고 PyTorch로 작성된 참조 구현으로 조용히 내려갑니다. 이 동작은
`transformers/integrations/hub_kernels.py` 의 `use_kernel_func_from_hub_with_fallback` 에
들어 있습니다.

참조 구현은 정확도 면에서는 문제가 없습니다. 다만 속도가 크게 다릅니다. 저희 측정으로는
optimizer step 하나가 58초에서 13.5초로 줄었으므로 약 4.3배 차이였습니다.

### 증상과 확인 방법

로그에 다음과 같은 경고가 남습니다. 학습 로그가 길어지면 진행 표시줄에 묻히기 쉬우므로 따로
찾아보셔야 합니다.

```
`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation
because ... is not installed
```

더 확실한 확인 방법은 패키지를 직접 import해 보는 것입니다.

```bash
python -c "import fla; print(fla.__version__)"
```

여기에서 `ModuleNotFoundError` 가 나면 참조 구현으로 돌아가고 있는 상태입니다. 저희의 경우
step당 58초, microbatch당 7.3초가 나왔는데, 같은 설정의 참고 실행이 microbatch당 3.37초였기
때문에 뒤늦게 이상하다는 것을 알았습니다.

### 해결 방법

```bash
pip install --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2
pip install einops
```

`--no-deps` 를 붙이는 이유가 중요합니다. 이 두 패키지는 자기 의존성으로 torch와 triton을 다시
끌어옵니다. 그대로 설치하면 Colab 기본 이미지의 torch 2.11.0+cu128이 다른 빌드로 바뀌고,
저희가 검증해 둔 bf16 환경이 바뀔 수 있습니다. 그래서 의존성 해석을 막고 두 패키지만
넣습니다. 다만 `einops` 는 fla가 import 시점에 바로 사용하므로, `--no-deps` 로 설치했다면
별도로 받아 두셔야 합니다. einops는 의존성이 가벼워서 torch를 건드리지 않습니다.

설치한 다음에는 torch 버전이 설치 전후로 같은지 한 번 더 확인하시기를 권합니다.

```bash
python -c "import torch; print(torch.__version__)"
```

### 실측값

측정 조건은 다음과 같습니다. GPU는 A100 80GB 한 장이고, 모델은 Qwen3.5-27B를 bf16으로
올렸습니다. LoRA는 r=16이며 grad_accum은 8입니다. 두 측정 모두 같은 날 같은 코드로 돌렸고,
TF32는 두 경우 모두 켠 상태였습니다.

| 항목 | fla 없음 | fla 0.5.2 |
|---|---|---|
| 학습 microbatch | 7.3초 | 1.50초 (prepare 0.035초 / compute 1.46초) |
| preflight optimizer step | 측정하지 않음 | 11.99초 |
| 본 학습 optimizer step | 58초 | 13.5초 |
| 문항당 추론 | 측정하지 않음 | 0.41초 |
| peak VRAM | 측정하지 않음 | 55.4GB (preflight) / 57.0GB (학습) |

preflight에서 잰 11.99초와 본 학습의 13.5초가 다른 이유는, preflight가 무작위로 고른 64개
표본으로만 재기 때문입니다. 실제 데이터에는 시각 토큰이 더 많은 이미지가 섞여 있어서 본
학습이 조금 더 느립니다.

### 주의하실 점 세 가지

첫째로, 커널을 바꾸면 계산 결과의 끝자리가 조금 달라집니다. TF32를 켰을 때와 같은 성격의
차이입니다. 정확도에 의미 있는 영향을 주지는 않았지만, 이전 실행과 숫자를 그대로 비교하실
생각이라면 커널 설정이 같은지 함께 적어 두시는 편이 좋습니다.

둘째로, `causal-conv1d` 는 별개의 패키지입니다. 이 패키지가 없을 때 생기는 fallback은 Gated
DeltaNet 경로가 아니라 short convolution 경로이고, 속도에 주는 영향이 훨씬 작습니다. wheel이
없어서 소스 빌드가 필요하고 Colab에서는 빌드에 몇 분이 걸리는 데다 실패하기도 쉬워서, 저희는
설치하지 않고 넘어갔습니다. 그래도 위 표만큼의 개선은 그대로 얻었습니다.

셋째로, Colab에서 bf16 LoRA를 돌리신다면 `torchao` 를 먼저 제거하셔야 합니다. Colab 기본
이미지에는 torchao 0.10.0이 들어 있는데, PEFT가 양자화하지 않은 Linear에 LoRA를 붙이는
시점에 다음 오류로 실패합니다.

```
Found an incompatible version of torchao ... only versions above 0.16.0 are supported
```

`pip uninstall -y torchao` 를 설치 단계의 맨 앞에 두면 해결됩니다. 저희는 이 문제까지 겹쳐서
원인을 찾는 데 시간을 더 썼습니다.

### 환경 버전

아래 조합에서 preflight와 bf16 학습을 모두 통과했습니다.

| 패키지 | 버전 |
|---|---|
| torch | 2.11.0+cu128 |
| transformers | 5.16.1 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.50.2 |
| fla (flash-linear-attention) | 0.5.2 |

위 수치는 저희 환경에서 한 번씩 측정한 값이라서, 다른 GPU나 다른 배치 구성에서는 배수가
달라질 수 있습니다. 다만 참조 구현으로 내려갔는지 여부는 로그와 `import fla` 로 바로 확인할
수 있으므로, 학습을 시작하시기 전에 한 번 보시기를 권합니다.
