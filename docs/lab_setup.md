# 실습실 Windows 환경 준비

실습실 PC는 다섯 대이며 사양은 i7-14700, RAM 32GB, RTX 5060 Ti 16GB입니다. GPU가 Blackwell 세대이고
compute capability가 12.0이므로, CUDA 12.8 이상을 대상으로 빌드된 wheel이 필요합니다. 실습실 PC는
인터넷에 연결되어 있으므로 pip 설치와 Hugging Face 가중치 다운로드가 모두 동작합니다. 따라서
온라인 설치를 기본 경로로 삼고, 오프라인 준비는 대비책으로만 유지합니다.

작업은 `00_lab_env_check.ipynb` 가 대신 수행합니다. 이 문서는 그 노트북이 무엇을 하는지, 실패했을
때 무엇을 확인해야 하는지를 설명합니다.

## 1. 가상환경을 두 개 만드는 이유

Qwen2.5-VL-3B는 transformers 4.x에서 동작하고, Qwen3.5 계열은 transformers 5.x가 필요합니다.
두 계열을 한 환경에 함께 설치할 수 없으므로 conda 가상환경을 두 개 만들어서 병행합니다.

| 환경 이름 | transformers | 대상 모델 | 용도 |
|---|---|---|---|
| `vqa-tf4` | 4.x | Qwen2.5-VL-3B-Instruct | 오늘 바로 첫 숫자를 뽑는 용도입니다. |
| `vqa-tf5` | 5.x | Qwen3.5-4B, Qwen3.5-9B | 실습실 주력 경로입니다. |

두 환경을 만드는 명령은 다음과 같습니다.

```
conda create -n vqa-tf4 python=3.11 -y
conda create -n vqa-tf5 python=3.11 -y
conda activate vqa-tf4
python -m pip install ipykernel
python -m ipykernel install --user --name vqa-tf4 --display-name "Python (vqa-tf4)"
```

`vqa-tf5` 에도 같은 방식으로 커널을 등록합니다. 노트북을 열 때 오른쪽 위에서 커널을 고르면
어느 환경에서 실행할지 정할 수 있습니다.

위 명령의 `python=3.11` 은 예시 값입니다. `requirements/` 아래의 세 파일은 대상 파이썬 버전을
명시하고 있지 않고, 주최측 baseline은 실습실 PC에서 Python 3.13으로 실행되었습니다. 우리가 아직
검증하지 않은 버전을 규칙으로 못박지 말고, `00_lab_env_check.ipynb` 가 통과한 버전을 그대로 쓰되
그 값을 팀 채널과 run manifest에 기록하십시오. 통과한 버전이 확정되면 이 문서와 requirements
파일 머리말에 그 버전을 적습니다.

노트북의 `TF_TRACK` 파라미터를 커널과 맞추십시오. `"tf4"` 로 두면 `requirements/lab_windows_tf4.txt`
를 설치하고 Qwen2.5-VL-3B config를 사용하며, `"tf5"` 로 두면 `requirements/lab_windows_tf5.txt` 를
설치하고 Qwen3.5-9B config를 사용합니다. 커널과 파라미터가 어긋나면 설치가 서로 덮어써서 두
환경이 모두 망가질 수 있습니다.

## 2. 온라인 설치 (기본 경로)

노트북의 `INSTALL_MODE` 를 `"online"` 으로 두고 설치 셀을 실행합니다. torch와 torchvision은
CUDA 12.8 빌드를 같은 인덱스에서 함께 받아야 합니다. torchvision은 transformers의 이미지 처리
경로가 사용하므로 선택 항목이 아니며, requirements 파일에도 `torchvision==0.26.0` 이 torch
2.11.0의 짝으로 고정되어 있습니다.

```
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
pip install -r requirements/lab_windows_tf5.txt
```

`scripts/audit_data.py` 는 phash 근접 중복 탐지에 ImageHash를 사용하며, 이 패키지가 없으면
검사를 조용히 건너뛰지 않고 오류로 중단합니다. requirements 세 파일에 모두 들어 있으므로
requirements를 통째로 설치했다면 따로 신경 쓸 일은 없습니다.

설치가 끝난 다음 torch를 처음 import 하는 경우에는 커널을 한 번 재시작하는 편이 안전합니다.
재시작한 뒤에는 파라미터 셀과 부트스트랩 셀부터 다시 실행하십시오.

### 대비책: 오프라인 설치

실습실 인터넷이 막히는 상황에 대비해 준비 절차만 남겨 둡니다. 인터넷이 되는 PC에서 wheel을
`downloads/libs` 에 모으고, 모델 가중치를 `downloads/models` 아래에 내려받습니다.

```
python -m pip download -r requirements/lab_windows_tf5.txt -d downloads/libs
hf download Qwen/Qwen3.5-9B --local-dir downloads/models/Qwen3.5-9B
```

내려받는 PC와 실습실 PC의 운영체제와 파이썬 버전이 같아야 하며, 다르면 맞는 wheel이 없다는
오류가 납니다. 노트북의 `INSTALL_MODE` 를 `"offline"` 으로 두면 설치 셀이
`--no-index --find-links downloads/libs` 를 붙이고 `HF_HUB_OFFLINE` 과 `TRANSFORMERS_OFFLINE` 도
함께 설정합니다. 모델을 미리 받았다면 `MODEL_ROOT` 파라미터에 `downloads/models` 를 적고,
snapshot의 revision을 재현 기록에 남기십시오.

9B 가중치는 용량이 크므로 다섯 대에 모두 복사하려면 시간이 걸립니다. 학습은 한 대에서만
수행하고, 완성된 adapter만 다른 PC에 배포해서 추론 대상 ID를 나누는 방식으로 진행합니다.

## 3. torch가 GPU를 인식하는지 확인

노트북의 3번 셀이 별도 프로세스에서 확인합니다. 통과 기준은 다음과 같습니다.

1. `torch.__version__` 에 `+cu128` 이 포함되어야 합니다.
2. `torch.cuda.is_available()` 이 True여야 합니다.
3. `torch.cuda.get_device_name()` 이 RTX 5060 Ti를 가리켜야 합니다.
4. compute capability가 `(12, 0)` 이어야 합니다.

`+cu` 가 없으면 CPU 전용 torch가 설치된 것입니다. 오프라인 설치에서 이 문제가 생기면
`downloads/libs` 에 `torch-*+cu128` wheel이 실제로 들어 있는지 확인하십시오. pip가 맞는 wheel을
찾지 못하면 조용히 CPU 빌드를 고르는 경우가 있습니다.

compute capability가 다르게 나오면 다른 GPU가 잡힌 것이거나 드라이버가 오래된 것입니다.
`nvidia-smi` 로 드라이버 버전을 확인하십시오.

## 4. Blackwell에서 bitsandbytes를 쓸 때의 주의

9B 모델은 NF4 4비트 양자화로 학습합니다. 이때 bitsandbytes가 필요합니다.

bitsandbytes 공식 설치표에는 CUDA 12.8 계열의 Linux와 Windows sm120 대상이 포함되어 있습니다.
따라서 Blackwell을 무조건 미지원으로 간주하지 않습니다. 다만 설치에 성공했다는 것과 실제 연산이
동작한다는 것은 다른 문제이므로, 실제 wheel과 드라이버, PyTorch, 연산의 조합이 동작하는지를 직접
확인해야 합니다.

확인은 `python -m vqa.preflight` 가 수행합니다. 4비트 상태에서 순전파와 역전파가 모두 성공하고
peak VRAM이 16GB 안에 들어오면 통과입니다. 점검 결과는 화면에 출력되는 동시에
`<RUN_ROOT>/preflight/<시각>.json` 에 저장되며, `RUN_ROOT` 의 기본값은 `artifacts/runs` 입니다.

속도 보고에는 값이 세 개 들어 있습니다. `sec_per_microbatch_mean` 은 microbatch 하나의 평균
시간이고, `sec_per_microbatch_longest` 는 가장 긴 표본이며, `sec_per_optimizer_step` 은 앞의
평균에 `grad_accum` 을 곱한 값입니다. 학습 시간 예산은 반드시 `sec_per_optimizer_step` 으로
계산하십시오. microbatch 값으로 계산하면 누적 배수만큼 시간을 낮게 잡게 됩니다. 예전 키인
`warm_seconds_mean` 은 `sec_per_microbatch_mean` 의 별칭이므로 예산에 쓰지 않습니다.

`train_infer_prefix_match` 점검도 함께 돌아갑니다. 학습 텍스트는 추론용 프롬프트에 정답 letter와
end-of-turn 접미사를 이어 붙여서 만들기 때문에, 정답 letter 직전까지의 prefix가 추론 프롬프트와
글자 단위로 같아야 합니다. 이 점검이 실패하면 학습과 추론이 서로 다른 입력을 보고 있다는 뜻이며,
정확도 숫자를 신뢰할 수 없으므로 학습을 시작하기 전에 원인을 찾아야 합니다.

여기에서 실패하면 다음 순서로 대응합니다.

1. 먼저 `max_pixels` 를 낮춥니다. config의 `max_pixels` 는 시각 토큰 수에 패치 넓이를 곱한
   값으로 적혀 있습니다. Qwen3.5 계열은 패치가 32곱하기32이므로 시각 토큰 768개가 786432이고,
   512개로 내리면 524288이 됩니다. 이때 학습과 validation, test에 모두 같은 값을 적용합니다.
2. 그래도 실패하면 4B 모델을 BF16 LoRA로 점검합니다. 다만 가중치가 약 8GB라는 계산만으로 16GB
   환경에서 학습이 가능하다고 단정하지 않습니다. 실제로 역전파까지 확인해야 합니다.
3. 두 경로 모두 실패하면 실습실에서는 추론만 담당하고 학습은 Colab에 맡기는 구성으로 바꿉니다.

대회 기간에 운영체제를 다시 구축하거나 드라이버를 크게 바꾸는 작업에는 시간을 쓰지 않습니다.
이미 동작하는 환경과 허용된 wheel을 우선하고, 최적화 경로가 없으면 실측 처리량에 맞춰 작업량을
줄이거나 배치 구성을 바꾸는 편이 낫습니다.

## 5. 가속 커널에 대한 확인

Qwen3.5 계열은 Gated DeltaNet과 causal convolution을 사용합니다. Transformers에는 전용 커널을 쓰는
경로와 순수 torch로 대체하는 경로가 함께 있으므로, 설치에 성공하고 import가 되었다는 사실만으로
가속 경로가 실제로 쓰인다고 판단하면 안 됩니다.

preflight가 순전파와 역전파를 실행하면서 경고와 실제 지연 시간을 보고합니다. 고정한 버전에서
가속 경로가 실제로 쓰이고 처리량이 예산에 맞아야 점검을 통과한 것으로 봅니다. FlashAttention
설정만 확인하고 linear attention 가속까지 확인했다고 판단하지 마십시오.

FLA나 Triton 같은 구성 요소는 Linux와 Windows에서 설치 조건이 다를 수 있습니다. Colab에서 되는
방법이 실습실에서 그대로 되지 않을 수 있으므로, 두 환경의 확인 결과를 따로 기록합니다.

## 6. 야간 실행 전에 확인할 것

실습실에서 학습을 밤새 돌리려면 시작하기 전에 세 가지를 확인해야 합니다.

1. 퇴실 시각과 전원 정책을 확인합니다. PC가 자동으로 꺼지거나 로그아웃되면 분리된 프로세스도
   함께 종료됩니다.
2. 절전 모드와 화면 보호기가 GPU 작업을 중단시키지 않는지 확인합니다.
3. checkpoint 저장 간격을 확인합니다. 중단 시각이 정해져 있다면 그 전에 완결된 저장이 남도록
   간격을 잡아야 합니다.

지속 실행이 불가능한 환경이라면, optimizer step 경계에서 저장하고 다음 날 같은 명령을 다시
실행해서 이어 가는 방식으로 진행합니다. `12_lab_train_9b_qlora.ipynb` 의 중단과 재개 시험을
반드시 먼저 통과시키십시오. 그 시험을 건너뛰면 실제로 중단되었을 때 이어지는지 알 수 없습니다.
