#!/usr/bin/env python3
"""Generate the team-facing notebooks under notebooks/ with nbformat.

The notebooks are deliberately thin: a parameter cell, a bootstrap cell and
then cells that shell out to the `python -m vqa.*` / `scripts/*` CLIs.
No model logic lives inside a notebook, so this generator stays readable and
the produced .ipynb files stay diff-friendly (no outputs, no execution counts).

Usage:
    .venv/bin/python scripts/build_notebooks.py
    .venv/bin/python scripts/build_notebooks.py --check   # validate only
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

KERNELSPEC = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


# --------------------------------------------------------------------------
# cell helpers
# --------------------------------------------------------------------------
def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str, parameters: bool = False):
    cell = new_code_cell(text.strip("\n"))
    cell["execution_count"] = None
    cell["outputs"] = []
    if parameters:
        cell["metadata"]["tags"] = ["parameters"]
    return cell


def notebook(cells):
    nb = new_notebook(cells=cells, metadata=dict(KERNELSPEC))
    return nb


# --------------------------------------------------------------------------
# shared source fragments
# --------------------------------------------------------------------------
BOOT = r'''
# --- 공통 부트스트랩: 경로, 환경 변수, 명령 실행 헬퍼 (수정할 일 거의 없음) ---
import os, sys, json, shlex, subprocess, time, csv, hashlib, random
from pathlib import Path

PROJECT_ROOT = Path(PROJECT_ROOT).expanduser().resolve()
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _setenv(name, value, create=False):
    if not value:
        return None
    p = Path(value).expanduser()
    if create:
        p.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(p)
    return p


DATA_ROOT_P = _setenv("VQA_DATA_ROOT", DATA_ROOT)
RUN_ROOT_P = _setenv("VQA_RUN_ROOT", RUN_ROOT, create=True)
SCRATCH_P = _setenv("VQA_LOCAL_SCRATCH", LOCAL_SCRATCH, create=True)
MODEL_ROOT_P = _setenv("VQA_MODEL_ROOT", MODEL_ROOT)

PY = sys.executable


def run(cmd, check=True, echo=True, quiet=False, hint=True):
    """명령을 실행하면서 출력을 그대로 보여 주고 종료 코드와 소요 시간을 돌려줍니다."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(x) for x in cmd]
    if echo:
        print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1,
    )
    buf = []
    for line in proc.stdout:
        buf.append(line)
        if not quiet:
            print(line, end="", flush=True)
    rc = proc.wait()
    dt = time.time() - t0
    print("[rc=%d] %.1f초" % (rc, dt), flush=True)
    if rc != 0:
        if hint:
            print("실패했습니다. 같은 명령에 --help 를 붙여서 인자 이름부터 확인하십시오.", flush=True)
        if check:
            raise RuntimeError("명령이 실패했습니다: " + " ".join(cmd))
    return {"rc": rc, "seconds": dt, "output": "".join(buf)}


def vqa(module, *args, **kw):
    """python -m vqa.<module> 을 실행합니다."""
    return run([PY, "-m", "vqa." + module] + [str(a) for a in args], **kw)


def script(name, *args, **kw):
    """scripts/<name> 을 실행합니다."""
    return run([PY, str(Path("scripts") / name)] + [str(a) for a in args], **kw)


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_ids(path):
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def sha256_of(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:limit] if limit else digest


print("PROJECT_ROOT      :", PROJECT_ROOT)
for _name in ("VQA_DATA_ROOT", "VQA_RUN_ROOT", "VQA_LOCAL_SCRATCH", "VQA_MODEL_ROOT"):
    print("%-18s: %s" % (_name, os.environ.get(_name, "(미설정)")))
'''

THUMBS = r'''
# --- 이미지 미리보기 헬퍼 (표시 전용) ---
import base64, io
from IPython.display import HTML, display
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def thumb_b64(path, size=380):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def show_images(paths, captions=None, size=380):
    captions = captions or [Path(p).name for p in paths]
    html = ['<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start">']
    for p, cap in zip(paths, captions):
        html.append(
            '<figure style="margin:0;font-size:12px">'
            '<img src="data:image/jpeg;base64,%s" style="max-width:%dpx">'
            "<figcaption>%s</figcaption></figure>" % (thumb_b64(p, size), size, cap)
        )
    html.append("</div>")
    display(HTML("".join(html)))


print("이미지 미리보기 헬퍼를 준비했습니다.")
'''

ASCII_CURVE = r'''
def ascii_curve(xs, ys, width=64, height=12):
    """matplotlib 없이도 loss 추이를 확인할 수 있도록 문자로 곡선을 그립니다."""
    if not ys:
        return "(표시할 데이터가 없습니다.)"
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    n = len(ys)
    grid = [[" "] * width for _ in range(height)]
    for i in range(width):
        j = int(i * (n - 1) / max(width - 1, 1))
        r = int((hi - ys[j]) / (hi - lo) * (height - 1))
        grid[r][i] = "*"
    out = []
    for r, row in enumerate(grid):
        v = hi - (hi - lo) * r / (height - 1)
        out.append("%9.4f |%s" % (v, "".join(row)))
    out.append(" " * 10 + "+" + "-" * width)
    out.append(" " * 11 + "step %s -> %s (%d개 기록)" % (xs[0], xs[-1], n))
    return "\n".join(out)
'''

CLI_ASSUMPTIONS_MD = """
### 명령 인자에 대한 안내

이 노트북의 명령은 `vqa/` 와 `scripts/` 에 실제로 구현된 인자에 맞춰 작성했습니다. 그래도 코어
코드가 계속 바뀌는 동안에는 인자 이름이 달라질 수 있습니다. 셀이 `unrecognized arguments` 로
실패하면 먼저 해당 명령에 `--help` 를 붙여서 실제 인자 이름을 확인하고, 셀 안의 문자열만 고치면
됩니다. 노트북의 다른 부분은 고치지 않아도 됩니다. 고친 내용은 팀 채널에 알려서 다른 사람이 같은
곳에서 막히지 않게 하십시오.
"""

# Colab 전용. 부트스트랩보다 먼저 실행되므로 BOOT의 헬퍼를 하나도 쓰지 않는다.
CODE_FETCH = r'''
# --- 코드 가져오기 (부트스트랩보다 먼저 실행합니다. BOOT의 헬퍼를 쓰지 않습니다.) ---
import os, re, shutil, subprocess, zipfile
from pathlib import Path

_code_root = Path(PROJECT_ROOT).expanduser()


def scrub_secret(text, secret=None):
    """출력에 토큰이 섞여 나가지 않도록 가립니다."""
    if not text:
        return ""
    if secret:
        text = text.replace(secret, "***")
    return re.sub(r"(https://)[^/@\s]*@", r"\1***@", text)


def _git(args, secret=None, check=True):
    """git 명령을 조용히 실행하고, 출력에서 토큰을 지운 뒤 돌려줍니다."""
    proc = subprocess.run(
        ["git"] + [str(a) for a in args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    out = scrub_secret(proc.stdout, secret)
    if check and proc.returncode != 0:
        raise RuntimeError("git 명령이 실패했습니다 (rc=%d).\n%s" % (proc.returncode, out.strip()))
    return proc.returncode, out


def _read_github_token():
    """Colab 비밀 저장소에서 토큰을 읽습니다. 없으면 None을 돌려줍니다."""
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN") or None
    except Exception:
        # SecretNotFoundError, NotebookAccessError, ImportError를 모두 여기에서 받습니다.
        return None


def _flatten_single_top_dir(target):
    """압축 안에 최상위 폴더가 하나뿐이면 그 안의 내용을 target 바로 아래로 올립니다."""
    entries = [p for p in target.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(target / item.name))
        inner.rmdir()
        return True
    return False


CODE_VERSION = None

if (_code_root / ".git").exists():
    rc, out = _git(["-C", str(_code_root), "pull", "--ff-only"], check=False)
    if rc == 0:
        print("코드를 최신 상태로 갱신했습니다.")
    else:
        print("git pull이 실패했습니다. 이미 받아 둔 코드를 그대로 사용하고 계속 진행합니다.")
        print(out.strip()[:600])
    _, _head = _git(["-C", str(_code_root), "rev-parse", "HEAD"], check=False)
    CODE_VERSION = _head.strip()
else:
    _token = _read_github_token()
    if _token:
        print("Colab 비밀 GITHUB_TOKEN을 찾았습니다. 저장소를 복제합니다.")
        _code_root.parent.mkdir(parents=True, exist_ok=True)
        _auth_url = REPO_URL.replace("https://", "https://x-access-token:%s@" % _token, 1)
        _rc, _out = _git(["clone", "--depth", "1", "--branch", REPO_BRANCH,
                          _auth_url, str(_code_root)], secret=_token, check=False)
        if _rc != 0:
            raise RuntimeError("저장소 복제에 실패했습니다. 토큰 권한과 저장소 주소를 확인하십시오.\n"
                               + _out.strip()[:600])
        # 토큰이 .git/config에 남지 않도록 곧바로 원래 주소로 되돌립니다.
        _git(["-C", str(_code_root), "remote", "set-url", "origin", REPO_URL], secret=_token)
        _, _head = _git(["-C", str(_code_root), "rev-parse", "HEAD"], check=False)
        CODE_VERSION = _head.strip()
        del _auth_url, _token
    elif CODE_ZIP and Path(CODE_ZIP).expanduser().exists():
        _zip_path = Path(CODE_ZIP).expanduser()
        print("토큰이 없어 Drive의 코드 압축 파일을 사용합니다:", _zip_path)
        _code_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_zip_path) as _zf:
            _zf.extractall(_code_root)
        if _flatten_single_top_dir(_code_root):
            print("압축 안의 최상위 폴더 하나를 펼쳐서 %s 바로 아래로 옮겼습니다." % _code_root)
        CODE_VERSION = "zip:" + _zip_path.name
    else:
        raise RuntimeError(
            "코드를 가져올 방법이 없습니다. 이 저장소는 비공개이므로 다음 두 가지 중 하나를 준비해야 합니다.\n"
            "1) GitHub에서 이 저장소에만 Contents 읽기 권한을 준 fine-grained personal access token을 만든 다음, "
            "Colab 왼쪽의 열쇠 아이콘 패널에서 이름을 GITHUB_TOKEN으로 등록하고 현재 노트북의 접근을 허용하십시오.\n"
            "2) 코드 전체를 zip으로 묶어 Drive에 올린 다음, 파라미터 셀의 CODE_ZIP에 그 경로를 적으십시오."
        )

if not (_code_root / "vqa").is_dir():
    print("주의: %s 아래에 vqa 패키지가 보이지 않습니다. 압축 파일의 구조를 확인하십시오." % _code_root)

print("코드 위치 :", _code_root)
print("코드 버전 :", CODE_VERSION)
'''

# 부트스트랩 직후에 실행합니다. CODE_VERSION과 RUN_ROOT_P가 모두 준비된 뒤여야 합니다.
CODE_VERSION_LOG = r'''
# --- 어떤 코드로 실행했는지 남깁니다 ---
_code_version_path = RUN_ROOT_P / "code_version.txt"
with _code_version_path.open("a", encoding="utf-8") as _fh:
    _fh.write("%s\t%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                            CODE_VERSION or "(알 수 없음)"))
print("코드 버전을 기록했습니다: %s -> %s" % (_code_version_path, CODE_VERSION))
'''

# 20번과 21번이 그대로 공유하는 설치 셀입니다. 여러 번 실행해도 결과가 같습니다.
ENV_SETUP = r'''
# --- 환경 설치와 확인 (20번 노트북과 21번 노트북이 같은 셀을 씁니다) ---
# Colab은 노트북마다 가상 머신을 따로 배정합니다. 20번에서 설치를 끝냈더라도 21번 가상 머신에는
# 아무것도 깔려 있지 않으므로, 새 가상 머신을 받을 때마다 이 셀을 처음부터 다시 실행해야 합니다.
REQ = "requirements/colab.txt"
if not Path(REQ).exists():
    raise RuntimeError(REQ + " 이 없습니다. 코드 가져오기 셀이 제대로 끝났는지 먼저 확인하십시오.")

# Colab 기본 이미지에는 torchao 0.10.0이 들어 있습니다. PEFT가 비양자화 Linear에 LoRA를 붙일 때
# 이 버전과 충돌해서 실패하므로 bf16 경로에서는 반드시 먼저 지웁니다. 설치되어 있지 않으면
# 아무 일도 일어나지 않습니다.
run([PY, "-m", "pip", "uninstall", "-y", "torchao"], check=False, quiet=True)

_install = run([PY, "-m", "pip", "install", "-q", "-r", REQ], check=False)
if _install["rc"] != 0:
    raise RuntimeError(
        "requirements/colab.txt 설치가 실패했습니다. 위 출력에서 어느 패키지가 막혔는지 확인한 다음 "
        "이 셀을 다시 실행하십시오."
    )

# 점검은 반드시 별도 프로세스에서 수행합니다. 지금 커널은 예전 버전을 이미 import 해 두었을 수
# 있고, 그러면 화면에는 옛 버전이 그대로 찍혀서 문제를 놓치게 됩니다.
_ENV_CHECK_SRC = """
import importlib
import torch

print("torch                :", torch.__version__)
print("torch.cuda.available :", torch.cuda.is_available())
if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(0)
    print("gpu                  :", _props.name)
    print("vram_gb              : %.1f" % (_props.total_memory / 1024 ** 3))
for _name in ("bitsandbytes", "transformers", "peft", "accelerate"):
    _mod = importlib.import_module(_name)
    print("%-21s: %s" % (_name, getattr(_mod, "__version__", "(버전 정보 없음)")))
try:
    import torchao
except ImportError:
    print("torchao              : 설치되어 있지 않습니다 (정상)")
else:
    raise AssertionError(
        "torchao가 아직 설치되어 있습니다(%s). PEFT가 bf16 LoRA를 붙일 때 실패합니다. "
        "pip uninstall -y torchao 를 다시 실행한 다음 런타임을 재시작하십시오."
        % getattr(torchao, "__version__", "버전 미상")
    )
if not torch.cuda.is_available():
    raise AssertionError("CUDA를 쓸 수 없습니다. 런타임 유형이 GPU인지 확인하십시오.")
print("환경 확인을 통과했습니다.")
"""

_verify = run([PY, "-c", _ENV_CHECK_SRC], check=False, echo=False)
if _verify["rc"] != 0:
    raise RuntimeError(
        "환경 확인이 실패했습니다. 바로 위 메시지를 그대로 읽고 원인을 해결한 다음 이 셀을 다시 "
        "실행하십시오. torchao가 남아 있다는 메시지가 보이면 런타임을 재시작하고 이 셀부터 다시 "
        "실행하면 됩니다."
    )
'''


# 압축 해제 부분만 공유합니다. 확인 출력은 노트북마다 따로 붙입니다.
DATA_STAGE = r'''
# --- 데이터를 로컬 디스크에 준비합니다 ---
import getpass, shutil

data_root = Path(DATA_ROOT)


def locate_data_root(base):
    """압축 안에 최상위 폴더가 하나 있었다면 그 하위 폴더를 데이터 위치로 삼습니다."""
    if (base / "train.csv").exists():
        return base
    if base.is_dir():
        subs = [p for p in sorted(base.iterdir())
                if p.is_dir() and p.name != "__MACOSX" and (p / "train.csv").exists()]
        if len(subs) == 1:
            return subs[0]
    return base


def ensure_7z():
    """7z 명령이 없으면 조용히 설치합니다."""
    if shutil.which("7z"):
        return
    print("7z 명령이 없어 p7zip-full을 설치합니다. 30초 정도 걸립니다.")
    run(["apt-get", "-qq", "install", "-y", "p7zip-full"], check=False, quiet=True)
    if not shutil.which("7z"):
        raise RuntimeError("p7zip-full 설치에 실패했습니다. 런타임의 네트워크 상태를 확인하십시오.")


data_root.mkdir(parents=True, exist_ok=True)
found_root = locate_data_root(data_root)

if (found_root / "train.csv").exists() and (found_root / "train").is_dir():
    print("이미 데이터가 준비되어 있습니다:", found_root)
else:
    ensure_7z()
    if str(ARCHIVE_PASSWORD).strip().lower() == "none":
        pwd = None
        print("ARCHIVE_PASSWORD가 none이므로 암호가 없는 압축 파일로 처리합니다.")
    else:
        pwd = ARCHIVE_PASSWORD or getpass.getpass("압축 파일 비밀번호를 입력하십시오: ")
    cmd = ["7z", "x", "-y"]
    if pwd:
        cmd.append("-p" + pwd)
    cmd += ["-o" + str(data_root), ARCHIVE_PATH]
    print("$ 7z x -y %s -o%s %s" % ("-p***" if pwd else "", data_root, ARCHIVE_PATH))
    run(cmd, echo=False)
    found_root = locate_data_root(data_root)

if found_root != data_root:
    print("압축 안의 하위 폴더를 데이터 위치로 사용합니다:", found_root)

data_root = found_root
DATA_ROOT = str(data_root)
DATA_ROOT_P = data_root
os.environ["VQA_DATA_ROOT"] = str(data_root)
'''


# --------------------------------------------------------------------------
# 00_lab_env_check
# --------------------------------------------------------------------------
def nb_00():
    cells = []
    cells.append(md("""
# 00. 실습실 PC 환경 점검 (전원 공통)

담당: 팀원 전원이 각자의 실습실 Windows PC에서 가장 먼저 실행합니다.
목표: 설치를 끝내고, torch가 RTX 5060 Ti를 cu128 빌드로 인식하는지 확인하고, 가벼운 preflight를 통과시킨 뒤,
그 결과를 팀 채널에 붙여 넣을 수 있는 형태로 출력하는 것입니다.

소요 시간은 온라인 설치 기준으로 15분에서 30분 사이입니다. 오프라인 설치라면 5분 내외로 끝납니다.

## 통과 기준(PASS)
1. `torch.__version__` 에 `+cu128` 이 포함되고 `torch.cuda.is_available()` 이 True입니다.
2. `torch.cuda.get_device_name()` 이 RTX 5060 Ti를 가리키고, compute capability가 `(12, 0)` 입니다.
3. `python -m vqa.preflight` 가 종료 코드 0으로 끝나고 결과 JSON이 저장됩니다.

## 넘겨줄 내용
마지막 셀이 출력하는 결과 블록을 팀 채널에 그대로 붙여 넣습니다. 준혁과 유림이 이 값을 보고
각 PC에 어떤 작업을 배정할지 정합니다.
"""))

    cells.append(md("""
## 1. 파라미터

아래 셀만 자신의 PC에 맞게 고치면 됩니다.

- `INSTALL_MODE`: 인터넷이 되면 `"online"`, 안 되면 `"offline"` 으로 둡니다.
- `TF_TRACK`: `"tf4"` 는 transformers 4.x 환경이며 Qwen2.5-VL-3B로 오늘 바로 돌릴 수 있습니다.
  `"tf5"` 는 transformers 5.x 환경이며 Qwen3.5 계열을 돌릴 때 필요합니다. 두 환경은 conda 가상환경을
  따로 만들어서 병행하는 것을 권장합니다. 자세한 절차는 `docs/lab_setup.md` 를 보십시오.
"""))

    cells.append(code(r'''
# ===== 파라미터 =====
PROJECT_ROOT = r"."              # 이 저장소의 위치. 예: r"C:\vqa\ssafy-16-2-ai"
DATA_ROOT = r"."                 # train.csv, train/, test/ 가 들어 있는 폴더
RUN_ROOT = r"./artifacts/runs"   # 산출물이 쌓이는 영속 폴더
LOCAL_SCRATCH = r"./artifacts/scratch"   # 빠른 로컬 디스크
MODEL_ROOT = r""                 # 오프라인 모델 폴더나 HF 캐시. 없으면 빈 문자열로 둡니다.

TEAM_MEMBER = "이름을적으세요"    # 예: "수빈"
PC_LABEL = "lab-pc-1"            # 실습실 PC를 구분할 이름

INSTALL_MODE = "online"          # "online" 또는 "offline"
TF_TRACK = "tf4"                 # "tf4" 또는 "tf5"
WHEEL_DIR = r"./downloads/libs"  # 오프라인 설치에 쓸 wheel 폴더

PREFLIGHT_TRAIN_STEPS = 0        # 가벼운 점검이므로 학습 step은 생략합니다.
PREFLIGHT_INFER_ITEMS = 5        # 추론은 5문항만 확인합니다.
''', parameters=True))

    cells.append(code(BOOT))

    cells.append(md("""
## 2. 라이브러리 설치

`INSTALL_MODE` 와 `TF_TRACK` 조합에 따라 설치 명령이 정해집니다. 오프라인이면 `--no-index` 를 붙여서
`downloads/libs` 의 wheel만 사용하므로 인터넷에 접속하지 않습니다.

설치가 끝난 뒤에 torch를 처음 import 하는 경우에는 커널을 한 번 재시작하는 편이 안전합니다.
재시작한 뒤에는 파라미터 셀과 부트스트랩 셀부터 다시 실행하십시오.
"""))

    cells.append(code(r'''
REQ_FILE = {"tf4": "requirements/lab_windows_tf4.txt",
            "tf5": "requirements/lab_windows_tf5.txt"}[TF_TRACK]

if not Path(REQ_FILE).exists():
    raise FileNotFoundError(
        REQ_FILE + " 파일이 없습니다. 코어 담당자가 requirements/ 를 올렸는지 확인하십시오."
    )

pip_cmd = [PY, "-m", "pip", "install", "-r", REQ_FILE]
if INSTALL_MODE == "offline":
    wheel_dir = Path(WHEEL_DIR).expanduser()
    if not wheel_dir.is_dir():
        raise FileNotFoundError(str(wheel_dir) + " 폴더가 없습니다. 오프라인 wheel을 먼저 준비하십시오.")
    pip_cmd += ["--no-index", "--find-links", str(wheel_dir)]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("오프라인 모드로 설치합니다. 인터넷에 접속하지 않습니다.")
else:
    print("온라인 모드로 설치합니다.")

run(pip_cmd)
'''))

    cells.append(md("""
## 3. torch가 GPU를 제대로 인식하는지 확인

별도 프로세스에서 확인하므로, torch import 단계에서 문제가 생겨도 노트북 커널은 살아 있습니다.
`+cu128` 이 아닌 CPU 전용 torch가 잡히면 실패로 처리합니다.
"""))

    cells.append(code(r'''
PROBE = r"""
import json, platform, sys
info = {"python": sys.version.split()[0], "platform": platform.platform()}
try:
    import torch
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
except Exception as exc:
    info["torch_error"] = repr(exc)
for mod in ("transformers", "peft", "accelerate", "bitsandbytes"):
    try:
        info[mod] = __import__(mod).__version__
    except Exception as exc:
        info[mod] = "미설치 또는 오류: " + repr(exc)[:80]
print("VQA_ENV_JSON " + json.dumps(info, ensure_ascii=False))
"""

res = run([PY, "-c", PROBE], check=False, quiet=True)
ENV_INFO = {}
for line in res["output"].splitlines():
    if line.startswith("VQA_ENV_JSON "):
        ENV_INFO = json.loads(line[len("VQA_ENV_JSON "):])
    else:
        print(line)

for k, v in ENV_INFO.items():
    print("%-16s: %s" % (k, v))

problems = []
if "+cu" not in str(ENV_INFO.get("torch", "")):
    problems.append("torch가 CPU 전용 빌드입니다. cu128 wheel로 다시 설치해야 합니다.")
if "cu128" not in str(ENV_INFO.get("torch", "")):
    problems.append("torch 빌드 태그에 cu128이 없습니다. 실습실 표준과 다른 빌드일 수 있습니다.")
if not ENV_INFO.get("cuda_available"):
    problems.append("torch.cuda.is_available()가 False입니다. 드라이버와 설치를 확인하십시오.")
if ENV_INFO.get("capability") and list(ENV_INFO["capability"]) != [12, 0]:
    problems.append("compute capability가 (12, 0)이 아닙니다. 실제 값: %s" % (ENV_INFO["capability"],))

if problems:
    print()
    print("확인이 필요한 항목")
    for p in problems:
        print(" - " + p)
else:
    print()
    print("PASS: torch가 Blackwell GPU를 cu128 빌드로 인식했습니다.")
'''))

    cells.append(md("""
## 4. 가벼운 preflight

학습 step은 0으로 두고 추론만 몇 문항 확인합니다. 본격적인 20-step 점검은 학습 담당인 유림이
`12_lab_train_9b_qlora.ipynb` 에서 수행하므로, 여기에서 반복할 필요는 없습니다.

`tf4` 환경이면 Qwen2.5-VL-3B config를, `tf5` 환경이면 Qwen3.5-9B config를 사용합니다.
"""))

    cells.append(code(r'''
CONFIG = {"tf4": "configs/lab_qwen25vl_3b.yaml",
          "tf5": "configs/lab_qwen35_9b.yaml"}[TF_TRACK]

print("사용할 config:", CONFIG)
if not Path(CONFIG).exists():
    print("아직 config 파일이 없습니다. 코어 담당자의 커밋을 기다렸다가 다시 실행하십시오.")
else:
    preflight_res = vqa(
        "preflight",
        "--config", CONFIG,
        "--n-train-steps", PREFLIGHT_TRAIN_STEPS,
        "--n-infer", PREFLIGHT_INFER_ITEMS,
        check=False,
    )
    print("preflight 종료 코드:", preflight_res["rc"])
'''))

    cells.append(md("""
## 5. 팀 채널에 공유할 결과 블록

아래 셀의 출력을 그대로 복사해서 팀 채널에 붙여 넣으십시오. 준혁이 이 값을 모아서 PC 배정과
시간 예산을 계산합니다.
"""))

    cells.append(code(r'''
latest_preflight = None
pf_dir = RUN_ROOT_P / "preflight" if RUN_ROOT_P else None
if pf_dir and pf_dir.is_dir():
    files = sorted(pf_dir.glob("*.json"))
    if files:
        latest_preflight = files[-1]

lines = []
lines.append("===== 실습실 환경 점검 결과 =====")
lines.append("담당자      : %s" % TEAM_MEMBER)
lines.append("PC          : %s" % PC_LABEL)
lines.append("설치 모드   : %s / transformers 계열 %s" % (INSTALL_MODE, TF_TRACK))
lines.append("python      : %s" % ENV_INFO.get("python"))
lines.append("torch       : %s (cuda %s)" % (ENV_INFO.get("torch"), ENV_INFO.get("torch_cuda")))
lines.append("GPU         : %s / VRAM %s GB / capability %s"
             % (ENV_INFO.get("device_name"), ENV_INFO.get("vram_gb"), ENV_INFO.get("capability")))
lines.append("transformers: %s" % ENV_INFO.get("transformers"))
lines.append("peft        : %s" % ENV_INFO.get("peft"))
lines.append("bitsandbytes: %s" % ENV_INFO.get("bitsandbytes"))
lines.append("preflight   : %s" % (latest_preflight.name if latest_preflight else "아직 없음"))
lines.append("판정        : %s" % ("PASS" if not problems else "확인 필요 %d건" % len(problems)))
for p in problems:
    lines.append("  - %s" % p)
lines.append("================================")

print("\n".join(lines))
'''))
    return notebook(cells)


# --------------------------------------------------------------------------
# 01_data_audit_split
# --------------------------------------------------------------------------
def nb_01():
    cells = []
    cells.append(md("""
# 01. 데이터 검사와 공식 split 확정

담당: 수빈
목표: CSV와 이미지를 검사하고, 근접 중복 후보를 사람이 직접 확인한 다음, 팀 공식 분할 파일을
vqa가 읽는 형식으로 바꿔서 내용 해시와 함께 배포하는 것입니다.

공식 분할은 팀이 이미 확정했습니다. 검증은 `fold_csv_files/val_fold0.csv`의 fold 0 1,344문항이고,
학습은 `fold_csv_files/train_fold1234.csv`의 5,358문항입니다. 학습 파일에서는 test와 이미지가
동일한 train 24문항을 빼 두었습니다. 분할 방식은 `fold_csv_files/데이터처리_검증분할.md`에
정리되어 있습니다. 이 노트북은 그 두 파일을 `artifacts/data/split_fold0.json` 하나로 합칩니다.
검증 500문항을 쓰던 `split_v1.json`은 폐기했으니 새 실행에서는 쓰지 마십시오.

## 통과 기준(PASS)
1. `scripts/audit_data.py` 가 종료 코드 0으로 끝나고, 경로 누락과 손상 이미지가 0건입니다.
2. 근접 중복 후보를 사람이 눈으로 확인해서 같은 장면인지 여부를 기록했습니다.
3. `scripts/folds_to_split.py` 가 val 1,344문항과 train 5,358문항을 겹침 없이 기록했습니다.
4. 정답 a~d 분포가 한쪽으로 크게 치우치지 않았습니다.

## 넘겨줄 내용
마지막 셀이 출력하는 split 해시 블록을 팀 채널에 공유합니다. 전원이 같은 split을 쓰는지 확인하는
기준값이 됩니다.
"""))

    cells.append(md(CLI_ASSUMPTIONS_MD))

    cells.append(code(r'''
# ===== 파라미터 =====
PROJECT_ROOT = r"."
DATA_ROOT = r"."
RUN_ROOT = r"./artifacts/runs"
LOCAL_SCRATCH = r"./artifacts/scratch"
MODEL_ROOT = r""

OUT_DIR = "artifacts/data"
AUDIT_REPORT = "artifacts/data/audit_report.json"
PAIRS_CSV = "artifacts/data/near_duplicate_pairs.csv"
FOLD_DIR = "fold_csv_files"                      # 팀 공식 분할 CSV가 있는 폴더
SPLIT_OUT = "artifacts/data/split_fold0.json"    # 위 CSV 두 개를 합쳐서 만드는 공식 split
VAL_IDS_OUT = "artifacts/data/val_fold0_ids.txt"
PHASH_THRESHOLD = 8     # 근접 중복 판정에 쓸 perceptual hash 거리 상한
DUP_PAIRS_TO_SHOW = 24  # 한 번에 눈으로 확인할 근접 중복 후보 쌍의 개수
''', parameters=True))

    cells.append(code(BOOT))
    cells.append(code(THUMBS))

    cells.append(md("""
## 1. 데이터 검사

`scripts/audit_data.py` 가 CSV의 필수 컬럼, 경로 존재 여부, 이미지 열림 여부, 중복 ID, 근접 중복
후보를 한 번에 검사합니다. 이미지 지문을 모두 계산하므로 처음 실행할 때는 몇 분이 걸립니다.

결과로 세 개의 파일이 만들어집니다. `audit_report.json` 은 기계가 읽는 요약이고,
`audit_report.md` 는 사람이 읽는 보고서이며, `near_duplicate_pairs.csv` 는 눈으로 확인해야 할
근접 중복 후보 목록입니다.
"""))

    cells.append(code(r'''
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

script("audit_data.py",
       "--data-root", os.environ["VQA_DATA_ROOT"],
       "--out-dir", OUT_DIR,
       "--threshold", PHASH_THRESHOLD)

AUDIT = json.loads(Path(AUDIT_REPORT).read_text(encoding="utf-8"))
print()
print("검사 결과 최상위 키:", sorted(AUDIT.keys()))
for split, info in (AUDIT.get("splits") or {}).items():
    print("[%s] %s" % (split, json.dumps(info, ensure_ascii=False)[:200]))
for key in ("exact_duplicates", "near_duplicates"):
    if key in AUDIT:
        print("%-18s: %s쌍" % (key, AUDIT[key].get("n_pairs")))
print()
print("사람이 읽을 보고서: artifacts/data/audit_report.md")
'''))

    cells.append(md("""
## 2. 근접 중복 후보를 눈으로 확인

자동으로 뽑은 후보를 사람이 확인합니다. 비슷한 간판이라는 이유만으로 전부 같은 그룹으로 묶으면
학습 데이터가 크게 줄어드니, 실제로 같은 장면을 찍은 사진인지를 기준으로 판단하십시오.

두 장이 같은 장면이면 `SAME` 목록에, 서로 다른 장면이면 `DIFFERENT` 목록에 쌍 번호를 적습니다.
"""))

    cells.append(code(r'''
DUP_PAIRS = read_csv_rows(PAIRS_CSV)
print("근접 중복 후보 쌍: %d개" % len(DUP_PAIRS))
if DUP_PAIRS:
    print("컬럼:", list(DUP_PAIRS[0].keys()))

ALL_ROWS = []
for name in ("train.csv", "dev.csv", "test.csv"):
    p = Path(os.environ["VQA_DATA_ROOT"]) / name
    if p.exists():
        ALL_ROWS += read_csv_rows(p)
ID2PATH = {r["id"]: Path(os.environ["VQA_DATA_ROOT"]) / r["path"] for r in ALL_ROWS}

TRAIN_ROWS = read_csv_rows(Path(os.environ["VQA_DATA_ROOT"]) / "train.csv")
ID2ROW = {r["id"]: r for r in TRAIN_ROWS}


def pair_ids(row):
    return row["id_a"], row["id_b"]


# train 안쪽 쌍을 먼저 봅니다. split 경계를 넘는 쌍은 누수 검사에서 따로 다룹니다.
priority = [r for r in DUP_PAIRS if r.get("split_a") == "train" and r.get("split_b") == "train"]
priority += [r for r in DUP_PAIRS if r not in priority]

for idx, row in enumerate(priority[:DUP_PAIRS_TO_SHOW]):
    ia, ib = pair_ids(row)
    print("--- 쌍 %d --- %s(%s) vs %s(%s), 거리 %s"
          % (idx, ia, row.get("split_a"), ib, row.get("split_b"), row.get("distance")))
    for i in (ia, ib):
        r = ID2ROW.get(i)
        if r:
            print("     질문: %s" % r["question"][:70])
    try:
        show_images([ID2PATH[ia], ID2PATH[ib]], captions=[ia, ib], size=320)
    except Exception as exc:
        print("  이미지를 열지 못했습니다:", exc)
'''))

    cells.append(code(r'''
# 위에서 확인한 결과를 쌍 번호로 적습니다. 예: SAME = [0, 3, 7]
SAME = []
DIFFERENT = []

confirm = {
    "reviewer": "수빈",
    "pairs_csv": PAIRS_CSV,
    "n_candidates": len(DUP_PAIRS),
    "same_pairs": [list(pair_ids(priority[i])) for i in SAME],
    "different_pairs": [list(pair_ids(priority[i])) for i in DIFFERENT],
    "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
CONFIRM_PATH = Path("artifacts/data/dup_confirm.json")
CONFIRM_PATH.write_text(json.dumps(confirm, ensure_ascii=False, indent=2), encoding="utf-8")
print("검수 결과를 저장했습니다:", CONFIRM_PATH)
print("같은 장면 %d쌍, 다른 장면 %d쌍" % (len(SAME), len(DIFFERENT)))
print()
print("이 파일은 사람이 남기는 검수 기록입니다. make_split.py는 이 파일을 읽지 않고,")
print("자체 threshold로 그룹을 묶습니다. 검수 결과와 자동 판정이 크게 어긋나면")
print("--threshold 값을 조정해서 다시 실행하는 방식으로 반영하십시오.")
'''))

    cells.append(md("""
## 3. 공식 split 파일 만들기

팀 공식 분할은 이미 확정되어 있습니다. 이 셀은 새로 나누지 않고, `fold_csv_files/`의 CSV 두
개를 읽어서 `vqa.train --split` 이 읽는 JSON 하나로 옮겨 적습니다. 학습 id와 검증 id가 하나라도
겹치면 스크립트가 실패하면서 멈춥니다.

이 분할은 이미지 MD5를 그룹 키로 쓰는 StratifiedGroupKFold 결과입니다. 픽셀이 조금 다른 근접
중복 이미지까지 묶지는 않기 때문에, fold 0과 학습 분할에 걸쳐 있는 유사 이미지가 한 쌍
남아 있습니다. `train_0395.jpg`와 `train_4458.jpg`이며, perceptual hash로만 일치합니다. 문항
수로 따지면 영향이 무시할 만한 수준이라서 그대로 둡니다.
"""))

    cells.append(code(r'''
script("folds_to_split.py",
       "--fold-dir", FOLD_DIR,
       "--data-root", os.environ["VQA_DATA_ROOT"],
       "--out-dir", OUT_DIR)

SPLIT = json.loads(Path(SPLIT_OUT).read_text(encoding="utf-8"))
train_ids = SPLIT["train_ids"]
val_ids = SPLIT["val_ids"]
print()
print("train %d개 / val %d개" % (len(train_ids), len(val_ids)))
overlap = set(train_ids) & set(val_ids)
print("겹치는 ID: %d개 (0이어야 합니다)" % len(overlap))
assert not overlap, "train과 val이 겹칩니다. fold CSV를 다시 확인하십시오."
print("검증 id 목록:", VAL_IDS_OUT, "존재:", Path(VAL_IDS_OUT).exists())

leak_path = Path(OUT_DIR) / "split_leakage.csv"
if leak_path.exists():
    leaks = read_csv_rows(leak_path)
    print("참고용 누수 검사 결과(감사 산출물): %d행" % len(leaks))
    for r in leaks[:10]:
        print("  ", json.dumps(r, ensure_ascii=False)[:160])
'''))

    cells.append(md("""
## 4. val 정답 분포 확인

한 선택지가 지나치게 많으면 정확도 해석이 왜곡됩니다. 대체로 각 선택지가 20퍼센트에서 30퍼센트
사이에 들어오면 충분합니다.
"""))

    cells.append(code(r'''
from collections import Counter

val_answers = Counter(ID2ROW[i]["answer"].strip().lower() for i in val_ids if i in ID2ROW)
total = sum(val_answers.values())
print("val 정답 분포 (전체 %d문항)" % total)
for letter in ["a", "b", "c", "d"]:
    n = val_answers.get(letter, 0)
    bar = "#" * int(40 * n / max(total, 1))
    print("  %s: %4d (%5.1f%%) %s" % (letter, n, 100.0 * n / max(total, 1), bar))

train_answers = Counter(ID2ROW[i]["answer"].strip().lower() for i in train_ids if i in ID2ROW)
print()
print("train 정답 분포 (전체 %d문항)" % sum(train_answers.values()))
for letter in ["a", "b", "c", "d"]:
    n = train_answers.get(letter, 0)
    print("  %s: %5d (%5.1f%%)" % (letter, n, 100.0 * n / max(sum(train_answers.values()), 1)))
'''))

    cells.append(md("""
## 5. split 해시를 팀에 공유

아래 출력의 `split_sha256` 값이 모든 팀원에게서 같아야 합니다. 값이 다르면 서로 다른 데이터로
평가하고 있는 것이므로 즉시 알려야 합니다.
"""))

    cells.append(code(r'''
content = json.dumps({"train_ids": sorted(train_ids), "val_ids": sorted(val_ids)},
                     ensure_ascii=False, sort_keys=True)
content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

block = [
    "===== 고정 split 공유 =====",
    "파일        : %s" % SPLIT_OUT,
    "train 문항  : %d" % len(train_ids),
    "val 문항    : %d" % len(val_ids),
    "split_sha256: %s" % content_hash,
    "파일 sha256 : %s" % sha256_of(SPLIT_OUT),
    "근접중복검수: 같은 장면 %d쌍 / 다른 장면 %d쌍" % (len(SAME), len(DIFFERENT)),
    "다음 작업   : 창혁은 11번 노트북, 유림은 12번 노트북을 시작할 수 있습니다.",
    "==========================",
]
print("\n".join(block))
'''))
    return notebook(cells)


# --------------------------------------------------------------------------
# 10_lab_baseline_zero_shot
# --------------------------------------------------------------------------
def nb_10():
    cells = []
    cells.append(md("""
# 10. 실습실 zero-shot 베이스라인

담당: 손이 비는 팀원 누구나 실행할 수 있습니다.
목표: 학습 없이 첫 정확도 숫자와 첫 제출 파일을 가장 빠르게 확보하는 것입니다.

이 노트북은 재실행해도 안전합니다. `python -m vqa.predict` 가 이미 끝난 ID를 건너뛰기 때문에,
중간에 커널이 죽거나 셀을 멈추더라도 같은 셀을 다시 실행하면 이어서 진행됩니다.

## 통과 기준(PASS)
1. val 예측이 끝나고 `evaluate_run` 이 정확도를 출력합니다. 무작위 추측은 25퍼센트이므로 그보다
   확실히 높아야 의미가 있습니다.
2. 전체 test 예측 전에 측정한 문항당 시간으로 6714문항 예상 소요 시간을 확인했습니다.
3. 제출 파일이 `scripts/check_submission.py` 검사를 통과합니다.

## 넘겨줄 내용
정확도, 문항당 시간, 제출 파일 경로와 해시를 수정에게 전달합니다.
"""))

    cells.append(md(CLI_ASSUMPTIONS_MD))

    cells.append(md("""
## 1. 파라미터와 모델 선택

| MODEL_CHOICE | 모델 | 필요한 환경 | 비고 |
|---|---|---|---|
| `qwen25vl_3b` | Qwen2.5-VL-3B-Instruct | transformers 4.x | 오늘 바로 돌아갑니다. 첫 숫자를 뽑는 용도로 가장 안전합니다. |
| `qwen35_4b` | Qwen3.5-4B | transformers 5.x | 비양자화 fallback 후보입니다. |
| `qwen35_9b` | Qwen3.5-9B (NF4) | transformers 5.x | 실습실 주력 모델이며 가장 느립니다. |

여러 대의 PC로 test를 나눠서 처리하려면 `SHARD` 를 `"1/3"`, `"2/3"` 처럼 서로 다르게 지정합니다.
한 대에서 전부 처리할 때는 빈 문자열로 둡니다.
"""))

    cells.append(code(r'''
# ===== 파라미터 =====
PROJECT_ROOT = r"."
DATA_ROOT = r"."
RUN_ROOT = r"./artifacts/runs"
LOCAL_SCRATCH = r"./artifacts/scratch"
MODEL_ROOT = r""

MODEL_CHOICE = "qwen25vl_3b"   # "qwen25vl_3b" | "qwen35_4b" | "qwen35_9b"
RUN_ID = "lab_zeroshot_v1"     # 모델을 바꾸면 RUN_ID도 함께 바꾸십시오.

VAL_IDS_FILE = "fold_csv_files/val_fold0.csv"    # 팀 공식 검증 분할 (fold 0, 1,344문항)
SPLIT_FILE = "artifacts/data/split_fold0.json"   # 팀 공식 학습 분할 (train_fold1234, 5,358문항)
VAL_OUT_NAME = "fold0_base"
TEST_CSV = "test.csv"
N_TEST_ITEMS = 6714            # 전체 test 문항 수

SHARD = ""                     # 예: "1/3". 한 대에서 전부 돌리면 빈 문자열입니다.
N_SHARDS_TOTAL = 3             # 병합 셀에서 사용할 전체 shard 개수
TIMING_LIMIT = 20              # 속도 측정에 사용할 문항 수
''', parameters=True))

    cells.append(code(BOOT))

    cells.append(code(r'''
CONFIGS = {
    "qwen25vl_3b": "configs/lab_qwen25vl_3b.yaml",
    "qwen35_4b": "configs/lab_qwen35_4b.yaml",
    "qwen35_9b": "configs/lab_qwen35_9b.yaml",
}
CONFIG = CONFIGS[MODEL_CHOICE]
RUN_DIR = RUN_ROOT_P / RUN_ID
PRED_DIR = RUN_DIR / "preds"
print("config :", CONFIG, "(존재:", Path(CONFIG).exists(), ")")
print("run dir:", RUN_DIR)
if not Path(VAL_IDS_FILE).exists():
    print("경고: val ID 파일이 아직 없습니다. 수빈이 01번 노트북을 끝낸 뒤에 진행하십시오.")
'''))

    cells.append(md("""
## 2. validation zero-shot 예측

adapter를 붙이지 않은 base 모델로 고정 validation을 추론합니다. 시간이 오래 걸리면 셀을 멈췄다가
다시 실행해도 됩니다. 이미 처리한 ID는 건너뜁니다.
"""))

    cells.append(code(r'''
vqa("predict",
    "--config", CONFIG,
    "--run-id", RUN_ID,
    "--csv", "train.csv",
    "--ids-file", VAL_IDS_FILE,
    "--adapter", "none",
    "--out-name", VAL_OUT_NAME)
'''))

    cells.append(md("""
## 3. validation 정확도

정확도가 25퍼센트 근처라면 프롬프트나 단일 토큰 매핑에 문제가 있을 가능성이 큽니다.
그 경우에는 preflight 로그의 단일 토큰 검사 결과부터 확인하십시오.
"""))

    cells.append(code(r'''
Path("artifacts/eval").mkdir(parents=True, exist_ok=True)
script("evaluate_run.py",
       "--pred", str(PRED_DIR / (VAL_OUT_NAME + ".csv")),
       "--gold", str(Path(os.environ["VQA_DATA_ROOT"]) / "train.csv"),
       "--val-ids", VAL_IDS_FILE,
       "--json-out", "artifacts/eval/%s_%s.json" % (RUN_ID, VAL_OUT_NAME),
       "--wrong-out", "artifacts/eval/%s_%s_wrong.csv" % (RUN_ID, VAL_OUT_NAME),
       "--note", "zero-shot %s" % MODEL_CHOICE)
'''))

    cells.append(md("""
## 4. 전체 test에 들어가기 전에 시간부터 재기

적은 수의 문항으로 실제 속도를 측정하고, 6714문항 기준 예상 시간을 계산합니다. 이 숫자를 보고
몇 대의 PC에 나눌지 결정하십시오. 예상 시간이 남은 실습 시간보다 길면 더 작은 모델로 내려가거나
PC를 더 투입해야 합니다.
"""))

    cells.append(code(r'''
timing = vqa("predict",
             "--config", CONFIG,
             "--run-id", RUN_ID,
             "--csv", TEST_CSV,
             "--adapter", "none",
             "--limit", TIMING_LIMIT,
             "--out-name", "timing_probe",
             check=False)

sec_per_item = timing["seconds"] / max(TIMING_LIMIT, 1)
print()
print("측정 결과: %.2f초/문항 (모델 로딩 시간이 포함된 값이라 보수적입니다)" % sec_per_item)
print()
print("전체 %d문항 예상 소요 시간" % N_TEST_ITEMS)
print("  PC 대수 | 예상 시간")
for n_pc in (1, 2, 3, 4, 5):
    hours = sec_per_item * N_TEST_ITEMS / 3600.0 / n_pc
    print("  %6d대 | %5.2f시간" % (n_pc, hours))
print()
print("모델 로딩이 한 번뿐인 긴 실행에서는 실제 속도가 이보다 빠릅니다.")
print("전체 실행을 시작한 뒤 진행률 표시로 실제 속도를 다시 확인하십시오.")
'''))

    cells.append(md("""
## 5. 전체 test 예측

위 예상 시간을 확인한 뒤에 실행하십시오. `SHARD` 를 지정하면 이 PC가 맡은 부분만 처리합니다.
중간에 멈춰도 같은 셀을 다시 실행하면 이어서 진행됩니다.
"""))

    cells.append(code(r'''
args = ["--config", CONFIG, "--run-id", RUN_ID, "--csv", TEST_CSV,
        "--adapter", "none", "--out-name", "test_base"]
if SHARD:
    args += ["--shard", SHARD]
    print("이 PC가 맡은 구간:", SHARD)
else:
    print("전체 구간을 이 PC에서 처리합니다.")

vqa("predict", *args)
'''))

    cells.append(md("""
## 6. shard 병합

여러 PC로 나눠서 돌렸다면, 각 PC의 `preds/test_base.shard{i}of{n}.csv` 파일을 한 폴더에 모은 뒤에
이 셀을 실행합니다. 각 ID가 정확히 한 번씩만 들어 있는지 병합기가 검사합니다.
한 대에서 전부 처리했다면 이 셀은 건너뛰어도 됩니다.
"""))

    cells.append(code(r'''
shard_files = sorted(PRED_DIR.glob("test_base.shard*of*.csv"))
print("모인 shard 파일 %d개" % len(shard_files))
for f in shard_files:
    print(" -", f.name)

if shard_files:
    script("merge_shards.py",
           "--shards", *[str(f) for f in shard_files],
           "--expect-ids", str(Path(os.environ["VQA_DATA_ROOT"]) / TEST_CSV),
           "--seconds-per-item", "%.4f" % sec_per_item,
           "--out", str(PRED_DIR / "test_base.csv"))
else:
    print("병합할 shard가 없습니다. 단일 PC 실행이라면 정상입니다.")
'''))

    cells.append(md("""
## 7. 제출 파일 생성과 검사

`check_submission.py` 가 통과해야 실제로 제출할 수 있습니다. 파싱에 실패한 행을 임의로 `a` 로
채우지 않도록 되어 있으므로, 결측이 보고되면 해당 ID를 다시 추론해야 합니다.
"""))

    cells.append(code(r'''
SUB_PATH = "artifacts/submission/%s_base.csv" % RUN_ID
Path("artifacts/submission").mkdir(parents=True, exist_ok=True)

vqa("submit",
    "--pred", str(PRED_DIR / "test_base.csv"),
    "--sample", "sample_submission.csv",
    "--out", SUB_PATH)

script("check_submission.py",
       "--submission", SUB_PATH,
       "--sample", "sample_submission.csv")

print()
print("===== 제출 후보 공유 =====")
print("run_id     : %s (%s)" % (RUN_ID, MODEL_CHOICE))
print("제출 파일  : %s" % SUB_PATH)
print("파일 sha256: %s" % sha256_of(SUB_PATH))
print("문항당 시간: %.2f초" % sec_per_item)
print("다음 담당  : 수정이 30번 노트북에서 실험 기록에 추가합니다.")
print("=========================")
'''))
    return notebook(cells)


# --------------------------------------------------------------------------
# 11_lab_prompt_ocr_compare
# --------------------------------------------------------------------------
def nb_11():
    cells = []
    cells.append(md("""
# 11. 프롬프트와 OCR, 해상도 비교

담당: 창혁
목표: 프롬프트 두 가지와 max_pixels 후보를 고정 표본에서 비교한 뒤, 팀 전체가 사용할 입력 설정을
하나로 동결하는 것입니다.

계획서 5절의 원칙을 그대로 따릅니다. 비교는 이미지만 사용하는 방식과 OCR을 보조로 붙이는 방식,
두 가지로만 제한합니다. 후보를 늘리면 시간이 부족해지고 표본에 과적합될 위험이 커집니다.

## 선행 조건
수빈의 01번 노트북이 끝나서 `artifacts/data/split_fold0.json` 이 존재해야 합니다. 개발 표본은 반드시
`train_fold1234` 의 학습 ID에서만 뽑습니다. fold 0에서 뽑으면 이후의 모든 정확도 비교가 오염됩니다.
아래 셀은 fold 0과 겹치는 ID가 하나라도 있으면 멈춥니다.

## 통과 기준(PASS)
1. 대표 이미지 20장 정도를 실제로 눈으로 확인했습니다.
2. 고정 표본 128문항에서 조합별 정확도와 소요 시간을 표로 확인했습니다.
3. `artifacts/input_config_frozen.json` 을 기록했고, 그 내용을 준혁과 유림에게 전달했습니다.

## 넘겨줄 내용
동결한 prompt_id, OCR 사용 여부, 모델별 max_pixels 값입니다. 학습이 시작된 뒤에는 이 값을
바꾸지 않습니다.
"""))

    cells.append(md(CLI_ASSUMPTIONS_MD))

    cells.append(code(r'''
# ===== 파라미터 =====
PROJECT_ROOT = r"."
DATA_ROOT = r"."
RUN_ROOT = r"./artifacts/runs"
LOCAL_SCRATCH = r"./artifacts/scratch"
MODEL_ROOT = r""

SPLIT_FILE = "artifacts/data/split_fold0.json"
DEV_SAMPLE_FILE = "artifacts/data/prompt_dev_128.txt"
DEV_SAMPLE_N = 128
SAMPLE_SEED = 20260921

CONFIG_MAIN = "configs/lab_qwen35_9b.yaml"      # 9B가 준비되면 이것을 사용합니다.
CONFIG_FALLBACK = "configs/lab_qwen25vl_3b.yaml"  # 9B가 아직 안 되면 3B로 대신합니다.
USE_FALLBACK = False

RUN_ID = "lab_prompt_cmp_v1"
PROMPT_IDS = ["v1_image_only", "v1_ocr"]
# config의 max_pixels는 "시각 토큰 수 x 패치 넓이"로 적습니다. Qwen3.5는 패치가 32x32이고
# Qwen2.5-VL은 28x28이므로, 후보는 토큰 수로 적고 아래에서 환산합니다.
VISUAL_TOKEN_CANDIDATES = [768, 512]
PATCH_AREA = {"lab_qwen35_9b": 32 * 32, "lab_qwen25vl_3b": 28 * 28}

IMAGE_SAMPLE_N = 20
KEYWORDS = ["영수증", "메뉴", "전화", "번호", "간판", "안내", "표지", "가격", "시간", "주소"]
''', parameters=True))

    cells.append(code(BOOT))
    cells.append(code(THUMBS))

    cells.append(code(r'''
CONFIG = CONFIG_FALLBACK if USE_FALLBACK else CONFIG_MAIN
CONFIG_KEY = Path(CONFIG).stem
AREA = PATCH_AREA[CONFIG_KEY]
MAX_PIXELS_CANDIDATES = [tok * AREA for tok in VISUAL_TOKEN_CANDIDATES]
print("사용할 config:", CONFIG, "(존재:", Path(CONFIG).exists(), ")")
print("패치 넓이 %d, max_pixels 후보 %s" % (AREA, MAX_PIXELS_CANDIDATES))

if not Path(SPLIT_FILE).exists():
    raise FileNotFoundError(
        SPLIT_FILE + " 이 아직 없습니다. 수빈의 01번 노트북이 끝날 때까지 기다려야 합니다."
    )
SPLIT = json.loads(Path(SPLIT_FILE).read_text(encoding="utf-8"))
TRAIN_IDS = list(SPLIT["train_ids"])
VAL_IDS = set(SPLIT["val_ids"])
print("train %d개 / val %d개" % (len(TRAIN_IDS), len(VAL_IDS)))
'''))

    cells.append(md("""
## 1. 대표 이미지 20장 확인

작은 글자, 영수증, 메뉴판, 전화번호처럼 판독이 어려운 사례를 우선 고릅니다. 키워드로 1차 후보를
뽑은 다음, 부족하면 무작위로 채웁니다. 이 단계의 목적은 어떤 해상도에서 글자가 읽히는지를
사람이 직접 판단하는 것입니다.
"""))

    cells.append(code(r'''
ROWS = read_csv_rows(Path(os.environ["VQA_DATA_ROOT"]) / "train.csv")
ID2ROW = {r["id"]: r for r in ROWS}

def row_text(r):
    return " ".join(str(r.get(k, "")) for k in ("question", "a", "b", "c", "d"))

keyword_hits = [r for r in ROWS if any(k in row_text(r) for k in KEYWORDS)]
rng = random.Random(SAMPLE_SEED)
rng.shuffle(keyword_hits)
picked = keyword_hits[:IMAGE_SAMPLE_N]
if len(picked) < IMAGE_SAMPLE_N:
    rest = [r for r in ROWS if r not in picked]
    rng.shuffle(rest)
    picked += rest[: IMAGE_SAMPLE_N - len(picked)]

print("대표 이미지 %d장을 골랐습니다." % len(picked))
REPRESENTATIVE_IDS = [r["id"] for r in picked]
Path("artifacts/data").mkdir(parents=True, exist_ok=True)
Path("artifacts/data/representative_ids.txt").write_text(
    "\n".join(REPRESENTATIVE_IDS) + "\n", encoding="utf-8")

for r in picked:
    print("[%s] %s" % (r["id"], r["question"][:60]))
show_images([Path(os.environ["VQA_DATA_ROOT"]) / r["path"] for r in picked],
            captions=[r["id"] for r in picked], size=300)
'''))

    cells.append(md("""
## 2. 고정 개발 표본 128문항 만들기

반드시 train ID에서만 뽑습니다. 아래 셀은 val과 겹치지 않는지 직접 확인하고, 겹치면 멈춥니다.
표본은 seed로 고정되므로 누가 언제 실행해도 같은 문항이 나옵니다.
"""))

    cells.append(code(r'''
rng2 = random.Random(SAMPLE_SEED)
pool = sorted(set(TRAIN_IDS) - VAL_IDS)
sample_ids = sorted(rng2.sample(pool, min(DEV_SAMPLE_N, len(pool))))

leak = set(sample_ids) & VAL_IDS
assert not leak, "개발 표본에 val ID가 섞였습니다: %s" % sorted(leak)[:5]

Path(DEV_SAMPLE_FILE).write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
print("개발 표본 %d문항을 저장했습니다: %s" % (len(sample_ids), DEV_SAMPLE_FILE))
print("val과 겹치는 ID: 0개")
print("표본 sha256:", sha256_of(DEV_SAMPLE_FILE)[:16])
'''))

    cells.append(md("""
## 3. 조합별 비교 실행

프롬프트 2종과 max_pixels 후보를 조합해서 순서대로 돌립니다. 조합이 늘어나면 시간이 비례해서
늘어나므로, 후보는 최소한으로 유지하십시오. 도중에 멈춰도 다시 실행하면 이미 끝난 조합의 예측은
건너뜁니다.
"""))

    cells.append(code(r'''
COMBOS = [(p, mp) for p in PROMPT_IDS for mp in MAX_PIXELS_CANDIDATES]
RESULTS = []

for prompt_id, max_pixels in COMBOS:
    tokens = max_pixels // AREA
    name = "cmp_%s_%dtok" % (prompt_id, tokens)
    print("=" * 70)
    print("조합: prompt_id=%s / max_pixels=%d (시각 토큰 %d개)" % (prompt_id, max_pixels, tokens))
    res = vqa("predict",
              "--config", CONFIG,
              "--run-id", RUN_ID,
              "--csv", "train.csv",
              "--ids-file", DEV_SAMPLE_FILE,
              "--adapter", "none",
              "--set", "prompt_id=%s" % prompt_id,
              "--set", "max_pixels=%d" % max_pixels,
              "--out-name", name,
              check=False)
    RESULTS.append({"prompt_id": prompt_id, "max_pixels": max_pixels, "tokens": tokens,
                    "out_name": name, "seconds": res["seconds"], "rc": res["rc"]})
'''))

    cells.append(code(r'''
Path("artifacts/eval").mkdir(parents=True, exist_ok=True)
PRED_DIR = RUN_ROOT_P / RUN_ID / "preds"

for row in RESULTS:
    eval_out = "artifacts/eval/%s_%s.json" % (RUN_ID, row["out_name"])
    r = script("evaluate_run.py",
               "--pred", str(PRED_DIR / (row["out_name"] + ".csv")),
               "--gold", str(Path(os.environ["VQA_DATA_ROOT"]) / "train.csv"),
               "--val-ids", DEV_SAMPLE_FILE,
               "--json-out", eval_out,
               "--no-breakdown", "--no-log",
               check=False, quiet=True)
    acc = None
    if Path(eval_out).exists():
        data = json.loads(Path(eval_out).read_text(encoding="utf-8"))
        for key in ("accuracy", "acc", "top1"):
            if key in data:
                acc = data[key]
                break
    row["accuracy"] = acc
'''))

    cells.append(md("""
## 4. 비교 표

정확도 차이가 128문항에서 1퍼센트에서 2퍼센트 수준이라면 표본 오차 범위 안입니다. 그 경우에는
더 단순하고 빠른 쪽을 고르는 편이 낫습니다. OCR을 채택하면 학습과 validation, test 전부에서 같은
입력 규칙을 써야 하고 OCR 캐시도 준비해야 하므로, 뚜렷한 이득이 있을 때만 선택하십시오.
"""))

    cells.append(code(r'''
print("%-16s %-12s %-10s %-10s" % ("prompt_id", "시각토큰", "정확도", "소요(초)"))
print("-" * 54)
for row in sorted(RESULTS, key=lambda r: (-(r["accuracy"] or 0), r["seconds"])):
    acc_txt = "%.3f" % row["accuracy"] if row["accuracy"] is not None else "미산출"
    print("%-16s %-12s %-10s %-10.1f"
          % (row["prompt_id"], "%d개" % row["tokens"], acc_txt, row["seconds"]))
print()
print("표본은 %d문항이므로 작은 차이는 우연일 수 있습니다." % DEV_SAMPLE_N)
'''))

    cells.append(md("""
## 5. 입력 설정 동결

아래 값을 직접 정해서 적은 뒤 셀을 실행합니다. 이 파일이 저장되는 순간부터 팀 전체의 입력 규칙이
고정됩니다. 준혁의 27B 실행과 유림의 9B 학습이 이 파일을 보고 설정을 맞춥니다.
"""))

    cells.append(code(r'''
# ===== 여기에 최종 결정을 적습니다 =====
FROZEN_PROMPT_ID = "v1_image_only"
FROZEN_OCR_ENABLED = False
# config와 같은 규칙으로 적습니다. 시각 토큰 수에 패치 넓이를 곱한 값입니다.
FROZEN_MAX_PIXELS = {
    "lab_qwen35_9b": 768 * 32 * 32,
    "lab_qwen25vl_3b": 768 * 28 * 28,
    "colab_qwen35_27b": 1024 * 32 * 32,
}
DECIDED_BY = "창혁"
NOTE = "128문항 비교 결과 OCR의 이득이 뚜렷하지 않아 이미지 단독으로 시작합니다."

frozen = {
    "prompt_id": FROZEN_PROMPT_ID,
    "ocr_enabled": FROZEN_OCR_ENABLED,
    "max_pixels": FROZEN_MAX_PIXELS,
    "decided_by": DECIDED_BY,
    "note": NOTE,
    "dev_sample_file": DEV_SAMPLE_FILE,
    "dev_sample_sha256": sha256_of(DEV_SAMPLE_FILE),
    "split_file": SPLIT_FILE,
    "comparison": RESULTS,
    "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
out = Path("artifacts/input_config_frozen.json")
out.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

print("===== 입력 설정 동결 =====")
print("파일       : %s" % out)
print("prompt_id  : %s" % FROZEN_PROMPT_ID)
print("OCR 사용   : %s" % ("예" if FROZEN_OCR_ENABLED else "아니오"))
for k, v in FROZEN_MAX_PIXELS.items():
    area = PATCH_AREA.get(k, 32 * 32)
    print("max_pixels : %-22s %d (시각 토큰 %d개)" % (k, v, v // area))
print("받는 사람  : 준혁(27B 실행), 유림(9B 학습)")
print("주의       : 학습이 시작된 뒤에는 이 값을 바꾸지 않습니다.")
print("=========================")
'''))
    return notebook(cells)


# --------------------------------------------------------------------------
# 12_lab_train_9b_qlora
# --------------------------------------------------------------------------
def nb_12():
    cells = []
    cells.append(md("""
# 12. 실습실 9B QLoRA 학습

담당: 유림
목표: 9B 모델의 preflight를 통과시키고, 저장과 재개가 실제로 동작하는지 확인한 뒤, 본 학습을
분리된 프로세스로 띄워서 밤새 돌아가게 만드는 것입니다.

가장 중요한 단계는 3번의 중단과 재개 시험입니다. 여기를 건너뛰면 밤중에 세션이 끊겼을 때
처음부터 다시 학습해야 하는 상황이 생깁니다.

## 선행 조건
- 수빈의 split이 확정되어 있어야 합니다.
- 창혁의 `artifacts/input_config_frozen.json` 이 있으면 그 값에 맞춰 config를 덮어씁니다.

## 통과 기준(PASS)
1. preflight가 20 optimizer step과 50문항 추론을 통과하고 peak VRAM이 16GB 안에 들어옵니다.
2. 20-step smoke 학습이 끝나고 checkpoint가 생성됩니다.
3. 중단한 뒤 같은 명령을 다시 실행했을 때 저장된 step부터 이어지고, loss가 튀지 않습니다.
4. 본 학습 job이 살아 있고 `train_log.jsonl` 에 기록이 쌓입니다.

## 넘겨줄 내용
adapter 경로, 학습 step 수, validation 정확도를 준혁과 수정에게 전달합니다.
"""))

    cells.append(md(CLI_ASSUMPTIONS_MD + """
`python -m vqa.jobs start` 의 `--auto-restart` 는 재시작 횟수를 받는 숫자 인자입니다. 아래
`AUTO_RESTART` 를 0으로 두면 자동 재시작 없이 실행합니다. 재시작은 비정상 종료에만 적용되며,
정상적으로 끝난 작업을 다시 돌리지는 않습니다.
"""))

    cells.append(code(r'''
# ===== 파라미터 =====
PROJECT_ROOT = r"."
DATA_ROOT = r"."
RUN_ROOT = r"./artifacts/runs"
LOCAL_SCRATCH = r"./artifacts/scratch"
MODEL_ROOT = r""

CONFIG = "configs/lab_qwen35_9b.yaml"
SPLIT_FILE = "artifacts/data/split_fold0.json"   # 팀 공식 학습 분할 (train_fold1234, 5,358문항)
VAL_IDS_FILE = "fold_csv_files/val_fold0.csv"    # 팀 공식 검증 분할 (fold 0, 1,344문항)
VAL_OUT_NAME = "fold0_ft"
FROZEN_INPUT = "artifacts/input_config_frozen.json"
FROZEN_KEY = "lab_qwen35_9b"

SMOKE_RUN_ID = "lab9b_smoke"
RESUME_RUN_ID = "lab9b_resume_test"
TRAIN_RUN_ID = "lab9b_ft_v1"

SMOKE_STEPS = 20
SMOKE_LIMIT_TRAIN = 200
RESUME_TEST_STEPS = 60      # 절반쯤에서 끊었다가 이어 붙일 목표 step
RESUME_TEST_WAIT_SEC = 180  # job을 띄운 뒤 강제로 중단하기까지 기다릴 시간

AUTO_RESTART = 3            # 비정상 종료 시 다시 띄울 횟수입니다. 0이면 재시작하지 않습니다.

DEV_TEACHER_RUN_ID = "lab9b_dev_teacher"
DEV_SHARD = "1/4"           # teacher 추론을 여러 PC로 나눌 때 각자 다르게 지정합니다.
''', parameters=True))

    cells.append(code(BOOT))

    cells.append(code(r'''
OVERRIDES = []
if Path(FROZEN_INPUT).exists():
    frozen = json.loads(Path(FROZEN_INPUT).read_text(encoding="utf-8"))
    OVERRIDES += ["--set", "prompt_id=%s" % frozen["prompt_id"]]
    mp = frozen.get("max_pixels", {}).get(FROZEN_KEY)
    if mp:
        OVERRIDES += ["--set", "max_pixels=%d" % mp]
    print("창혁이 동결한 입력 설정을 적용합니다:", OVERRIDES)
else:
    print("아직 동결된 입력 설정이 없습니다. config 기본값으로 진행합니다.")

JOB_EXTRA = ["--auto-restart", str(AUTO_RESTART)] if AUTO_RESTART else []
print("job 추가 옵션:", JOB_EXTRA or "(없음)")
'''))

    cells.append(md("""
## 1. preflight

계획서 3절의 항목을 한 번에 확인합니다. 단일 토큰 매핑, chat template 꼬리, 정답 마스킹, LoRA 대상
목록, 가장 긴 표본에서의 역전파, 속도, peak VRAM이 모두 여기에서 나옵니다.

출력에서 LoRA 대상 목록이 언어 모델 쪽 모듈만 포함하는지 직접 확인하십시오. vision 쪽에 이름이
같은 projection이 섞이면 학습 대상이 잘못 잡힙니다.
"""))

    cells.append(code(r'''
vqa("preflight",
    "--config", CONFIG,
    "--n-train-steps", 20,
    "--n-infer", 50,
    *OVERRIDES)
'''))

    cells.append(md("""
## 2. 20-step smoke 학습

적은 데이터로 학습 경로 전체가 동작하는지 확인합니다. 여기에서 만들어진 adapter는 사용하지 않고
버립니다. 본 학습은 seed를 다시 설정한 새 adapter로 시작합니다.
"""))

    cells.append(code(r'''
vqa("train",
    "--config", CONFIG,
    "--run-id", SMOKE_RUN_ID,
    "--split", SPLIT_FILE,
    "--max-steps", SMOKE_STEPS,
    "--limit-train", SMOKE_LIMIT_TRAIN,
    "--resume", "none",
    *OVERRIDES)

smoke_ckpt = RUN_ROOT_P / SMOKE_RUN_ID / "checkpoints"
print()
print("생성된 checkpoint:")
for p in sorted(smoke_ckpt.glob("step_*")):
    complete = (p / "COMPLETE").exists()
    print("  %s  COMPLETE=%s" % (p.name, complete))
'''))

    cells.append(md("""
## 3. 중단과 재개 시험 (가장 중요한 단계)

실제로 한 번 끊어 보고 이어지는지 확인합니다. 절차는 다음과 같습니다.

1. 분리된 job으로 학습을 시작합니다.
2. 몇 분 기다린 뒤 `jobs stop` 으로 강제 종료합니다.
3. 같은 시작 명령을 다시 실행합니다.
4. `train_log.jsonl` 에서 재개 지점의 step이 이어지고 loss가 매끄럽게 연결되는지 확인합니다.

재개 후 loss가 크게 튀면 optimizer 상태나 scheduler, RNG 중 하나가 복원되지 않은 것입니다.
그 상태로 본 학습을 시작하면 안 됩니다.
"""))

    cells.append(code(ASCII_CURVE + r'''

def read_train_log(run_id):
    path = RUN_ROOT_P / run_id / "train_log.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def log_summary(run_id, tail=8):
    rows = read_train_log(run_id)
    if not rows:
        print("아직 학습 기록이 없습니다:", run_id)
        return rows
    steps = [r.get("step") for r in rows if r.get("step") is not None]
    losses = [r.get("loss") for r in rows if isinstance(r.get("loss"), (int, float))]
    print("기록 %d줄 / step %s ~ %s" % (len(rows), steps[0] if steps else "?", steps[-1] if steps else "?"))
    for r in rows[-tail:]:
        print("  step=%s loss=%s lr=%s" % (r.get("step"), r.get("loss"), r.get("lr")))
    if losses:
        print(ascii_curve(steps[: len(losses)] or list(range(len(losses))), losses))
    return rows


RESUME_TRAIN_CMD = [PY, "-m", "vqa.train",
                    "--config", CONFIG,
                    "--run-id", RESUME_RUN_ID,
                    "--split", SPLIT_FILE,
                    "--max-steps", str(RESUME_TEST_STEPS),
                    "--resume", "auto"] + OVERRIDES
print("재개 시험에 사용할 명령:")
print("  " + " ".join(RESUME_TRAIN_CMD))
'''))

    cells.append(code(r'''
# (1) 분리된 job으로 학습 시작
run([PY, "-m", "vqa.jobs", "start", "--name", RESUME_RUN_ID, "--"] + RESUME_TRAIN_CMD)
'''))

    cells.append(code(r'''
# (2) 몇 분 기다렸다가 상태를 확인하고 강제로 중단합니다.
time.sleep(RESUME_TEST_WAIT_SEC)
run([PY, "-m", "vqa.jobs", "status", "--name", RESUME_RUN_ID], check=False)
run([PY, "-m", "vqa.jobs", "tail", "--name", RESUME_RUN_ID], check=False)

before = log_summary(RESUME_RUN_ID)
STEP_BEFORE_STOP = max([r.get("step", 0) for r in before] or [0])
print()
print("중단 직전 step:", STEP_BEFORE_STOP)

run([PY, "-m", "vqa.jobs", "stop", "--name", RESUME_RUN_ID], check=False)
time.sleep(10)
run([PY, "-m", "vqa.jobs", "status", "--name", RESUME_RUN_ID], check=False)
'''))

    cells.append(code(r'''
# (3) 똑같은 시작 명령을 다시 실행합니다. 수정하는 부분은 없습니다.
run([PY, "-m", "vqa.jobs", "start", "--name", RESUME_RUN_ID, "--"] + RESUME_TRAIN_CMD)
time.sleep(RESUME_TEST_WAIT_SEC)
run([PY, "-m", "vqa.jobs", "status", "--name", RESUME_RUN_ID], check=False)
'''))

    cells.append(code(r'''
# (4) 재개 지점을 확인합니다.
after = log_summary(RESUME_RUN_ID)
steps_after = [r.get("step", 0) for r in after]
STEP_AFTER = max(steps_after or [0])

print()
print("중단 직전 step: %s" % STEP_BEFORE_STOP)
print("재개 후 step  : %s" % STEP_AFTER)

resumed_from_scratch = len([s for s in steps_after if s and s <= 1]) > 1
if STEP_AFTER > STEP_BEFORE_STOP and not resumed_from_scratch:
    print("PASS: 저장된 step부터 이어졌습니다.")
else:
    print("확인 필요: 처음부터 다시 시작했을 가능성이 있습니다. checkpoints/latest.json을 확인하십시오.")

losses = [r.get("loss") for r in after if isinstance(r.get("loss"), (int, float))]
if len(losses) > 4:
    print()
    print("loss 추이가 재개 지점에서 급격히 튀지 않는지 위 곡선에서 확인하십시오.")
'''))

    cells.append(md("""
## 4. 본 학습 시작

`--resume auto` 가 기본이므로, 이 셀을 몇 번 다시 실행해도 처음부터 학습하지 않습니다. 분리된
프로세스로 띄우기 때문에 노트북을 닫아도 계속 돌아갑니다. 다만 PC가 꺼지거나 로그아웃되면
중단되므로, 퇴실 전에 실습실의 전원 정책을 반드시 확인하십시오.
"""))

    cells.append(code(r'''
TRAIN_CMD = [PY, "-m", "vqa.train",
             "--config", CONFIG,
             "--run-id", TRAIN_RUN_ID,
             "--split", SPLIT_FILE,
             "--resume", "auto"] + OVERRIDES

run([PY, "-m", "vqa.jobs", "start", "--name", TRAIN_RUN_ID] + JOB_EXTRA + ["--"] + TRAIN_CMD)
print()
print("학습 job을 띄웠습니다. 아래 감시 셀로 진행 상황을 확인하십시오.")
print("로그 파일:", RUN_ROOT_P / "jobs" / (TRAIN_RUN_ID + ".log"))
'''))

    cells.append(md("""
## 5. 감시

이 셀은 여러 번 실행해도 안전합니다. 상태, 최근 로그, loss 곡선을 한 번에 보여 줍니다.
"""))

    cells.append(code(r'''
run([PY, "-m", "vqa.jobs", "status", "--name", TRAIN_RUN_ID], check=False)
run([PY, "-m", "vqa.jobs", "tail", "--name", TRAIN_RUN_ID], check=False)
print()
log_summary(TRAIN_RUN_ID, tail=10)

latest = RUN_ROOT_P / TRAIN_RUN_ID / "checkpoints" / "latest.json"
if latest.exists():
    print()
    print("latest.json:", latest.read_text(encoding="utf-8")[:400])
'''))

    cells.append(md("""
## 6. 학습 뒤 validation 평가

학습이 끝나면 `final_adapter` 로 고정 validation을 추론하고 정확도를 확인합니다. base보다 낮으면
그 사실을 그대로 보고하십시오. 계획서 8절에 따라 무조건 FT를 제출하지는 않습니다.
"""))

    cells.append(code(r'''
ADAPTER = RUN_ROOT_P / TRAIN_RUN_ID / "final_adapter"
print("adapter:", ADAPTER, "(존재:", ADAPTER.exists(), ")")

vqa("predict",
    "--config", CONFIG,
    "--run-id", TRAIN_RUN_ID,
    "--csv", "train.csv",
    "--ids-file", VAL_IDS_FILE,
    "--adapter", str(ADAPTER),
    "--out-name", VAL_OUT_NAME,
    *OVERRIDES)

Path("artifacts/eval").mkdir(parents=True, exist_ok=True)
script("evaluate_run.py",
       "--pred", str(RUN_ROOT_P / TRAIN_RUN_ID / "preds" / (VAL_OUT_NAME + ".csv")),
       "--gold", str(Path(os.environ["VQA_DATA_ROOT"]) / "train.csv"),
       "--val-ids", VAL_IDS_FILE,
       "--json-out", "artifacts/eval/%s_%s.json" % (TRAIN_RUN_ID, VAL_OUT_NAME),
       "--wrong-out", "artifacts/eval/%s_%s_wrong.csv" % (TRAIN_RUN_ID, VAL_OUT_NAME),
       "--note", "9B QLoRA 미세조정 후 validation")
'''))

    cells.append(md("""
## 7. dev teacher 추론 (의사 라벨 후보 만들기)

미세조정 전 9B base로 `dev.csv` 를 추론합니다. 이 결과와 학생 응답의 다수결이 일치하는 행만
의사 라벨 후보가 됩니다.

계획서 7절의 규칙을 반드시 지켜야 합니다. teacher 프롬프트에는 학생 응답이나 다수결 결과를
절대 넣지 않습니다. 응답을 본 뒤 같은 답을 내도록 유도하면 일치 조건 자체가 의미를 잃습니다.
아래 명령이 `dev.csv` 를 그대로 읽되 `answer1`~`answer5` 컬럼을 입력에 사용하지 않는다는 점은
코어 구현 쪽에서 보장합니다. 여기에서는 `--adapter none` 으로 base 모델임을 명시합니다.

여러 PC로 나눌 때는 `DEV_SHARD` 를 각자 다르게 지정하십시오. GPU를 쓰는 다른 작업과 같은 PC에서
동시에 돌리지 않습니다.
"""))

    cells.append(code(r'''
args = ["--config", CONFIG,
        "--run-id", DEV_TEACHER_RUN_ID,
        "--csv", "dev.csv",
        "--adapter", "none",
        "--out-name", "dev_teacher"] + OVERRIDES
if DEV_SHARD:
    args += ["--shard", DEV_SHARD]
    print("이 PC가 맡은 dev 구간:", DEV_SHARD)

vqa("predict", *args)

print()
print("===== teacher 추론 결과 공유 =====")
print("run_id   : %s" % DEV_TEACHER_RUN_ID)
print("shard    : %s" % (DEV_SHARD or "전체"))
print("출력 폴더: %s" % (RUN_ROOT_P / DEV_TEACHER_RUN_ID / "preds"))
print("다음 담당: 수빈이 다수결과 대조해서 후보를 추리고 100개를 검수합니다.")
print("================================")
'''))
    return notebook(cells)


# --------------------------------------------------------------------------
# 20_colab_27b_setup_preflight
# --------------------------------------------------------------------------
def nb_20():
    cells = []
    cells.append(md("""
# 20. Colab 27B 환경 준비와 사전 점검

담당: 준혁 한 사람만 Colab 런타임을 조작합니다.
목표: 자원을 기록하고, 데이터와 가중치를 준비하고, 버전을 고정하고, 실제 측정값으로 시간과
compute unit 예산을 계산하는 것입니다.

이 노트북은 준비와 측정만 담당합니다. 오래 걸리는 실제 실행은 `21_colab_27b_run.ipynb` 에서
분리된 job으로 띄웁니다.

## 통과 기준(PASS)
1. GPU, VRAM, RAM, 디스크, Drive 여유, 잔여 CU를 모두 기록했습니다.
2. 데이터가 Drive가 아니라 런타임 로컬 디스크에 풀려 있습니다.
3. `RUN_ROOT/env_lock.txt` 에 설치된 버전이 고정되어 있습니다.
4. 27B snapshot을 revision과 함께 받았습니다.
5. preflight가 20 optimizer step과 50문항 추론을 통과했습니다.
6. 예산 계산 결과가 잔여 CU의 80퍼센트 안에 들어옵니다.

## 넘겨줄 내용
마지막 셀의 예산 블록을 팀 채널에 공유합니다. 여기에서 27B가 예산을 통과하지 못하면 9B 경로로
전환해야 하므로, 판단을 미루지 말고 바로 알려야 합니다.
"""))

    cells.append(md(CLI_ASSUMPTIONS_MD))

    cells.append(code(r'''
# ===== 파라미터 =====
PROJECT_ROOT = "/content/ssafy-16-2-ai"          # 코드 저장소를 풀어 둘 위치
REPO_URL = "https://github.com/bnbong/Astralopithecus.git"   # 비공개 저장소입니다.
REPO_BRANCH = "main"
CODE_ZIP = ""                                     # 토큰을 쓸 수 없을 때 사용할 Drive의 코드 zip 경로
DRIVE_MOUNT = "/content/drive"
DRIVE_DIR = "/content/drive/MyDrive/vqa"          # 영속 산출물을 둘 Drive 폴더
ARCHIVE_PATH = "/content/drive/MyDrive/ssafy-16-2-ai.zip"   # 암호가 걸린 데이터 압축 파일
ARCHIVE_PASSWORD = ""                             # 비워 두면 실행할 때 입력받습니다.

LOCAL_SCRATCH = "/content/scratch"
DATA_ROOT = "/content/data"                       # 압축을 풀 로컬 디스크 위치
RUN_ROOT = "/content/drive/MyDrive/vqa/runs"      # checkpoint가 남을 영속 위치
MODEL_ROOT = ""                                   # 아래 셀에서 정해집니다.

MODEL_ID = "Qwen/Qwen3.5-27B"
# 2026-09-21 Colab A100에서 받은 스냅샷 커밋입니다. configs/colab_qwen35_27b_*.yaml의 revision과
# 같은 값을 씁니다.
MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"

# 수동으로 확인해서 적는 값입니다. Colab에는 잔여 CU를 조회하는 API가 없습니다.
CU_REMAINING = 600.0        # 계정 화면에서 본 잔여 compute unit
CU_PER_HOUR = 0.0           # 계정 화면에서 본 현재 GPU의 시간당 소모량. 모르면 0으로 두십시오.
CU_EXPIRY = "미확인"         # 만료일

SPLIT_FILE = "artifacts/data/split_fold0.json"   # 팀 공식 분할 (fold 0 검증)
N_TEST_ITEMS = 6714
N_VAL_ITEMS = 1344          # fold 0 검증 문항 수. split 파일이 있으면 아래에서 덮어씁니다.
N_TRAIN_SAMPLES = 5358      # train_fold1234 표본 수. split 파일이 있으면 아래에서 덮어씁니다.
MICRO_BATCH = 1
GRAD_ACCUM = 8

# 정밀도 정책입니다. 팀 결정은 bf16이며, "auto"는 VRAM만 보고 고르고 "nf4"는 항상 nf4를 씁니다.
PRECISION_POLICY = "bf16"
''', parameters=True))

    cells.append(md("""
## 1. Drive 연결

런타임 조작자는 준혁 한 사람입니다. 다른 팀원이 같은 노트북을 열어도 실행 중인 가상 머신은
공유되지 않으므로, 진행 상황은 Drive에 저장된 로그로 확인해야 합니다.
"""))

    cells.append(code(r'''
from google.colab import drive
drive.mount(DRIVE_MOUNT)
print("Drive를 연결했습니다.")
'''))

    cells.append(md("""
## 1-1. 코드 가져오기

새로 만들어진 Colab 가상 머신에는 저장소가 들어 있지 않습니다. 부트스트랩 셀이 `PROJECT_ROOT` 로
작업 디렉터리를 옮기기 때문에, 그보다 먼저 이 셀로 코드를 내려받아야 합니다.

저장소가 비공개이므로 인증 수단이 필요하며, 다음 두 가지 중 하나를 준비하면 됩니다.

1. GitHub의 Settings에서 이 저장소에만 Contents 읽기 권한을 준 fine-grained personal access
   token을 만듭니다. 그다음 Colab 화면 왼쪽의 열쇠 아이콘 패널을 열고, 이름을 `GITHUB_TOKEN` 으로
   해서 토큰 값을 등록하고 현재 노트북의 접근을 허용합니다. 토큰 값은 화면과 로그에 출력되지
   않으며, 복제가 끝나는 즉시 remote 주소에서 제거하므로 `.git/config` 에도 남지 않습니다.
2. 토큰을 쓰기 어려운 상황이라면 코드 전체를 zip으로 묶어 Drive에 올리고, 파라미터 셀의
   `CODE_ZIP` 에 그 경로를 적습니다. 압축 안에 최상위 폴더가 하나만 있어도 알아서 펼칩니다.

이미 코드를 받아 둔 상태라면 `git pull --ff-only` 로 갱신만 합니다. 갱신이 실패하더라도 멈추지
않고 기존 코드를 그대로 사용합니다.
"""))

    cells.append(code(CODE_FETCH))

    cells.append(code(BOOT))

    cells.append(code(CODE_VERSION_LOG + r'''

# --- 고정 split이 있으면 표본 수를 실제 값으로 맞춥니다 ---
_split_path = Path(SPLIT_FILE)
if _split_path.exists():
    _split = json.loads(_split_path.read_text(encoding="utf-8"))
    N_TRAIN_SAMPLES = len(_split["train_ids"])
    N_VAL_ITEMS = len(_split["val_ids"])
    print("고정 split을 읽어서 표본 수를 갱신했습니다: train %d / val %d"
          % (N_TRAIN_SAMPLES, N_VAL_ITEMS))
else:
    print("%s 이 아직 없어서 파라미터 셀의 기본값을 그대로 씁니다: train %d / val %d"
          % (_split_path, N_TRAIN_SAMPLES, N_VAL_ITEMS))
'''))

    cells.append(md("""
## 2. 자원 기록

GPU 종류와 VRAM, CPU RAM, 로컬 디스크 여유, Drive 여유를 한 번에 기록합니다. 이 값으로 아래에서
bf16과 nf4 중 어느 설정을 쓸지 결정합니다.
"""))

    cells.append(code(r'''
import shutil

RESOURCES = {}

smi = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
           "--format=csv,noheader"], check=False, quiet=True)
if smi["rc"] == 0 and smi["output"].strip():
    first = smi["output"].strip().splitlines()[0]
    parts = [x.strip() for x in first.split(",")]
    RESOURCES["gpu_name"] = parts[0]
    RESOURCES["vram_mib"] = int(parts[1].split()[0])
    RESOURCES["vram_gb"] = round(RESOURCES["vram_mib"] / 1024.0, 1)
    RESOURCES["driver"] = parts[2] if len(parts) > 2 else "?"
else:
    print("nvidia-smi를 사용할 수 없습니다. GPU 런타임인지 확인하십시오.")
    RESOURCES["gpu_name"] = None
    RESOURCES["vram_gb"] = 0.0

try:
    meminfo = Path("/proc/meminfo").read_text()
    total_kb = int([l for l in meminfo.splitlines() if l.startswith("MemTotal")][0].split()[1])
    RESOURCES["ram_gb"] = round(total_kb / 1024.0 / 1024.0, 1)
except Exception as exc:
    RESOURCES["ram_gb"] = None
    print("RAM 정보를 읽지 못했습니다:", exc)

for label, path in (("local_disk", "/content"), ("drive", DRIVE_DIR)):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(path)
        RESOURCES[label + "_free_gb"] = round(usage.free / 1024 ** 3, 1)
        RESOURCES[label + "_total_gb"] = round(usage.total / 1024 ** 3, 1)
    except Exception as exc:
        print("%s 용량을 확인하지 못했습니다: %s" % (label, exc))

for k, v in RESOURCES.items():
    print("%-18s: %s" % (k, v))
'''))

    cells.append(md("""
## 3. 잔여 compute unit 기록

Colab은 잔여 CU를 조회하는 API를 제공하지 않습니다. 화면 오른쪽 위의 자원 표시나 구독 화면에서
직접 확인한 값을 파라미터 셀에 적고, 아래 셀로 기록을 남기십시오. 시간당 소모량을 모르면
30분 정도 실행한 뒤 줄어든 양을 보고 역산합니다.
"""))

    cells.append(code(r'''
if CU_PER_HOUR <= 0:
    print("시간당 CU 소모량이 아직 입력되지 않았습니다.")
    print("계산용 가정값 7.5 CU/h를 임시로 사용하되, 실제 값을 확인해서 파라미터 셀을 고치십시오.")
    CU_RATE = 7.5
    CU_RATE_SOURCE = "가정값"
else:
    CU_RATE = CU_PER_HOUR
    CU_RATE_SOURCE = "계정 화면에서 확인"

usable_cu = CU_REMAINING * 0.8
print("잔여 CU        : %.1f (%s 만료)" % (CU_REMAINING, CU_EXPIRY))
print("시간당 소모     : %.2f CU/h (%s)" % (CU_RATE, CU_RATE_SOURCE))
print("20%% 예비를 뺀 가용: %.1f CU = 약 %.1f GPU시간" % (usable_cu, usable_cu / CU_RATE))

RESOURCES["cu_remaining"] = CU_REMAINING
RESOURCES["cu_per_hour"] = CU_RATE
RESOURCES["cu_rate_source"] = CU_RATE_SOURCE
RESOURCES["cu_expiry"] = CU_EXPIRY
'''))

    cells.append(md("""
## 4. 데이터를 로컬 디스크로 옮기기

Drive를 그대로 읽으면 작은 파일을 반복해서 읽을 때 매우 느립니다. 압축 파일을 런타임 로컬
디스크에 풀어 놓고, 학습과 추론은 로컬 경로만 사용합니다. 이미 풀려 있으면 건너뜁니다.

압축 파일에 암호가 걸려 있지 않다면 파라미터 셀의 `ARCHIVE_PASSWORD` 에 `none` 이라고 적으십시오.
그러면 입력을 받지 않고 그대로 풉니다. 압축 안에 최상위 폴더가 하나만 있는 경우에는 그 폴더를
데이터 위치로 삼습니다.
"""))

    cells.append(code(DATA_STAGE + r'''

for name in ("train.csv", "test.csv", "dev.csv", "sample_submission.csv"):
    p = data_root / name
    print("%-24s %s" % (name, "있음" if p.exists() else "없음"))
for name in ("train", "test", "dev"):
    p = data_root / name
    n = len(list(p.glob("*.jpg"))) if p.is_dir() else 0
    print("%-24s 이미지 %d장" % (name + "/", n))

print("VQA_DATA_ROOT:", os.environ["VQA_DATA_ROOT"])
'''))

    cells.append(md("""
## 5. 라이브러리 설치와 버전 고정

`requirements/colab.txt`에 고정된 버전을 설치한 다음, 별도 프로세스에서 실제로 import 해 보고
버전과 GPU를 확인합니다. 지금 커널이 예전 버전을 이미 import 해 두었을 수 있어서, 같은 커널
안에서 확인하면 옛 버전이 그대로 보이기 때문입니다.

Colab 기본 이미지에 들어 있는 torchao 0.10.0은 PEFT가 bf16 Linear에 LoRA를 붙일 때 충돌하므로
설치 전에 제거합니다. 21번 노트북은 가상 머신이 다르기 때문에 같은 셀을 거기에서도 다시
실행해야 합니다.

설치가 끝나면 곧바로 `pip freeze` 결과를 Drive에 저장합니다. 가상 머신이 재활용되어도 같은
환경을 다시 만들 수 있어야 하기 때문입니다.
"""))

    cells.append(code(ENV_SETUP))

    cells.append(code(r'''
freeze = run([PY, "-m", "pip", "freeze"], quiet=True)
lock_path = RUN_ROOT_P / "env_lock.txt"
lock_path.write_text(freeze["output"], encoding="utf-8")
print("버전을 고정했습니다:", lock_path)
print("총 %d개 패키지" % len(freeze["output"].splitlines()))
for line in freeze["output"].splitlines():
    if line.lower().startswith(("torch", "transformers", "peft", "accelerate", "bitsandbytes")):
        print("  " + line)
'''))

    cells.append(md("""
## 6. 27B 가중치 받기

Drive 여유 공간이 충분하면 Drive의 캐시에 받아 둡니다. 가상 머신이 바뀌어도 다시 받지 않아도
되기 때문입니다. 여유가 부족하면 로컬 디스크에 받습니다. 이 경우 런타임이 재활용될 때마다
다시 받아야 하며, 회선 상태에 따라 수십 분이 걸릴 수 있습니다.

revision을 `main` 으로 두지 말고 실제 commit 해시로 고정하는 편이 재현에 유리합니다.
"""))

    cells.append(code(r'''
NEEDED_GB = 60.0   # 27B 원본 가중치에 필요한 대략적인 여유 공간
drive_free = RESOURCES.get("drive_free_gb", 0.0)
local_free = RESOURCES.get("local_disk_free_gb", 0.0)

if drive_free >= NEEDED_GB:
    CACHE_DIR = str(Path(DRIVE_DIR) / "hf_cache")
    CACHE_NOTE = "Drive 캐시에 받았습니다. 런타임이 바뀌어도 다시 받지 않아도 됩니다."
else:
    CACHE_DIR = "/content/hf_cache"
    CACHE_NOTE = ("Drive 여유가 %.1fGB뿐이라 로컬 디스크에 받았습니다. "
                  "가상 머신이 재활용되면 다시 받아야 하며 수십 분이 걸릴 수 있습니다." % drive_free)

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["VQA_MODEL_ROOT"] = CACHE_DIR
print("캐시 위치:", CACHE_DIR)
print(CACHE_NOTE)
print("로컬 디스크 여유: %.1fGB / Drive 여유: %.1fGB" % (local_free, drive_free))
'''))

    cells.append(code(r'''
os.environ["VQA_MODEL_ID"] = MODEL_ID
os.environ["VQA_MODEL_REV"] = MODEL_REVISION
print("모델 식별자와 revision을 환경 변수에 넣었습니다.")
'''))

    cells.append(code(r'''
hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
if hf_bin:
    dl = run([hf_bin, "download", MODEL_ID, "--revision", MODEL_REVISION,
              "--cache-dir", CACHE_DIR], check=False)
else:
    print("hf 명령을 찾지 못해 python 모듈로 받습니다.")
    dl = run([PY, "-c",
              "import os, huggingface_hub as h;"
              "p=h.snapshot_download(os.environ['VQA_MODEL_ID'],"
              " revision=os.environ['VQA_MODEL_REV'], cache_dir=os.environ['HF_HOME']);"
              "print('snapshot:', p)"],
             check=False)

MODEL_SNAPSHOT_NOTE = {
    "model_id": MODEL_ID,
    "revision": MODEL_REVISION,
    "cache_dir": CACHE_DIR,
    "download_rc": dl["rc"],
    "download_seconds": round(dl["seconds"], 1),
}
(RUN_ROOT_P / "model_snapshot.json").write_text(
    json.dumps(MODEL_SNAPSHOT_NOTE, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(MODEL_SNAPSHOT_NOTE, ensure_ascii=False, indent=2))
if dl["rc"] != 0:
    print()
    print("다운로드가 실패했습니다. 네트워크 상태와 남은 디스크 용량을 확인한 뒤 이 셀을 다시 실행하십시오.")
    print("이미 받은 부분은 캐시에 남아 있으므로 처음부터 다시 받지 않습니다.")
'''))

    cells.append(md("""
## 7. VRAM으로 config 고르기

이 노트북은 지금 배정받은 GPU 기준으로 설정을 고릅니다.

| 감지된 VRAM | 사용할 설정 |
|---|---|
| 70GB 이상 | `colab_qwen35_27b_bf16.yaml` |
| 35GB 이상 70GB 미만 | `colab_qwen35_27b_nf4.yaml` |
| 35GB 미만 | 27B를 포기하고 9B 경로로 전환합니다. |

세 번째 경우에는 아래 셀이 오류를 내면서 멈춥니다. 그때는 팀에 알리고 실습실 9B 경로에
인력을 더 투입해야 합니다.

여기에서 고른 값은 이 가상 머신에서 측정할 때만 쓰는 값입니다. 21번 노트북은 자기 가상 머신의
VRAM을 다시 재고 `PRECISION_POLICY`를 함께 보기 때문에, 40GB에서 고른 nf4가 80GB 가상 머신에
그대로 따라가지 않습니다. 그래서 아래 셀은 감지한 VRAM과 정책도 함께 기록합니다.
"""))

    cells.append(code(r'''
vram = RESOURCES.get("vram_gb", 0.0)
if vram >= 70:
    CONFIG = "configs/colab_qwen35_27b_bf16.yaml"
    PRECISION = "bf16"
elif vram >= 35:
    CONFIG = "configs/colab_qwen35_27b_nf4.yaml"
    PRECISION = "nf4"
else:
    raise RuntimeError(
        "감지된 VRAM이 %.1fGB입니다. 27B 경로를 포기하고 9B 경로로 전환해야 합니다. "
        "팀 채널에 즉시 알리고 docs/README_team.md의 대체 경로를 따르십시오." % vram
    )

print("감지된 VRAM %.1fGB -> %s (%s)" % (vram, CONFIG, PRECISION))
print("config 존재:", Path(CONFIG).exists())
if PRECISION_POLICY == "bf16" and PRECISION != "bf16":
    print("팀 정책은 bf16입니다. 이 가상 머신에서는 VRAM이 모자라 nf4로 측정합니다.")
    print("본 실행은 80GB A100을 받은 뒤에 21번 노트북에서 시작하십시오.")

# 21번 노트북이 판단할 때 참고하도록 선택 결과와 근거를 함께 남깁니다.
COLAB_CHOICE = {
    "config": CONFIG,
    "precision": PRECISION,
    "policy": PRECISION_POLICY,
    "vram_gb": vram,
    "gpu_name": RESOURCES.get("gpu_name"),
    "code_version": CODE_VERSION,
    "decided_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "note": "이 값은 위 VRAM에서 고른 것입니다. 21번 노트북은 자기 가상 머신에서 다시 고릅니다.",
}
(RUN_ROOT_P / "colab_choice.json").write_text(
    json.dumps(COLAB_CHOICE, ensure_ascii=False, indent=2), encoding="utf-8")
print("선택 결과를 저장했습니다:", RUN_ROOT_P / "colab_choice.json")
'''))

    cells.append(md("""
## 8. preflight

20 optimizer step과 50문항 추론을 측정합니다. 누적이 8이면 20 optimizer step은 160
microbatch입니다. preflight는 먼저 가장 긴 표본 몇 개로 OOM을 보고, 그 다음 무작위 표본
(최대 `--max-timing-microbatches`개)으로 대표 속도를 잽니다. 가장 긴 표본만으로 잰 시간은
상한이라 예산에 쓰면 안 됩니다.

출력에서 `sec_per_microbatch_mean`(초/microbatch)과 `sec_per_optimizer_step`(초/optimizer
step)을 구분해서 확인하십시오. 예산 계산에 들어가는 값은 뒤쪽입니다.

측정한 속도가 다른 환경에서 보고된 값보다 느리다면 아래 셀의 preflight 호출에
`--tf32-ab` 옵션을 덧붙여서 다시 실행하십시오. 이 옵션을 지정하면 동일한 표본으로 TF32를
끈 상태와 켠 상태의 microbatch 시간을 각각 측정하여 두 평균을 함께 출력합니다.
"""))

    cells.append(code(r'''
OVERRIDES = []
frozen_path = Path("artifacts/input_config_frozen.json")
if frozen_path.exists():
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    OVERRIDES += ["--set", "prompt_id=%s" % frozen["prompt_id"]]
    mp = frozen.get("max_pixels", {}).get("colab_qwen35_27b")
    if mp:
        OVERRIDES += ["--set", "max_pixels=%d" % mp]
    print("창혁이 동결한 입력 설정을 적용합니다:", OVERRIDES)

pf = vqa("preflight",
         "--config", CONFIG,
         "--n-train-steps", 20,
         "--n-infer", 50,
         "--max-timing-microbatches", 64,
         *OVERRIDES,
         check=False)
'''))

    cells.append(code(r'''
# preflight 결과 JSON에서 측정값을 읽습니다.
pf_dir = RUN_ROOT_P / "preflight"
PF = {}
if pf_dir.is_dir():
    files = sorted(pf_dir.glob("*.json"))
    if files:
        PF = json.loads(files[-1].read_text(encoding="utf-8"))
        print("preflight 파일:", files[-1].name)

def pick(node, *keys):
    """preflight 결과는 검사별로 중첩되어 있으므로 키를 재귀로 찾습니다."""
    if isinstance(node, dict):
        for k in keys:
            if k in node:
                return node[k]
        for v in node.values():
            got = pick(v, *keys)
            if got is not None:
                return got
    elif isinstance(node, list):
        for v in node:
            got = pick(v, *keys)
            if got is not None:
                return got
    return None


SEC_PER_ITEM = pick(PF, "seconds_per_item", "sec_per_item")
PEAK_VRAM = pick(PF, "peak_vram_gb", "peak_memory_gb")

# 단위 주의: preflight의 forward+backward는 microbatch 1개(표본 MICRO_BATCH개)를 잽니다.
# optimizer step 1개는 microbatch GRAD_ACCUM개이므로, 예산에는 초/optimizer step을 써야 합니다.
SEC_PER_OPT_STEP = pick(PF, "sec_per_optimizer_step")
SEC_PER_MICRO = pick(PF, "sec_per_microbatch_mean", "warm_seconds_mean")

if SEC_PER_OPT_STEP is not None:
    SEC_PER_STEP = SEC_PER_OPT_STEP
elif SEC_PER_MICRO is not None:
    SEC_PER_STEP = SEC_PER_MICRO * GRAD_ACCUM
    print("preflight가 microbatch 단위 값만 남겼습니다. 누적 %d를 곱해 optimizer step으로 바꿉니다."
          % GRAD_ACCUM)
else:
    # 참고 실행(A100-80GB, 27B bf16)의 3.37초는 microbatch 1개 기준입니다. 여기서만 누적을 곱합니다.
    SEC_PER_MICRO = 3.37
    SEC_PER_STEP = SEC_PER_MICRO * GRAD_ACCUM
    print("학습 속도를 읽지 못해 참고 수치 3.37초/microbatch를 사용합니다. 실제 측정값으로 바꾸십시오.")

if SEC_PER_ITEM is None:
    SEC_PER_ITEM = 0.55
    print("추론 속도를 읽지 못해 참고 수치 0.55초/문항을 사용합니다. 실제 측정값으로 바꾸십시오.")

print("문항당 추론    : %.3f초/문항" % SEC_PER_ITEM)
if SEC_PER_MICRO is not None:
    print("microbatch     : %.2f초/microbatch" % SEC_PER_MICRO)
print("optimizer step : %.2f초/optimizer step (microbatch %d개)" % (SEC_PER_STEP, GRAD_ACCUM))
print("peak VRAM      : %s GB" % PEAK_VRAM)
'''))

    cells.append(md("""
## 9. 시간과 CU 예산

실측값으로 base validation, base 전체 test, 학습, FT validation, FT 전체 test를 모두 더합니다.
여기에 20퍼센트 예비를 붙인 값이 잔여 CU 안에 들어와야 27B 경로를 유지할 수 있습니다.

단위를 반드시 구분하십시오. preflight가 재는 것은 **초/microbatch**이고, 학습 시간 계산에
들어가는 것은 **초/optimizer step**입니다. 둘은 누적(GRAD_ACCUM, 기본 8)배만큼 차이가 납니다. 앞 셀에서
`sec_per_optimizer_step`을 그대로 읽고, 그 키가 없을 때만 microbatch 값에 누적을 곱합니다.

예산을 넘기면 계획서 8절의 예외를 적용합니다. 9B 전체 base를 조기 제출로 쓰고, 27B는
validation과 FT 전체 추론만 담당하도록 줄입니다. 이 결정은 실행을 시작하기 전에 내려야 합니다.
"""))

    cells.append(code(r'''
steps = max(1, int(N_TRAIN_SAMPLES / (MICRO_BATCH * GRAD_ACCUM)))

plan = [
    ("base validation 추론", N_VAL_ITEMS * SEC_PER_ITEM / 3600.0),
    ("base 전체 test 추론", N_TEST_ITEMS * SEC_PER_ITEM / 3600.0),
    ("학습 1 epoch (%d step)" % steps, steps * SEC_PER_STEP / 3600.0),
    ("FT validation 추론", N_VAL_ITEMS * SEC_PER_ITEM / 3600.0),
    ("FT 전체 test 추론", N_TEST_ITEMS * SEC_PER_ITEM / 3600.0),
]
subtotal = sum(h for _, h in plan)
reserve = subtotal * 0.2
total = subtotal + reserve

print("%-28s %8s %8s" % ("작업", "시간(h)", "CU"))
print("-" * 48)
for name, hours in plan:
    print("%-28s %8.2f %8.1f" % (name, hours, hours * CU_RATE))
print("-" * 48)
print("%-28s %8.2f %8.1f" % ("소계", subtotal, subtotal * CU_RATE))
print("%-28s %8.2f %8.1f" % ("20% 예비", reserve, reserve * CU_RATE))
print("%-28s %8.2f %8.1f" % ("합계", total, total * CU_RATE))
print()
print("잔여 CU %.1f 중 %.1f를 사용하게 됩니다." % (CU_REMAINING, total * CU_RATE))

if total * CU_RATE > CU_REMAINING * 0.8:
    print()
    print("경고: 예산을 초과합니다. 계획서 8절의 예외를 적용할지 지금 결정해야 합니다.")
    print("  - 9B 전체 base 결과를 조기 제출로 사용합니다.")
    print("  - 27B는 validation과 FT 전체 추론만 담당합니다.")
else:
    print()
    print("PASS: 잔여 CU의 80퍼센트 안에 들어옵니다.")

budget = {"resources": RESOURCES, "config": CONFIG, "precision": PRECISION,
          "sec_per_item": SEC_PER_ITEM,
          "sec_per_microbatch": SEC_PER_MICRO,
          "grad_accum": GRAD_ACCUM,
          "sec_per_optimizer_step": SEC_PER_STEP,
          "sec_per_step": SEC_PER_STEP,  # 예전 이름. optimizer step 기준입니다.
          "peak_vram_gb": PEAK_VRAM, "plan_hours": dict(plan),
          "total_hours": round(total, 2), "total_cu": round(total * CU_RATE, 1)}
(RUN_ROOT_P / "budget.json").write_text(
    json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("예산을 저장했습니다:", RUN_ROOT_P / "budget.json")
'''))

    cells.append(code(r'''
print("===== Colab 준비 결과 공유 =====")
print("GPU         : %s / VRAM %.1fGB" % (RESOURCES.get("gpu_name"), RESOURCES.get("vram_gb", 0)))
print("RAM         : %s GB" % RESOURCES.get("ram_gb"))
print("로컬 디스크 : 여유 %.1fGB" % RESOURCES.get("local_disk_free_gb", 0))
print("Drive       : 여유 %.1fGB" % RESOURCES.get("drive_free_gb", 0))
print("잔여 CU     : %.1f (%.2f CU/h, %s)" % (CU_REMAINING, CU_RATE, CU_RATE_SOURCE))
print("config      : %s (%s)" % (CONFIG, PRECISION))
print("model       : %s @ %s" % (MODEL_ID, MODEL_REVISION))
print("가중치 캐시 : %s" % CACHE_DIR)
print("실측        : %.3f초/문항, %.2f초/optimizer step (= microbatch %d개)"
      % (SEC_PER_ITEM, SEC_PER_STEP, GRAD_ACCUM))
print("예산        : %.2f시간 / %.1f CU (20%% 예비 포함)" % (total, total * CU_RATE))
print("다음 단계   : 21_colab_27b_run.ipynb 에서 base validation부터 시작합니다.")
print("===============================")
'''))
    return notebook(cells)


# --------------------------------------------------------------------------
# 21_colab_27b_run
# --------------------------------------------------------------------------
def nb_21():
    cells = []
    cells.append(md("""
# 21. Colab 27B 본 실행

담당: 준혁
목표: 오래 걸리는 작업을 전부 분리된 job으로 띄워서, 셀을 멈추거나 브라우저를 닫아도 계속
돌아가게 만드는 것입니다.

## Colab에서 실제로 보장되는 범위

정확히 알고 있어야 사고를 막을 수 있으므로, 사실 관계를 먼저 정리합니다.

분리된 프로세스가 살아남는 경우는 다음과 같습니다.
- 노트북 셀을 중단했을 때
- 브라우저 탭을 닫았을 때
- 노트북 화면과 런타임 사이의 연결이 일시적으로 끊겼을 때

분리된 프로세스도 함께 죽는 경우는 다음과 같습니다.
- 유휴 시간 초과로 런타임이 회수될 때
- 최대 연속 실행 시간에 도달했을 때
- compute unit이 소진되었을 때
- 자원 사정으로 인스턴스가 선점될 때
- 사용자가 런타임을 직접 종료하거나 다시 시작할 때

Pro+ 구독은 백그라운드 실행을 제공한다고 안내되어 있습니다. 다만 실제 계정과 실제 런타임에서
어떻게 동작하는지는 D1에 직접 확인해야 합니다. 아래 0번 절에 확인 절차를 적어 두었습니다.

세션을 유지하겠다고 자동 클릭 스크립트 같은 방법을 쓰지 않습니다. Colab 이용 약관에 어긋나고,
계정이 제한되면 팀 전체의 GPU 경로가 사라집니다. 대신 checkpoint를 Drive에 남겨서 언제 끊기더라도
이어서 실행할 수 있게 만드는 방식으로 대응합니다.
"""))

    cells.append(md("""
## 0. 재접속 절차 (연결이 끊겼다면 여기부터)

브라우저를 닫았다가 돌아왔거나 연결이 끊겼다면 아래 순서대로 진행하십시오.

1. 파라미터 셀과 Drive 연결 셀, 코드 가져오기 셀, 부트스트랩 셀을 실행합니다.
2. 0-1절의 환경 설치와 확인 셀을 실행합니다. 가상 머신이 바뀌었다면 패키지가 하나도 없습니다.
3. `jobs status` 셀을 실행해서 프로세스가 살아 있는지 확인합니다.
4. 살아 있으면 감시 셀만 실행하고 기다립니다. 추가로 할 일은 없습니다.
5. 프로세스가 없고 `/content` 의 데이터도 사라졌다면 가상 머신이 회수된 것입니다. 데이터 준비
   셀을 다시 실행한 다음, **원래 실행했던 시작 셀을 그대로 다시 실행합니다.** 명령을 고칠 필요는
   없습니다. Drive에 남은 checkpoint에서 이어집니다.
"""))

    cells.append(code(r'''
# ===== 파라미터 =====
PROJECT_ROOT = "/content/ssafy-16-2-ai"
REPO_URL = "https://github.com/bnbong/Astralopithecus.git"   # 비공개 저장소입니다.
REPO_BRANCH = "main"
CODE_ZIP = ""                                     # 토큰을 쓸 수 없을 때 사용할 Drive의 코드 zip 경로
DRIVE_MOUNT = "/content/drive"
DRIVE_DIR = "/content/drive/MyDrive/vqa"
ARCHIVE_PATH = "/content/drive/MyDrive/ssafy-16-2-ai.zip"
ARCHIVE_PASSWORD = ""

LOCAL_SCRATCH = "/content/scratch"
DATA_ROOT = "/content/data"
RUN_ROOT = "/content/drive/MyDrive/vqa/runs"
MODEL_ROOT = "/content/drive/MyDrive/vqa/hf_cache"   # model_snapshot.json이 있으면 덮어씁니다.

# 정밀도 정책입니다. 팀 결정은 bf16입니다. "auto"는 VRAM만 보고 고르고, "nf4"는 항상 nf4를 씁니다.
PRECISION_POLICY = "bf16"
CONFIG = "configs/colab_qwen35_27b_bf16.yaml"   # 아래 셀이 현재 GPU와 정책을 보고 다시 정합니다.
BASE_RUN_ID = "colab27b_base_bf16"
FT_RUN_ID = "colab27b_ft_bf16"
SPLIT_FILE = "artifacts/data/split_fold0.json"     # 팀 공식 학습 분할 (train_fold1234, 5,358문항)
VAL_IDS_FILE = "fold_csv_files/val_fold0.csv"      # 팀 공식 검증 분할 (fold 0, 1,344문항)
VAL_OUT_BASE = "fold0_base"
VAL_OUT_FT = "fold0_ft"
N_TEST_ITEMS = 6714

AUTO_RESTART = 3                  # 비정상 종료 시 다시 띄울 횟수입니다. 0이면 재시작하지 않습니다.
ALSO_BASE = True                  # FT 추론과 같은 forward에서 base 확률도 함께 남깁니다.

MONITOR_MINUTES = 30              # 감시 셀이 돌 시간
MONITOR_INTERVAL_SEC = 60
''', parameters=True))

    cells.append(code(r'''
from google.colab import drive
drive.mount(DRIVE_MOUNT)
print("Drive를 연결했습니다.")
'''))

    cells.append(md("""
### 0-0. 코드 가져오기

가상 머신이 회수되면 `/content` 아래가 전부 사라지므로 코드도 함께 사라집니다. 부트스트랩 셀이
`PROJECT_ROOT` 로 작업 디렉터리를 옮기기 전에 이 셀로 코드를 먼저 확보해야 합니다. 인증 준비
방법은 `20_colab_27b_setup_preflight.ipynb` 의 1-1절에 정리되어 있습니다. 요약하면 이 저장소에만
Contents 읽기 권한을 준 fine-grained personal access token을 Colab 비밀 `GITHUB_TOKEN` 으로
등록하거나, 코드 zip을 Drive에 올리고 `CODE_ZIP` 에 경로를 적으면 됩니다.

주의할 점이 하나 있습니다. 이미 돌고 있는 job은 이미 읽어 들인 코드로 계속 실행되므로, 지금
`git pull` 을 해도 그 job은 영향을 받지 않습니다. 다만 코드나 config가 바뀐 상태에서 job을 다시
띄우면 `--resume auto` 가 config 해시 불일치를 이유로 재개를 거부할 수 있습니다. 따라서 코드
갱신은 job과 job 사이, 즉 실행 중인 작업이 없을 때에만 하십시오.
"""))

    cells.append(code(CODE_FETCH))

    cells.append(code(BOOT))

    cells.append(code(CODE_VERSION_LOG + r'''

# --- 20번 노트북이 내린 결정을 그대로 따릅니다 ---
_snap_path = RUN_ROOT_P / "model_snapshot.json"
if _snap_path.exists():
    _snap = json.loads(_snap_path.read_text(encoding="utf-8"))
    _cache_dir = _snap.get("cache_dir") or ""
    if _cache_dir and Path(_cache_dir).is_dir():
        MODEL_ROOT = _cache_dir
        print("20번 노트북이 쓴 가중치 캐시를 그대로 사용합니다:", MODEL_ROOT)
    else:
        MODEL_ROOT = "/content/hf_cache"
        print("20번 노트북이 기록한 캐시 %s 가 이 런타임에는 없습니다." % (_cache_dir or "(미기록)"))
        print("로컬 캐시 %s 를 대신 사용합니다. 가중치를 다시 내려받으므로 수십 분이 걸릴 수 있습니다."
              % MODEL_ROOT)
else:
    print("model_snapshot.json이 없습니다. 파라미터 셀의 MODEL_ROOT를 그대로 사용합니다.")

MODEL_ROOT_P = Path(MODEL_ROOT)
MODEL_ROOT_P.mkdir(parents=True, exist_ok=True)
os.environ["VQA_MODEL_ROOT"] = str(MODEL_ROOT_P)
os.environ["HF_HOME"] = str(MODEL_ROOT_P)

# --- 지금 이 가상 머신의 VRAM을 직접 재서 정밀도를 정합니다 ---
# 20번 노트북의 colab_choice.json은 그 가상 머신의 GPU로 고른 값이라서 그대로 따르지 않습니다.
# 오늘 40GB에서 고른 nf4가 80GB 가상 머신까지 따라와서 조용히 nf4로 돌아간 사고가 있었습니다.
BF16_CONFIG = "configs/colab_qwen35_27b_bf16.yaml"
NF4_CONFIG = "configs/colab_qwen35_27b_nf4.yaml"

VRAM_GB = 0.0
_smi = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
           check=False, quiet=True, echo=False)
GPU_NAME = None
if _smi["rc"] == 0 and _smi["output"].strip():
    _parts = [x.strip() for x in _smi["output"].strip().splitlines()[0].split(",")]
    GPU_NAME = _parts[0]
    VRAM_GB = round(int(_parts[1].split()[0]) / 1024.0, 1)
else:
    print("nvidia-smi를 쓸 수 없습니다. 런타임 유형이 GPU인지 확인하십시오.")

RUN_SUFFIX = ""
TRAINING_ALLOWED = True

if PRECISION_POLICY == "nf4":
    CONFIG, PRECISION = NF4_CONFIG, "nf4"
elif PRECISION_POLICY == "auto":
    if VRAM_GB >= 70:
        CONFIG, PRECISION = BF16_CONFIG, "bf16"
    else:
        CONFIG, PRECISION = NF4_CONFIG, "nf4"
elif PRECISION_POLICY == "bf16":
    if VRAM_GB >= 70:
        CONFIG, PRECISION = BF16_CONFIG, "bf16"
    else:
        CONFIG, PRECISION = NF4_CONFIG, "nf4"
        RUN_SUFFIX = "_nf4"
        TRAINING_ALLOWED = False
        print("=" * 72)
        print("경고: 팀 정책은 bf16인데 이 GPU의 VRAM은 %.1fGB뿐입니다." % VRAM_GB)
        print("이 가상 머신에서는 bf16 학습을 시작할 수도, 이어서 할 수도 없습니다.")
        print("80GB A100을 받으려면 런타임을 끊고 다시 연결하십시오.")
        print("계속 진행하면 추론만 nf4로 돌리며, run-id 뒤에 \"_nf4\"를 붙여서")
        print("bf16 결과와 캐시가 절대 섞이지 않게 합니다.")
        print("=" * 72)
else:
    raise RuntimeError(
        "PRECISION_POLICY 는 bf16, auto, nf4 중 하나여야 합니다: %r" % (PRECISION_POLICY,))

# 이 셀을 여러 번 실행해도 접미사가 겹쳐 붙지 않도록 먼저 떼어 내고 다시 붙입니다.
def _with_suffix(run_id):
    if run_id.endswith("_nf4"):
        run_id = run_id[: -len("_nf4")]
    return run_id + RUN_SUFFIX


BASE_RUN_ID = _with_suffix(BASE_RUN_ID)
FT_RUN_ID = _with_suffix(FT_RUN_ID)

print("GPU              : %s / VRAM %.1fGB" % (GPU_NAME, VRAM_GB))
print("정밀도 정책      : %s -> %s" % (PRECISION_POLICY, PRECISION))
print("적용 중인 config : %s (존재: %s)" % (CONFIG, Path(CONFIG).exists()))
print("base run-id      :", BASE_RUN_ID)
print("ft run-id        :", FT_RUN_ID)
print("학습 허용        :", TRAINING_ALLOWED)
print("가중치 캐시      :", os.environ["VQA_MODEL_ROOT"])

_choice_path = RUN_ROOT_P / "colab_choice.json"
if _choice_path.exists():
    _choice = json.loads(_choice_path.read_text(encoding="utf-8"))
    print("참고: 20번 노트북은 VRAM %s GB에서 %s 를 골랐습니다 (정책 %s)."
          % (_choice.get("vram_gb"), _choice.get("config"), _choice.get("policy", "미기록")))
'''))

    cells.append(md("""
### 0-1. 환경 설치와 확인

Colab은 노트북 하나마다 가상 머신을 하나씩 따로 배정합니다. 그래서 20번 노트북에서 설치를
끝냈더라도 이 노트북의 가상 머신에는 아무 패키지도 깔려 있지 않습니다. 오늘 27B 작업이
`Using bitsandbytes 4-bit quantization requires bitsandbytes` 오류로 세 번 죽은 원인이 바로
이것입니다.

작업을 시작하기 전에 런타임 메뉴의 세션 관리 화면에서 다른 노트북의 세션을 반드시 종료하십시오.
A100 두 대가 동시에 돌면 compute unit이 두 배로 빠지고, 80GB를 받을 확률도 떨어집니다. 새
가상 머신을 받은 다음에는 0번 절을 코드 가져오기부터 데이터 준비와 설치까지 빠짐없이 실행한
뒤에 job을 시작하십시오.
"""))

    cells.append(code(ENV_SETUP))

    cells.append(code(r'''
JOB_EXTRA =["--auto-restart", str(AUTO_RESTART)] if AUTO_RESTART else []
ALSO_BASE_ARGS = ["--also-base"] if ALSO_BASE else []

OVERRIDES = []
frozen_path = Path("artifacts/input_config_frozen.json")
if frozen_path.exists():
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    OVERRIDES += ["--set", "prompt_id=%s" % frozen["prompt_id"]]
    mp = frozen.get("max_pixels", {}).get("colab_qwen35_27b")
    if mp:
        OVERRIDES += ["--set", "max_pixels=%d" % mp]

JOBS = {
    "base_val": None, "base_test": None, "train": None, "ft_val": None, "ft_test": None,
}

def job_start(name, cmd, extra=True):
    args = [PY, "-m", "vqa.jobs", "start", "--name", name]
    if extra:
        args += JOB_EXTRA
    args += ["--"] + [str(c) for c in cmd]
    return run(args, check=False)


def job_status(name):
    res = run([PY, "-m", "vqa.jobs", "status", "--name", name],
              check=False, quiet=True, hint=False)
    out = res["output"].strip()
    if res["rc"] == 0:
        print(out)
    elif "기록 없음" in out:
        print("job %s: 아직 시작하지 않았습니다." % name)
    else:
        print(out)
        print("실패했습니다. 같은 명령에 --help 를 붙여서 인자 이름부터 확인하십시오.", flush=True)
    return res


print("config    :", CONFIG)
print("job 옵션  :", JOB_EXTRA or "(없음)")
print("입력 설정 :", OVERRIDES or "(config 기본값)")
'''))

    cells.append(md("""
### 0-2. 현재 job 상태 확인

재접속했다면 이 셀부터 실행하십시오. 아직 실행하지 않은 뒷부분 섹션의 job은 "아직 시작하지
않았습니다"라고 나오는 것이 정상이니 걱정하지 않아도 됩니다.
"""))

    cells.append(code(r'''
JOB_NAMES = [BASE_RUN_ID + "_val", BASE_RUN_ID + "_test",
             FT_RUN_ID + "_train", FT_RUN_ID + "_val", FT_RUN_ID + "_test"]
for name in JOB_NAMES:
    print("=" * 60)
    print("job:", name)
    job_status(name)

data_ok = (Path(DATA_ROOT) / "train.csv").exists()
print()
print("로컬 데이터 존재:", data_ok)
if not data_ok:
    print("가상 머신이 회수된 것으로 보입니다. 아래 데이터 준비 셀을 먼저 실행하십시오.")
'''))

    cells.append(md("""
### 0-3. 가상 머신이 회수된 경우의 데이터 재준비

이 셀을 실행한 뒤에 원래 실행했던 시작 셀을 그대로 다시 실행하면 됩니다. checkpoint는 Drive에
남아 있으므로 학습은 저장된 step부터 이어집니다. 압축 파일에 암호가 없다면 파라미터 셀의
`ARCHIVE_PASSWORD` 에 `none` 이라고 적으십시오.
"""))

    cells.append(code(DATA_STAGE + r'''

for name in ("train", "test", "dev"):
    p = data_root / name
    print("%-8s 이미지 %d장" % (name, len(list(p.glob("*.jpg"))) if p.is_dir() else 0))
print("VQA_DATA_ROOT:", os.environ["VQA_DATA_ROOT"])
'''))

    cells.append(md("""
## 1. 백그라운드 실행 확인 절차 (D1에 한 번만)

Pro+의 백그라운드 실행이 이 계정에서 실제로 어떻게 동작하는지 직접 확인합니다. 안내 문구만 믿고
밤새 실행을 맡기면 안 됩니다.

절차는 다음과 같습니다.

1. 아래 셀을 실행해서 15분짜리 시험용 job을 띄웁니다. 이 job은 1분마다 Drive의 파일에 현재
   시각을 기록합니다.
2. 브라우저 탭을 완전히 닫습니다. 노트북을 백그라운드로 두는 것이 아니라 탭을 닫아야 합니다.
3. 10분 정도 지난 뒤에 다시 접속합니다.
4. 확인 셀을 실행해서 기록된 시각 사이에 빈 구간이 있는지 봅니다.

판정 기준은 다음과 같습니다. 기록이 1분 간격으로 끊김 없이 이어지면 탭을 닫아도 실행이 유지되는
것입니다. 탭을 닫은 시점부터 기록이 멈췄다가 다시 접속한 시점부터 다시 시작되면, 백그라운드
실행이 동작하지 않는 것이므로 야간 실행 계획을 다시 세워야 합니다.
"""))

    cells.append(code(r'''
HEARTBEAT_FILE = RUN_ROOT_P / "bg_test_heartbeat.txt"
SCRIPT_PATH = RUN_ROOT_P / "bg_test.py"
SCRIPT_PATH.write_text(
    "import time, datetime, sys, pathlib\n"
    "out = pathlib.Path(sys.argv[1])\n"
    "for i in range(15):\n"
    "    with out.open('a', encoding='utf-8') as fh:\n"
    "        fh.write(datetime.datetime.now().isoformat(timespec='seconds') + '\\n')\n"
    "        fh.flush()\n"
    "    time.sleep(60)\n"
    "print('시험용 job을 마쳤습니다.')\n",
    encoding="utf-8",
)
if HEARTBEAT_FILE.exists():
    HEARTBEAT_FILE.unlink()

job_start("bg_test", [PY, str(SCRIPT_PATH), str(HEARTBEAT_FILE)], extra=False)
print()
print("지금부터 브라우저 탭을 닫고 10분쯤 뒤에 다시 접속하십시오.")
print("기록 파일:", HEARTBEAT_FILE)
'''))

    cells.append(code(r'''
import datetime

if not HEARTBEAT_FILE.exists():
    print("아직 기록이 없습니다. job이 시작되었는지 status로 확인하십시오.")
else:
    stamps = [datetime.datetime.fromisoformat(s)
              for s in HEARTBEAT_FILE.read_text(encoding="utf-8").split() if s]
    print("기록 %d개" % len(stamps))
    gaps = []
    for a, b in zip(stamps, stamps[1:]):
        sec = (b - a).total_seconds()
        gaps.append(sec)
        flag = "  <-- 끊김" if sec > 120 else ""
        print("  %s -> %s  %5.0f초%s" % (a.time(), b.time(), sec, flag))
    if gaps and max(gaps) <= 120:
        print()
        print("판정: 탭을 닫아도 실행이 유지되었습니다. 야간 실행을 맡길 수 있습니다.")
    elif gaps:
        print()
        print("판정: 끊긴 구간이 있습니다. 백그라운드 실행을 믿지 말고,")
        print("      checkpoint 간격을 짧게 잡고 재개 절차를 자주 확인해야 합니다.")
'''))

    cells.append(md("""
## 2. base validation 추론

학습 전 27B로 고정 validation을 추론합니다. 결합 가중치를 고를 때 쓰는 기준 확률이므로 반드시
먼저 확보해야 합니다. 진행 상황을 보려면 `python -m vqa.jobs tail --name <job 이름> --lines 20`
을 실행하십시오. Drive에서 모델을 불러오는 데 수 분이 걸리므로 첫 진행 로그가 나오기까지 시간이
걸릴 수 있습니다.
"""))

    cells.append(code(r'''
cmd = [PY, "-m", "vqa.predict",
       "--config", CONFIG, "--run-id", BASE_RUN_ID,
       "--csv", "train.csv", "--ids-file", VAL_IDS_FILE,
       "--adapter", "none", "--out-name", VAL_OUT_BASE] + OVERRIDES
job_start(BASE_RUN_ID + "_val", cmd)
'''))

    cells.append(md("""
## 3. base 전체 test 추론

첫 유효 제출과 확률 캐시를 확보하는 단계입니다. 이 작업이 끝나야 수정이 제출을 준비할 수
있습니다. 같은 GPU에서 학습과 동시에 돌리지 말고 순서대로 실행하십시오.
"""))

    cells.append(code(r'''
cmd = [PY, "-m", "vqa.predict",
       "--config", CONFIG, "--run-id", BASE_RUN_ID,
       "--csv", "test.csv",
       "--adapter", "none", "--out-name", "test_base"] + OVERRIDES
job_start(BASE_RUN_ID + "_test", cmd)
'''))

    cells.append(md("""
## 4. base 제출 파일 만들기

전체 test 추론이 끝난 뒤에 실행합니다. 검사기를 통과하면 수정에게 파일과 해시를 넘깁니다.
"""))

    cells.append(code(r'''
PRED = RUN_ROOT_P / BASE_RUN_ID / "preds" / "test_base.csv"
SUB = "artifacts/submission/%s_base.csv" % BASE_RUN_ID
Path("artifacts/submission").mkdir(parents=True, exist_ok=True)

if not PRED.exists():
    print("아직 추론이 끝나지 않았습니다:", PRED)
else:
    vqa("submit", "--pred", str(PRED),
        "--sample", str(Path(os.environ["VQA_DATA_ROOT"]) / "sample_submission.csv"),
        "--out", SUB)
    script("check_submission.py", "--submission", SUB,
           "--sample", str(Path(os.environ["VQA_DATA_ROOT"]) / "sample_submission.csv"))
    print()
    print("제출 후보: %s" % SUB)
    print("sha256   : %s" % sha256_of(SUB))
'''))

    cells.append(md("""
## 5. 학습

`--resume auto` 가 기본이므로 이 셀을 여러 번 실행해도 안전합니다. 가상 머신이 회수된 뒤에도
같은 셀을 그대로 다시 실행하면 Drive의 checkpoint에서 이어집니다.

아래 셀은 시작하기 전에 두 가지를 확인합니다. 첫째로 지금 GPU가 정책이 요구하는 정밀도를
감당하는지 보고, 둘째로 이 run이 이미 다른 정밀도로 시작된 적이 있는지를 `run_manifest.json`
에서 확인합니다. 정밀도가 다르면 학습을 시작하지 않고 멈춥니다. 같은 run-id 아래에 bf16
checkpoint와 nf4 checkpoint가 섞이면 이어서 하는 실행을 믿을 수 없게 되기 때문입니다.
"""))

    cells.append(code(r'''
if not TRAINING_ALLOWED:
    raise RuntimeError(
        "현재 GPU의 VRAM이 %.1fGB이므로 bf16 학습을 시작하거나 이어서 할 수 없습니다. "
        "런타임을 끊고 80GB A100으로 다시 연결한 다음 이 셀을 실행하십시오. "
        "지금 이 가상 머신에서는 추론만 nf4로 돌릴 수 있습니다." % VRAM_GB
    )

# 이미 시작한 run이라면 그때 쓴 정밀도와 지금 정밀도가 같아야 합니다.
_manifest_path = RUN_ROOT_P / FT_RUN_ID / "run_manifest.json"
if _manifest_path.exists():
    _manifest = json.loads(_manifest_path.read_text(encoding="utf-8"))
    _prev = str(_manifest.get("config", {}).get("precision", "")).lower()
    if _prev and _prev != PRECISION:
        raise RuntimeError(
            "run %s 은 %s 로 시작했는데 지금 설정은 %s 입니다. 정밀도가 다른 채로 이어서 하면 "
            "checkpoint가 섞입니다. 같은 정밀도의 GPU로 다시 연결하거나, 다른 run-id를 쓰십시오.\n"
            "기록 파일: %s" % (FT_RUN_ID, _prev, PRECISION, _manifest_path)
        )
    print("run %s 의 기존 정밀도 %s 와 현재 정밀도가 같습니다. 이어서 진행합니다." % (FT_RUN_ID, _prev))

cmd = [PY, "-m", "vqa.train",
       "--config", CONFIG, "--run-id", FT_RUN_ID,
       "--split", SPLIT_FILE, "--resume", "auto"] + OVERRIDES
job_start(FT_RUN_ID + "_train", cmd)
'''))

    cells.append(md("""
## 6. FT validation과 전체 test 추론

`--also-base` 를 붙이면 같은 실행에서 base 확률도 함께 남깁니다. 결합 후보를 비교할 때 추가
GPU 추론 없이 CPU에서만 계산할 수 있게 됩니다.
"""))

    cells.append(code(r'''
ADAPTER = str(RUN_ROOT_P / FT_RUN_ID / "final_adapter")

cmd = [PY, "-m", "vqa.predict",
       "--config", CONFIG, "--run-id", FT_RUN_ID,
       "--csv", "train.csv", "--ids-file", VAL_IDS_FILE,
       "--adapter", ADAPTER, "--out-name", VAL_OUT_FT] + ALSO_BASE_ARGS + OVERRIDES
job_start(FT_RUN_ID + "_val", cmd)
'''))

    cells.append(code(r'''
cmd = [PY, "-m", "vqa.predict",
       "--config", CONFIG, "--run-id", FT_RUN_ID,
       "--csv", "test.csv",
       "--adapter", ADAPTER, "--out-name", "test_ft"] + ALSO_BASE_ARGS + OVERRIDES
job_start(FT_RUN_ID + "_test", cmd)
'''))

    cells.append(md("""
## 7. 감시

이 셀은 일정 시간 동안 상태와 마지막 checkpoint를 반복해서 출력합니다. 탭을 열어 둔 동안
세션이 유휴 상태로 판정되지 않게 하는 효과도 있습니다. 다만 이것이 백그라운드 실행을 대신하지는
않습니다. 탭을 닫아야 할 때는 1번 절의 확인 결과에 따라 판단하십시오.

감시할 job 이름을 `WATCH` 에 적고 실행하십시오.
"""))

    cells.append(code(r'''
from IPython.display import clear_output

WATCH = FT_RUN_ID + "_train"

deadline = time.time() + MONITOR_MINUTES * 60
while time.time() < deadline:
    clear_output(wait=True)
    print("감시 중: %s  (%s)" % (WATCH, time.strftime("%H:%M:%S")))
    print("남은 감시 시간: %.1f분" % ((deadline - time.time()) / 60.0))
    print("=" * 60)
    job_status(WATCH)
    for run_id in (BASE_RUN_ID, FT_RUN_ID):
        latest = RUN_ROOT_P / run_id / "checkpoints" / "latest.json"
        if latest.exists():
            print("[%s] latest: %s" % (run_id, latest.read_text(encoding="utf-8")[:200]))
        hb = RUN_ROOT_P / run_id / "heartbeat.json"
        if hb.exists():
            print("[%s] heartbeat: %s" % (run_id, hb.read_text(encoding="utf-8")[:200]))
    sys.stdout.flush()
    time.sleep(MONITOR_INTERVAL_SEC)

print("감시를 마쳤습니다. job은 계속 돌아갑니다.")
'''))

    cells.append(md("""
## 8. 실행 상태 공유

아래 출력을 팀 채널에 붙여 넣습니다. 다른 팀원은 실행 중인 런타임을 볼 수 없으므로, 이 기록이
유일한 공유 수단입니다.
"""))

    cells.append(code(r'''
lines = ["===== Colab 27B 실행 상태 ====="]
lines.append("시각    : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
lines.append("config  : %s" % CONFIG)
for name in JOB_NAMES:
    res = run([PY, "-m", "vqa.jobs", "status", "--name", name],
              check=False, quiet=True, echo=False, hint=False)
    first = res["output"].strip().splitlines()
    lines.append("%-24s %s" % (name, first[0] if first else "기록 없음"))
for run_id in (BASE_RUN_ID, FT_RUN_ID):
    pred_dir = RUN_ROOT_P / run_id / "preds"
    if pred_dir.is_dir():
        for f in sorted(pred_dir.glob("*.csv")):
            lines.append("예측 파일: %s (%d바이트)" % (f.name, f.stat().st_size))
lines.append("==============================")
print("\n".join(lines))
'''))
    return notebook(cells)


# --------------------------------------------------------------------------
# 30_eval_submit
# --------------------------------------------------------------------------
def nb_30():
    cells = []
    cells.append(md("""
# 30. 평가, 결합, 제출 검사

담당: 수정
목표: GPU 결과를 기다리지 않고 가짜 확률로 평가와 병합, 결합, 제출 검사 경로를 먼저 완성한 뒤,
같은 셀을 실제 예측 파일로 돌려서 최종 제출을 만드는 것입니다.

첫 30분 안에 앞부분을 끝내 두면, 준혁과 유림의 결과가 나오는 즉시 제출을 준비할 수 있습니다.

## 통과 기준(PASS)
1. 가짜 확률로 병합, 평가, 결합, 제출, 검사가 모두 종료 코드 0으로 끝납니다.
2. 일부러 망가뜨린 파일에서 검사기가 실패를 잡아냅니다.
3. 실제 예측 파일로 같은 경로를 돌려서 제출 후보를 만들었습니다.

## 넘겨줄 내용
제출 파일 경로와 해시, 실험 기록 표입니다.
"""))

    cells.append(md(CLI_ASSUMPTIONS_MD))

    cells.append(code(r'''
# ===== 파라미터 =====
PROJECT_ROOT = r"."
DATA_ROOT = r"."
RUN_ROOT = r"./artifacts/runs"
LOCAL_SCRATCH = r"./artifacts/scratch"
MODEL_ROOT = r""

VAL_IDS_FILE = "fold_csv_files/val_fold0.csv"    # 팀 공식 검증 분할 (fold 0, 1,344문항)
FAKE_RUN_ID = "fake_smoke"
N_FAKE_SHARDS = 2
SEED = 7

# 실제 파일로 전환할 때 여기를 고칩니다. 아직 없는 출처는 빈 문자열로 둡니다.
REAL_VAL_BASE27 = "artifacts/runs/colab27b_base_bf16/preds/fold0_base.csv"
REAL_VAL_FT27 = "artifacts/runs/colab27b_ft_bf16/preds/fold0_ft.csv"
REAL_VAL_FT9 = "artifacts/runs/lab9b_ft_v1/preds/fold0_ft.csv"
REAL_TEST_BASE27 = "artifacts/runs/colab27b_base_bf16/preds/test_base.csv"
REAL_TEST_FT27 = "artifacts/runs/colab27b_ft_bf16/preds/test_ft.csv"
REAL_TEST_FT9 = "artifacts/runs/lab9b_ft_v1/preds/test_ft.csv"
TIE_EPSILON = 0.002   # 정확도 차이가 이 값 이하면 더 단순한 후보를 고릅니다.

EXPERIMENTS_LOG = "artifacts/experiments.jsonl"
''', parameters=True))

    cells.append(code(BOOT))

    cells.append(md("""
## 1. 가짜 확률 만들기

`scripts/make_fake_predictions.py` 가 계약서에 정해진 예측 형식 그대로 만들어 줍니다. 컬럼은
`id, p_a, p_b, p_c, p_d, run_id` 이고, shard로 나누면 파일 이름이 `<이름>.shard{i}of{n}.csv` 가
됩니다.

`--mode oracle` 로 두면 `--gold` 의 정답을 `--accuracy` 비율만큼 맞히는 확률을 만듭니다. 평가기가
0이나 1이 아닌 값을 내는지 확인할 수 있기 때문에, 경로를 점검할 때는 이 방식이 편합니다.
"""))

    cells.append(code(r"""
FAKE_DIR = RUN_ROOT_P / FAKE_RUN_ID / "preds"
FAKE_DIR.mkdir(parents=True, exist_ok=True)
GOLD = str(Path(os.environ["VQA_DATA_ROOT"]) / "train.csv")
SAMPLE = str(Path(os.environ["VQA_DATA_ROOT"]) / "sample_submission.csv")
TEST_CSV = str(Path(os.environ["VQA_DATA_ROOT"]) / "test.csv")

if not Path(VAL_IDS_FILE).exists():
    raise FileNotFoundError(
        VAL_IDS_FILE + " 이 없습니다. 저장소의 fold_csv_files/ 가 최신인지 확인하기 전에는 "
        "train.csv의 id 컬럼을 --ids로 넘겨서 형식만 확인하십시오."
    )

script("make_fake_predictions.py",
       "--ids", VAL_IDS_FILE,
       "--out", str(FAKE_DIR / "val_fake_base27.csv"),
       "--mode", "oracle", "--gold", GOLD, "--accuracy", "0.70",
       "--run-id", FAKE_RUN_ID + "_base27",
       "--shards", N_FAKE_SHARDS,
       "--seed", SEED)

shard_files = sorted(FAKE_DIR.glob("val_fake_base27.shard*of*.csv"))
print()
for f in shard_files:
    print(" 만들어진 shard:", f.name)
"""))

    cells.append(md("""
## 2. shard 병합

각 ID가 정확히 한 번씩만 들어 있어야 합니다. `--expect-ids` 로 기대 ID 목록을 넘기면 병합기가
빠진 ID와 남는 ID를 모두 잡아냅니다.
"""))

    cells.append(code(r"""
MERGED = FAKE_DIR / "val_fake_base27.csv"
script("merge_shards.py",
       "--shards", *[str(f) for f in shard_files],
       "--expect-ids", VAL_IDS_FILE,
       "--out", str(MERGED))
"""))

    cells.append(md("""
## 3. 평가

정확도가 0.7 근처로 나오면 평가 경로가 정상입니다. 정확히 0이나 1이 나오면 ID 매칭이나 정답 컬럼
해석에 문제가 있는 것입니다.

가짜 확률로 하는 점검이므로 `--no-log` 를 붙여서 실험 기록에는 남기지 않습니다.
"""))

    cells.append(code(r"""
Path("artifacts/eval").mkdir(parents=True, exist_ok=True)
script("evaluate_run.py",
       "--pred", str(MERGED),
       "--gold", GOLD,
       "--val-ids", VAL_IDS_FILE,
       "--json-out", "artifacts/eval/fake_val.json",
       "--wrong-out", "artifacts/eval/fake_val_wrong.csv",
       "--no-log")
"""))

    cells.append(md("""
## 4. 결합

`scripts/combine.py` 는 계획서 8절에서 미리 정한 네 후보만 비교합니다. 후보는 base27 단독,
ft27 단독, ft27 0.8에 base27 0.2를 더한 것, ft27 0.8에 ft9 0.2를 더한 것입니다. 가중치를 자유롭게
탐색하지 않는 이유는 validation에 과적합되는 것을 막기 위해서입니다.

각 확률 파일은 `--val-base27`, `--val-ft27`, `--val-ft9` 처럼 출처 이름을 붙여서 넘깁니다.
test 파일은 같은 이름에 `--test-` 를 붙입니다.
"""))

    cells.append(code(r"""
for name, acc in (("ft27", 0.74), ("ft9", 0.66)):
    script("make_fake_predictions.py",
           "--ids", VAL_IDS_FILE,
           "--out", str(FAKE_DIR / ("val_fake_%s.csv" % name)),
           "--mode", "oracle", "--gold", GOLD, "--accuracy", str(acc),
           "--run-id", FAKE_RUN_ID + "_" + name,
           "--seed", SEED + 1,
           quiet=True)
    print("만들었습니다: val_fake_%s.csv (목표 정확도 %.2f)" % (name, acc))

for name in ("base27", "ft27", "ft9"):
    script("make_fake_predictions.py",
           "--ids", TEST_CSV,
           "--out", str(FAKE_DIR / ("test_fake_%s.csv" % name)),
           "--mode", "random",
           "--run-id", FAKE_RUN_ID + "_" + name,
           "--seed", SEED + 2,
           quiet=True)
    print("만들었습니다: test_fake_%s.csv" % name)
"""))

    cells.append(code(r"""
FAKE_TEST_COMBINED = FAKE_DIR / "test_fake_combined.csv"
script("combine.py",
       "--val-base27", str(FAKE_DIR / "val_fake_base27.csv"),
       "--val-ft27", str(FAKE_DIR / "val_fake_ft27.csv"),
       "--val-ft9", str(FAKE_DIR / "val_fake_ft9.csv"),
       "--test-base27", str(FAKE_DIR / "test_fake_base27.csv"),
       "--test-ft27", str(FAKE_DIR / "test_fake_ft27.csv"),
       "--test-ft9", str(FAKE_DIR / "test_fake_ft9.csv"),
       "--gold", GOLD,
       "--val-ids", VAL_IDS_FILE,
       "--out", str(FAKE_TEST_COMBINED),
       "--val-out", str(FAKE_DIR / "val_fake_combined.csv"),
       "--report-out", "artifacts/eval/fake_combine_report.json",
       "--no-log")
"""))

    cells.append(md("""
## 5. 제출 파일과 검사기

결합한 test 확률을 제출 파일로 바꾸고 검사기를 돌립니다. ID 집합과 행 수, 순서, 값 범위를 모두
확인합니다. `vqa.submit` 은 파싱에 실패한 행을 임의로 `a` 로 채우지 않습니다.
"""))

    cells.append(code(r"""
Path("artifacts/submission").mkdir(parents=True, exist_ok=True)
FAKE_SUB = "artifacts/submission/fake_smoke.csv"

vqa("submit", "--pred", str(FAKE_TEST_COMBINED),
    "--sample", SAMPLE, "--out", FAKE_SUB,
    "--margins-out", "artifacts/eval/fake_margins.csv")
script("check_submission.py", "--submission", FAKE_SUB, "--sample", SAMPLE)
"""))

    cells.append(md("""
## 6. 검사기가 실패를 잡는지 확인

일부러 망가뜨린 입력을 넣어서 검사가 실제로 막아 주는지 봅니다. 여기에서 전부 통과해 버리면
검사기를 신뢰할 수 없습니다.

두 방향으로 확인합니다. 먼저 예측 확률 파일을 망가뜨려서 `vqa.submit` 이 막는지 보고, 그다음
제출 CSV를 직접 망가뜨려서 `check_submission.py` 가 막는지 봅니다.
"""))

    cells.append(code(r"""
failures = []

for mode in ("missing-id", "duplicate-id", "nan", "bad-sum"):
    broken_pred = FAKE_DIR / ("broken_%s.csv" % mode)
    script("make_fake_predictions.py",
           "--ids", TEST_CSV,
           "--out", str(broken_pred),
           "--mode", "random",
           "--run-id", FAKE_RUN_ID + "_broken",
           "--break", mode, "--break-count", "3",
           "--seed", SEED,
           quiet=True)
    res = vqa("submit", "--pred", str(broken_pred), "--sample", SAMPLE,
              "--out", "artifacts/submission/broken_%s.csv" % mode,
              check=False, quiet=True)
    caught = res["rc"] != 0
    failures.append((mode, caught))
    print("%-14s -> %s" % (mode, "막았습니다" if caught else "통과시켰습니다"))
"""))

    cells.append(code(r"""
BROKEN_SUB = "artifacts/submission/fake_broken.csv"
rows = read_csv_rows(FAKE_SUB)
rows = rows[:-3]                     # 행을 몇 개 지웁니다.
if rows:
    rows[0]["answer"] = "e"          # 허용되지 않는 값을 넣습니다.
    rows.append(dict(rows[0]))       # 중복 ID를 만듭니다.
with open(BROKEN_SUB, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "answer"])
    w.writeheader()
    for r in rows:
        w.writerow({"id": r["id"], "answer": r["answer"]})

res = script("check_submission.py", "--submission", BROKEN_SUB,
             "--sample", SAMPLE, check=False)
failures.append(("제출 CSV 손상", res["rc"] != 0))

print()
missed = [m for m, caught in failures if not caught]
if missed:
    print("문제: 다음 결함을 잡지 못했습니다. 검사 항목을 보완해야 합니다.")
    for m in missed:
        print("  -", m)
else:
    print("PASS: 준비한 결함을 모두 잡아냈습니다.")
"""))

    cells.append(md("""
## 7. 실제 예측 파일로 전환

위 경로가 모두 통과했으면, 파라미터 셀의 `REAL_*` 값을 실제 파일 경로로 바꾸고 아래 셀을
실행합니다. 셀 구조는 위와 같으며 입력만 달라집니다.

아직 준비되지 않은 출처는 빈 문자열로 두면 그 출처를 뺀 채로 비교합니다. 예를 들어 27B 미세조정이
끝나기 전이라면 `REAL_VAL_FT27` 만 비워 두면 됩니다.
"""))

    cells.append(code(r"""
def source_args(prefix, mapping):
    args = []
    for name, path in mapping.items():
        if path and Path(path).exists():
            args += ["--%s-%s" % (prefix, name), path]
        elif path:
            print("아직 없는 파일이라 제외합니다:", path)
    return args


val_args = source_args("val", {"base27": REAL_VAL_BASE27,
                               "ft27": REAL_VAL_FT27,
                               "ft9": REAL_VAL_FT9})
test_args = source_args("test", {"base27": REAL_TEST_BASE27,
                                 "ft27": REAL_TEST_FT27,
                                 "ft9": REAL_TEST_FT9})

print("validation 출처 %d개, test 출처 %d개" % (len(val_args) // 2, len(test_args) // 2))
"""))

    cells.append(code(r"""
if not val_args:
    print("아직 실제 validation 확률 파일이 하나도 없습니다. 준비되면 다시 실행하십시오.")
else:
    for label, pred in (("base27", REAL_VAL_BASE27),
                        ("ft27", REAL_VAL_FT27),
                        ("ft9", REAL_VAL_FT9)):
        if not (pred and Path(pred).exists()):
            continue
        print("=" * 60)
        print("단독 성능:", label)
        script("evaluate_run.py",
               "--pred", pred, "--gold", GOLD, "--val-ids", VAL_IDS_FILE,
               "--json-out", "artifacts/eval/real_%s.json" % label,
               "--note", "단독 %s" % label,
               check=False)
"""))

    cells.append(code(r"""
if not (val_args and test_args):
    print("결합에는 validation과 test 확률이 모두 필요합니다.")
else:
    REAL_SUB = "artifacts/submission/final_candidate.csv"
    script("combine.py", *val_args, *test_args,
           "--gold", GOLD,
           "--val-ids", VAL_IDS_FILE,
           "--out", "artifacts/eval/real_test_combined.csv",
           "--val-out", "artifacts/eval/real_val_combined.csv",
           "--report-out", "artifacts/eval/real_combine_report.json",
           "--tie-epsilon", str(TIE_EPSILON))

    vqa("submit", "--pred", "artifacts/eval/real_test_combined.csv",
        "--sample", SAMPLE, "--out", REAL_SUB,
        "--margins-out", "artifacts/eval/real_margins.csv")
    script("check_submission.py", "--submission", REAL_SUB, "--sample", SAMPLE)

    print()
    print("===== 최종 제출 후보 =====")
    print("파일   : %s" % REAL_SUB)
    print("sha256 : %s" % sha256_of(REAL_SUB))
    print("결합 보고: artifacts/eval/real_combine_report.json")
    print("=========================")
"""))

    cells.append(md("""
## 8. 실험 기록

`evaluate_run.py` 와 `combine.py` 는 실행할 때마다 `artifacts/experiments.jsonl` 에 한 줄씩
덧붙입니다. 여러 사람이 동시에 기록해도 기존 내용을 덮어쓰지 않습니다.

아래 셀이 그 기록을 표로 정리합니다. 마지막 날 최종본을 고를 때 이 표가 근거가 됩니다.
Kaggle 점수처럼 사람만 알 수 있는 값은 그다음 셀로 덧붙입니다.
"""))

    cells.append(code(r"""
LOG_PATH = Path(EXPERIMENTS_LOG)
entries = []
if LOG_PATH.exists():
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue

if not entries:
    print("아직 기록된 실험이 없습니다.")
else:
    cols = ["logged_at", "run_id", "accuracy", "n", "seconds_per_item", "note"]
    widths = {c: max(len(c), max(len(str(e.get(c, ""))[:40]) for e in entries)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for e in entries[-40:]:
        print(" | ".join(str(e.get(c, ""))[:40].ljust(widths[c]) for c in cols))
    print()
    print("전체 %d건 가운데 최근 %d건을 보여 주었습니다." % (len(entries), min(40, len(entries))))
"""))

    cells.append(code(r"""
# Kaggle 점수처럼 자동으로 알 수 없는 값은 여기에 적어서 덧붙입니다.
MANUAL_NOTE = {
    "run_id": "",
    "submission": "",
    "sha256": "",
    "kaggle_public": "",
    "selected_as_final": False,
    "note": "",
}

if MANUAL_NOTE["run_id"]:
    MANUAL_NOTE["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    MANUAL_NOTE["source"] = "manual"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(MANUAL_NOTE, ensure_ascii=False) + "\n")
    print("기록을 추가했습니다:", LOG_PATH)
else:
    print("run_id가 비어 있어 추가하지 않았습니다. 기록할 내용이 있으면 MANUAL_NOTE를 채우십시오.")
"""))
    return notebook(cells)


# --------------------------------------------------------------------------
# build / validate
# --------------------------------------------------------------------------
NOTEBOOKS = {
    "00_lab_env_check.ipynb": nb_00,
    "01_data_audit_split.ipynb": nb_01,
    "10_lab_baseline_zero_shot.ipynb": nb_10,
    "11_lab_prompt_ocr_compare.ipynb": nb_11,
    "12_lab_train_9b_qlora.ipynb": nb_12,
    "20_colab_27b_setup_preflight.ipynb": nb_20,
    "21_colab_27b_run.ipynb": nb_21,
    "30_eval_submit.ipynb": nb_30,
}

# IPython 매직만 잡고, `% (...)` 로 시작하는 문자열 포매팅 연속 행이나 `!=` 는 건드리지 않는다.
MAGIC_RE = re.compile(r"^[ \t]*(![^\s=]|%[A-Za-z]\w*)")


def strip_magics(source: str) -> str:
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return "pass"
    out = []
    for line in lines:
        out.append("pass" if MAGIC_RE.match(line) else line)
    return "\n".join(out)


def assign_stable_ids(nb, name: str) -> None:
    """nbformat이 임의로 붙이는 cell id를 고정값으로 바꿔서 재생성 결과를 동일하게 만든다."""
    stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(name).stem).strip("-")
    for idx, cell in enumerate(nb["cells"]):
        cell["id"] = "%s-%03d" % (stem, idx)


def validate(nb, name: str) -> list[str]:
    problems: list[str] = []
    try:
        nbf.validate(nb)
    except Exception as exc:  # nbformat.ValidationError
        problems.append("%s: nbformat.validate 실패: %s" % (name, exc))

    n_params = 0
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        if "parameters" in cell["metadata"].get("tags", []):
            n_params += 1
        try:
            ast.parse(strip_magics(cell["source"]))
        except SyntaxError as exc:
            problems.append("%s cell %d: SyntaxError %s (line %s)" % (name, idx, exc.msg, exc.lineno))
    if n_params != 1:
        problems.append("%s: parameters 태그가 붙은 셀이 %d개입니다 (1개여야 합니다)." % (name, n_params))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="파일을 쓰지 않고 검증만 수행합니다.")
    args = ap.parse_args()

    NB_DIR.mkdir(parents=True, exist_ok=True)
    all_problems: list[str] = []
    for name, factory in NOTEBOOKS.items():
        nb = factory()
        assign_stable_ids(nb, name)
        problems = validate(nb, name)
        all_problems.extend(problems)
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = len(nb["cells"]) - n_code
        status = "OK" if not problems else "문제 %d건" % len(problems)
        print("%-36s cells=%2d (md %2d / code %2d)  %s"
              % (name, len(nb["cells"]), n_md, n_code, status))
        if not args.check:
            nbf.write(nb, str(NB_DIR / name), version=4)

    if all_problems:
        print()
        for p in all_problems:
            print("  " + p)
        return 1
    print()
    print("검증을 통과했습니다: nbformat.validate + 모든 코드 셀의 ast.parse")
    if not args.check:
        print("생성 위치:", NB_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
