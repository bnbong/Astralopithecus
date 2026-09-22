#!/usr/bin/env python3
"""VQA 4지선다 학습용 이미지 증강 함수 (SSAFY AI 챌린지 2회차)

이미지 속 텍스트(간판, 영수증, 표지판 등)를 정확히 읽어야 하는 문제 특성상,
증강 강도가 세면 텍스트가 아예 안 읽히게 되어 오히려 라벨과 이미지가
불일치하는 역효과가 난다. 그래서 여기 있는 증강들은:

  - 실제 이미지로 미리보기 검증을 거쳐 강도를 조정한 상태 (v2)
  - 좌우/상하 반전은 넣지 않는다 (텍스트가 거울상이 되어 완전히 읽을 수 없어짐)
  - 강한 크롭도 넣지 않는다 (정답에 필요한 텍스트가 잘려나갈 위험)

v2 변경 사항 (실제 이미지 검증 결과 반영):
  - [버그 수정] rotate: 빈 모서리가 검은색으로 채워지던 문제를 수정.
    이미지를 반사(reflect) 패딩한 뒤 회전하고 원래 크기로 잘라내는 방식으로,
    모서리에 검은 삼각형 없이 자연스럽게 채워지도록 함.
  - brightness_contrast: 0.25/0.25 -> 0.4/0.35 로 상향 (기존 강도가
    원본과 거의 구분이 안 될 정도로 약했음)
  - light_blur: 상한 1.2 -> 1.5 로 소폭 상향 (텍스트 가독성에 여유 있었음)
  - jpeg_compress: quality 하한 45 -> 30 으로 하향 (여유 있었음)

v3 변경 사항:
  - rotate 각도(±8도) -> ±3도로 축소 (결과 확인 후 조정 요청 반영)

의존성: Pillow, numpy (rotate의 reflect 패딩에 사용)
    pip install Pillow numpy
"""
import io
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# 개별 증강 함수
# ---------------------------------------------------------------------------

def rotate(image: Image.Image, degrees=(-3, 3)) -> Image.Image:
    """약한 회전. 삐딱하게 찍은 사진을 흉내낸다.

    [v3] 기본 각도를 ±8도 -> ±3도로 축소함 (결과 확인 후 조정 요청 반영).
    degrees 범위를 벗어나면(예: ±20도 이상) 표/영수증의 줄글이 심하게
    기울어져 모델 판독이 어려워지므로 이 기본값 이상은 권장하지 않음.

    [v2] 이미지를 가장자리 반사(reflect) 방식으로 미리 패딩한 뒤 회전하고
    원래 크기로 중앙을 잘라내어, 모서리가 검은 삼각형 대신 자연스러운
    이미지 내용으로 채워지도록 한다 (기존 fillcolor=None 은 검은색으로
    채워지는 버그가 있었음).
    """
    angle = random.uniform(*degrees)
    w, h = image.size

    arr = np.array(image)
    pad = int(max(w, h) * 0.25)
    arr_padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    padded = Image.fromarray(arr_padded)

    rotated = padded.rotate(angle, resample=Image.BICUBIC, expand=False)
    cropped = rotated.crop((pad, pad, pad + w, pad + h))
    return cropped


def brightness_contrast(image: Image.Image,
                         brightness_range=(0.6, 1.4),
                         contrast_range=(0.65, 1.35)) -> Image.Image:
    """밝기/대비 조정. 역광, 어두운 실내 촬영을 흉내낸다.

    [v2] 기존 범위(0.75~1.25)가 원본과 거의 구분이 안 될 정도로 약해서,
    범위를 넓혔다 (밝기 ±40%, 대비 ±35%).
    """
    b = random.uniform(*brightness_range)
    c = random.uniform(*contrast_range)
    image = ImageEnhance.Brightness(image).enhance(b)
    image = ImageEnhance.Contrast(image).enhance(c)
    return image


def light_blur(image: Image.Image, radius_range=(0.0, 1.5)) -> Image.Image:
    """약한 가우시안 블러. 손떨림/초점 미스를 흉내낸다.

    radius 2.0을 넘어가면 작은 글자(가격표 숫자 등)가 뭉개지기 시작하므로
    이 상한(1.5) 이상은 권장하지 않음.
    [v2] 기존 상한(1.2)이 텍스트 가독성에 여유가 있어 1.5로 소폭 상향.
    """
    radius = random.uniform(*radius_range)
    if radius <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def jpeg_compress(image: Image.Image, quality_range=(30, 85)) -> Image.Image:
    """JPEG 재압축 아티팩트. 저화질 캡처/전송 이미지를 흉내낸다.

    [v2] 기존 하한(45)이 원본과 거의 구분이 안 될 정도로 약해서 30으로 낮춤.
    quality 20 미만은 텍스트가 심하게 깨지기 시작하므로 하한을 더 낮추지 말 것.
    """
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# 사용 가능한 증강 목록. compose_augment / preview에서 이름으로 참조한다.
AUGMENTATIONS = {
    "rotate": rotate,
    "brightness_contrast": brightness_contrast,
    "blur": light_blur,
    "jpeg": jpeg_compress,
}


# ---------------------------------------------------------------------------
# 조합 적용
# ---------------------------------------------------------------------------

def compose_augment(image: Image.Image, use=None, p=0.5, seed=None) -> Image.Image:
    """여러 증강을 순서대로, 각각 확률 p로 적용한다.

    Args:
        image: PIL Image (RGB)
        use: 적용할 증강 이름 리스트. 기본은 전부
             (예: ["rotate", "brightness_contrast"] 로 일부만 선택 가능)
        p: 각 증강이 적용될 확률 (0~1). 매 증강마다 독립적으로 굴림.
           즉 p=0.5, use 4개면 이미지마다 평균 2개 정도의 증강이 겹쳐 적용됨.
        seed: 재현 필요하면 지정 (None이면 매번 랜덤)

    Returns:
        증강이 적용된 새 PIL Image (원본은 변경하지 않음)
    """
    if seed is not None:
        random.seed(seed)
    names = use if use is not None else list(AUGMENTATIONS.keys())

    out = image.copy()
    applied = []
    for name in names:
        if name not in AUGMENTATIONS:
            raise ValueError(f"알 수 없는 증강: {name}. 사용 가능: {list(AUGMENTATIONS)}")
        if random.random() < p:
            out = AUGMENTATIONS[name](out)
            applied.append(name)
    return out


# ---------------------------------------------------------------------------
# 시각적 검증용 미리보기 생성
# ---------------------------------------------------------------------------

def preview_augmentations(image_path: str, out_path: str, font_size=14):
    """원본 + 개별 증강 4종 + 전체 조합, 총 6컷을 가로로 이어붙여 저장한다.

    글자형 이미지로 이 함수를 돌려서, out_path를 직접 열어 텍스트가
    각 컷에서 여전히 읽히는지 눈으로 확인할 것. 뭉개지는 증강이 있으면
    해당 함수의 range 파라미터를 좁혀서 재확인.
    """
    from PIL import ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    variants = [("original", img)]
    variants.append(("rotate", rotate(img)))
    variants.append(("bright/contrast", brightness_contrast(img)))
    variants.append(("blur", light_blur(img)))
    variants.append(("jpeg", jpeg_compress(img)))
    variants.append(("combined(p=0.5)", compose_augment(img, p=0.5)))

    label_h = font_size + 8
    grid = Image.new("RGB", (w * len(variants), h + label_h), "white")
    draw = ImageDraw.Draw(grid)
    for i, (name, v) in enumerate(variants):
        grid.paste(v, (i * w, label_h))
        draw.text((i * w + 4, 2), name, fill="black")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    grid.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 베이스라인 Dataset에 꽂는 방법 (baseline 노트북 참고용, 여기서는 실행 안 함)
# ---------------------------------------------------------------------------
"""
베이스라인 노트북의 VQAMCDataset.__getitem__ 안에서, 이미지를 연 직후
(모델에 넣기 전) 아래처럼 학습(train)일 때만 적용하면 된다.

    from augmentation import compose_augment

    class VQAMCDataset(Dataset):
        def __init__(self, df, data_dir, processor, is_train=True):
            ...
            self.is_train = is_train   # 검증/추론 시에는 증강 끄기

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(os.path.join(self.data_dir, row["path"])).convert("RGB")

            if self.is_train:
                image = compose_augment(image, p=0.5)   # 학습 때만 증강

            prompt = build_mc_prompt(row["question"], row["a"], row["b"], row["c"], row["d"])
            ...

주의:
- 검증(val_fold0)·테스트 추론 시에는 절대 증강을 적용하지 말 것
  (원본 그대로 평가해야 실제 성능을 알 수 있음)
- p(적용 확률)와 각 함수의 range는 여기 preview_augmentations()로
  실제 글자형 이미지에서 확인한 뒤 확정할 것 (v2 기본값은 실제 이미지로
  1차 검증을 거친 값이지만, 다른 이미지들로도 추가 확인 권장)
"""


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="증강 미리보기 생성 (단일 이미지)")
    ap.add_argument("--image", required=True, help="원본 이미지 경로")
    ap.add_argument("--out", required=True, help="미리보기 저장 경로 (png)")
    args = ap.parse_args()

    path = preview_augmentations(args.image, args.out)
    print(f"미리보기 저장: {path}")