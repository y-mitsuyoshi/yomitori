"""Synthetic document image generator for TrOCR training data.

Generates fake driver's-license-like images with random text, distortions,
and noise. Uses Japanese system fonts (e.g. Noto Sans CJK JP).

商用対応: 郵便局KEN_ALL.CSV（全国住所）と拡充された人名データを使用し、
多様なフォント・ノイズ・歪みを適用して商用レベルの学習データを生成する。
"""

import argparse
import csv
import json
import os
import random
import string
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── 人名データ（商用向け拡充版）─────────────────────────────────────
# 全国の名字トップ300（厚生労働省「苗字ベスト1000」より抜粋）
_DUMMY_SURNAMES = [
    "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水",
    "山崎", "森", "池田", "橋本", "阿部", "石川", "前田", "藤田", "岡田", "後藤",
    "原", "中島", "小島", "松田", "竹内", "長谷川", "片山", "本田", "大塚", "村田",
    "岡本", "高田", "中野", "石橋", "宮本", "杉山", "中山", "福田", "遠藤", "太田",
    "斉藤", "三浦", "藤原", "松井", "小川", "和田", "高木", "宮崎", "中西", "石田",
    "上田", "小野", "田村", "永井", "金子", "近藤", "土屋", "宮田", "丸山", "今井",
    "大野", "矢野", "高野", "三好", "安藤", "小松", "河野", "野村", "新井", "松村",
    "木下", "森田", "原田", "酒井", "菊池", "谷口", "青木", "柴田", "安田", "小山",
    "内田", "金井", "荒木", "田口", "岩崎", "杉本", "望月", "平田", "島田", "渡部",
    "西村", "小田", "服部", "中川", "梅田", "飯田", "荒井", "鈴木", "大島", "川口",
    "神田", "木村", "松崎", "森本", "本多", "吉川", "中井", "矢口", "横山", "山内",
    "浅野", "水野", "大西", "黒田", "田辺", "川島", "村上", "佐久間", "石井", "早川",
    "松浦", "長田", "小野寺", "熊谷", "富田", "青木", "西田", "久保", "八木", "加茂",
    "真田", "赤松", "市川", "川田", "神谷", "黒川", "中田", "西川", "根本", "今村",
    "久米", "松木", "松崎", "山下", "増田", "藤井", "野口", "坂本", "石崎", "瀬戸",
    "三上", "西尾", "松尾", "岸本", "中西", "大木", "吉村", "松下", "鶴田", "小島",
    "上野", "笠井", "中澤", "武田", "佐野", "畑中", "中条", "安東", "永田", "宮下",
    "森山", "大橋", "本間", "田島", "永山", "横田", "秋山", "柳田", "中谷", "大久保",
    "松永", "中尾", "石原", "森川", "三輪", "五十嵐", "川上", "山脇", "宮川", "増田",
    "杉田", "小原", "船橋", "日高", "佐伯", "香川", "井出", "松崎", "市村", "飯塚",
    "坂田", "北村", "稲田", "中嶋", "松村", "宮島", "原口", "木原", "出口", "河合",
    "木村", "川端", "平井", "桑原", "成田", "小倉", "沼田", "栗原", "植木", "松島",
    "江口", "足立", "佐竹", "篠田", "佐々木", "東", "的場", "安齋", "浅見", "豊田",
]

# 名前データ（男女混合・幅広い年代カバー）
_DUMMY_GIVEN_NAMES = [
    # 男性名（昭和〜令和世代）
    "太郎", "一郎", "二郎", "三郎", "健太", "健一", "直樹", "大輔", "翔太", "翔",
    "拓海", "蓮", "悠真", "海斗", "颯太", "大和", "陽翔", "悠人", "奏多", "心春",
    "颯", "朝陽", "湊", "樹", "瑛太", "陽斗", "大翔", "結翔", "悠斗", "陽向",
    "光", "純一", "誠", "勇気", "隆", "明", "茂", "正", "和彦", "俊介",
    "徹", "淳", "聡", "哲也", "秀樹", "英樹", "昌彦", "雅彦", "典彦", "良介",
    "和也", "拓也", "達也", "裕也", "将也", "直也", "亮", "亮太", "亮介", "将太",
    "圭介", "健介", "洋介", "翔介", "涼介", "颯介", "大樹", "直輝", "拓真", "海翔",
    # 女性名（昭和〜令和世代）
    "花子", "美咲", "百合子", "陽子", "結衣", "陽菜", "芽依", "紗枝", "彩花", "莉子",
    "美月", "日葵", "陽葵", "凛", "紬", "芽", "咲", "楓", "葵", "結菜",
    "美穂", "幸子", "和子", "恵子", "久美子", "由美子", "裕子", "京子", "明子", "佳子",
    "真理", "亜紀", "由紀", "幸恵", "富美子", "千代", "千鶴", "千尋", "美和", "美紀",
    "真由美", "恵", "直美", "友美", "裕美", "佳奈", "彩乃", "愛", "結愛", "美桜",
    "朱里", "陽彩", "心愛", "結羽", "澪", "咲良", "美羽", "莉央", "朱音", "美桜",
]

# ─── 郵便局 KEN_ALL.CSV 読み込み ─────────────────────────────────────
_KEN_ALL_PATHS = [
    "/opt/ml/data/postal/KEN_ALL.CSV",
    "/opt/ml/code/data/postal/KEN_ALL.CSV",
    "data/postal/KEN_ALL.CSV",
]

_DUMMY_ADDRESSES = [
    "東京都千代田区霞が関1-1-1",
    "東京都新宿区西新宿2-8-1",
    "東京都品川区上大崎4-6-10",
    "東京都大田区蒲田5-1-1",
    "東京都世田谷区玉川1-2-3",
    "大阪府大阪市中央区大手前1-2-3",
    "大阪府大阪市北区梅田1-1-3",
    "大阪府大阪市天王寺区上本町5-1-1",
    "神奈川県横浜市中区桜丘1-5-10",
    "神奈川県横浜市西区高島2-1-1",
    "神奈川県川崎市幸区堀川町1-1",
    "愛知県名古屋市中区三の丸3-2-1",
    "愛知県名古屋市东区東桜1-1-1",
    "京都府京都市上京区烏丸通一条1-1",
    "京都府京都市中京区烏丸通御池1-1",
    "兵庫県神戸市中央区加納町6-5-1",
    "福岡県福岡市中央区天神1-1-1",
    "福岡県福岡市博多区博多駅前1-1",
    "北海道札幌市中央区北一条西5-1",
    "宮城県仙台市青葉区本町1-1-1",
    "広島県広島市中区基町1-1",
    "静岡県静岡市葵区追手町5-1",
    "茨城県水戸市笠間町1-1",
    "栃木県宇都宮市栄町1-1",
    "千葉県千葉市中央区市場町1-1",
]

_ERAS = ["昭和", "平成", "令和"]
_LICENSE_TYPES = ["普通", "中型", "大型", "自動二輪", "原付", "大型二輪", "けん引"]
_CONDITIONS = [
    "眼鏡等", "普通", "中型", "大型", "自動二輪", "原付", "大型二輪",
    "第一種普通", "第二種普通", "第一種中型", "第二種中型",
    "第一種大型", "第二種大型", "第一種自動二輪", "第二種自動二輪",
    "第一種原付", "第二種原付", "第一種大型二輪", "第二種大型二輪",
    "けん引", "小型限定", "普通自動二輪", "AT限定",
    "", "", "",
]

# ─── フォント（複数対応）──────────────────────────────────────────
_FONT_CANDIDATES = [
    # Noto Sans CJK（ゴシック体）
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    # Noto Serif CJK（明朝体）
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    # IPAフォント
    "/usr/share/fonts/ipa/ipagp.ttf",      # IPAゴシック
    "/usr/share/fonts/ipa/ipam.ttf",       # IPA明朝
    "/usr/share/fonts/ipaexfont/ipaexg.ttf",  # IPAexゴシック
    "/usr/share/fonts/ipaexfont/ipaexm.ttf",   # IPAex明朝
    # Takaoフォント
    "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
    "/usr/share/fonts/truetype/takao-mincho/TakaoMincho.ttf",
]

# 漢字プール（未知の漢字対応用）
def _build_kanji_pool() -> str:
    """JIS第1水準漢字の範囲から漢字プールを構築する。

    Returns:
        漢字文字列（重複なし）。
    """
    chars = set()
    for cp in range(0x4E00, 0x62FF + 1):
        ch = chr(cp)
        if "\u4E00" <= ch <= "\u9FFF":
            chars.add(ch)
    return "".join(sorted(chars))


_KANJI_POOL = _build_kanji_pool()


def _find_available_fonts() -> list[str]:
    """インストールされている日本語フォントのリストを返す。

    Returns:
        利用可能なフォントファイルパスのリスト。

    Raises:
        FileNotFoundError: フォントが1つも見つからない場合。
    """
    available = []
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            available.append(p)
    if not available:
        raise FileNotFoundError(
            "No Japanese font found. Install Noto Sans CJK JP or IPA fonts."
        )
    return available


def _find_font() -> str:
    """最初に見つかった日本語フォントを返す（後方互換用）。

    Returns:
        フォントファイルパス。
    """
    return _find_available_fonts()[0]


def _load_ken_all() -> list[str]:
    """郵便局のKEN_ALL.CSVを読み込み、住所リストを返す。

    KEN_ALL.CSVはShift_JISエンコーディング。
    列: [郵便番号, 都道府県カナ, 市区町村カナ, 町域カナ, 都道府県, 市区町村, 町域, ...]

    Returns:
        住所文字列のリスト（"東京都千代田区霞が関1-1" 形式）。
    """
    for path in _KEN_ALL_PATHS:
        if not Path(path).exists():
            continue
        try:
            addresses = []
            with open(path, "r", encoding="shift_jis", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 8:
                        continue
                    # KEN_ALL.CSV: 列5=都道府県, 列6=市区町村, 列7=町域
                    prefecture = row[5].strip()
                    city = row[6].strip()
                    town = row[7].strip()

                    if not prefecture or not city:
                        continue

                    # 町域が空白または以下の場合はスキップ
                    if not town or "以下に掲載がない" in town:
                        addr = f"{prefecture}{city}{random.randint(1, 30)}-{random.randint(1, 30)}-{random.randint(1, 30)}"
                    else:
                        # 町域名から番地部分を削除
                        town_clean = town.split("（")[0].strip()
                        chome = random.randint(1, 20)
                        ban = random.randint(1, 30)
                        go = random.randint(1, 30)
                        addr = f"{prefecture}{city}{town_clean}{chome}-{ban}-{go}"

                    addresses.append(addr)

            if addresses:
                logger.info("Loaded %d addresses from KEN_ALL.CSV (%s)", len(addresses), path)
                return addresses
        except Exception as e:
            logger.warning("Failed to load KEN_ALL.CSV from %s: %s", path, e)

    logger.warning("KEN_ALL.CSV not found. Using fallback addresses (%d entries).", len(_DUMMY_ADDRESSES))
    return _DUMMY_ADDRESSES


# ─── ランダムデータ生成 ───────────────────────────────────────────
def _random_license_number() -> str:
    """Generate a 12-digit license number.

    Returns:
        Random 12-digit string.
    """
    return "".join(random.choices(string.digits, k=12))


def _random_kanji_name(kanji_pool: str, min_len: int = 2, max_len: int = 4) -> str:
    """Generate a random kanji name from a kanji pool.

    Args:
        kanji_pool: String of kanji characters to sample from.
        min_len: Minimum name length in characters.
        max_len: Maximum name length in characters.

    Returns:
        Random kanji string.
    """
    length = random.randint(min_len, max_len)
    return "".join(random.choice(kanji_pool) for _ in range(length))


def _random_kanji_address(kanji_pool: str) -> str:
    """Generate a random address-like string using kanji from the pool.

    Args:
        kanji_pool: String of kanji characters to sample from.

    Returns:
        Random address string.
    """
    prefectures = [
        "東京都", "大阪府", "神奈川県", "愛知県", "京都府",
        "兵庫県", "福岡県", "北海道", "宮城県", "広島県",
        "静岡県", "茨城県", "栃木県", "千葉県", "埼玉県",
    ]
    district = "".join(random.choice(kanji_pool) for _ in range(random.randint(2, 5)))
    chome = random.randint(1, 9)
    ban = random.randint(1, 20)
    go = random.randint(1, 30)
    return f"{random.choice(prefectures)}{district}{chome}-{ban}-{go}"


def _random_era_date(era: str) -> str:
    """Generate a random date in Japanese-era format.

    Args:
        era: Era name.

    Returns:
        Date string like ``"昭和61年5月1日"``.
    """
    if era == "令和":
        year = random.randint(1, 8)
    elif era == "平成":
        year = random.randint(1, 31)
    elif era == "昭和":
        year = random.randint(1, 64)
    else:
        year = random.randint(1, 30)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{era}{year}年{month}月{day}日"


def _random_name() -> str:
    """ダミーリストから氏名を生成する。

    Returns:
        氏名文字列（名字+名前）。
    """
    return random.choice(_DUMMY_SURNAMES) + random.choice(_DUMMY_GIVEN_NAMES)


def _random_address(address_pool: list[str]) -> str:
    """住所プールからランダムに1件選ぶ。

    Args:
        address_pool: 住所文字列のリスト。

    Returns:
        住所文字列。
    """
    return random.choice(address_pool)


# ─── 画像 Augmentation（商用向け強化版）─────────────────────────
def _apply_distortions(
    image: np.ndarray,
    enable_blur: bool = True,
    enable_jpeg: bool = True,
    enable_perspective: bool = True,
) -> np.ndarray:
    """Apply random distortions to simulate real-world photo conditions.

    以下の変形をランダムに組み合わせて適用する：
    - 回転（±15°）
    - 明るさ・コントラスト変動
    - ガウシアンノイズ
    - 反射オーバーレイ
    - ぼかし（Blur）
    - JPEG圧縮劣化
    - 透視変形（Perspective）
    - 影オーバーレイ

    Args:
        image: Input image.
        enable_blur: ぼかしを有効化。
        enable_jpeg: JPEG圧縮劣化を有効化。
        enable_perspective: 透視変形を有効化。

    Returns:
        Augmented image.
    """
    h, w = image.shape[:2]

    # Rotation ±15°
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Brightness/contrast variation
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-30, 30)
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    # Partial reflection overlay
    if random.random() < 0.3:
        overlay = image.copy()
        cv2.ellipse(
            overlay,
            (random.randint(0, w), random.randint(0, h)),
            (random.randint(50, 200), random.randint(20, 80)),
            0, 0, 360,
            (200, 200, 200),
            -1,
        )
        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

    # Blur（ピンボケ・手ブレ再現）
    if enable_blur and random.random() < 0.3:
        blur_type = random.choice(["gaussian", "motion"])
        if blur_type == "gaussian":
            kernel_size = random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            # Motion blur（水平方向の手ブレ）
            kernel_size = random.randint(5, 15)
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1.0
            kernel /= kernel_size
            image = cv2.filter2D(image, -1, kernel)

    # JPEG圧縮劣化（スマホ撮影の画質低下再現）
    if enable_jpeg and random.random() < 0.3:
        quality = random.randint(30, 70)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # 影オーバーレイ（照明ムラ再現）
    if random.random() < 0.2:
        overlay = image.copy()
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        radius = random.randint(min(h, w) // 3, min(h, w) // 2)
        cv2.circle(overlay, (center_x, center_y), radius, (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

    return image


def _draw_field(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    label: str,
    value: str,
    pos: tuple[int, int],
    label_color: tuple[int, int, int] = (0, 0, 0),
    value_color: tuple[int, int, int] = (20, 20, 20),
) -> tuple[int, str]:
    """Draw a label + value field on the image.

    Args:
        draw: PIL ImageDraw object.
        font: Font to use.
        label: Label string.
        value: Value string.
        pos: (x, y) top-left position.
        label_color: Label text color.
        value_color: Value text color.

    Returns:
        Tuple of (y_offset, full_text) for label pairing.
    """
    x, y = pos
    draw.text((x, y), label, fill=label_color, font=font)
    bbox = draw.textbbox((x, y), label, font=font)
    value_x = bbox[2] + 8
    draw.text((value_x, y), value, fill=value_color, font=font)
    return y, f"{label} {value}"


def generate_one(
    width: int = 2400,
    height: int = 1512,
    font_path: str | None = None,
    kanji_boost: bool = False,
    address_pool: list[str] | None = None,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Generate a single synthetic driver's-license-like image.

    Args:
        width: Image width.
        height: Image height.
        font_path: Path to font file. None ならランダムに選択。
        kanji_boost: If True, use random kanji from the full CJK pool instead of
            the fixed dummy lists, to improve recognition of rare kanji.
        address_pool: 住所プール（KEN_ALL.CSVから読み込み）。None ならフォールバック。

    Returns:
        Tuple of (image_bgr, list_of_(line_text, field_name)).
    """
    # フォント選択（指定なければランダム）
    if font_path is None:
        available_fonts = _find_available_fonts()
        font_path = random.choice(available_fonts)

    font = ImageFont.truetype(font_path, 48)
    font_small = ImageFont.truetype(font_path, 32)

    pil = Image.new("RGB", (width, height), (240, 240, 245))
    draw = ImageDraw.Draw(pil)

    # Photo placeholder (right side)
    draw.rectangle(
        [width - 700, 80, width - 80, 600],
        fill=(200, 200, 210),
        outline=(100, 100, 100),
        width=3,
    )
    draw.text((width - 480, 320), "写真", fill=(120, 120, 120), font=font)

    # Fields
    if kanji_boost:
        name = _random_kanji_name(_KANJI_POOL)
        address = _random_kanji_address(_KANJI_POOL)
    else:
        name = _random_name()
        if address_pool:
            address = _random_address(address_pool)
        else:
            address = random.choice(_DUMMY_ADDRESSES)

    birth_era = random.choice(_ERAS)
    birth_date = _random_era_date(birth_era) + "生"
    issue_era = random.choice(_ERAS)
    issue_date = _random_era_date(issue_era)
    expiry_era = random.choice(_ERAS)
    expiry_date = _random_era_date(expiry_era)
    license_number = _random_license_number()
    license_type = random.choice(_LICENSE_TYPES)
    condition = random.choice(_CONDITIONS)

    lines: list[tuple[str, str]] = []
    y = 60
    for label, value, fname in [
        ("氏名", name, "name"),
        ("生年月日", birth_date, "birth_date"),
        ("住所", address, "address"),
        ("交付", issue_date, "issue_date"),
        ("有効期限", expiry_date, "expiry_date"),
        ("条件等", condition, "conditions"),
        ("免許証番号", license_number, "license_number"),
        ("免許種類", license_type, "license_type"),
    ]:
        draw.text((60, y), label, fill=(0, 0, 0), font=font)
        bbox = draw.textbbox((60, y), label, font=font)
        draw.text((bbox[2] + 16, y), value, fill=(20, 20, 20), font=font)
        lines.append((f"{label} {value}", fname))
        y += 100

    # Header
    draw.text((60, height - 80), "運転免許証", fill=(0, 0, 100), font=font_small)

    image = np.array(pil)
    image = _apply_distortions(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, lines


def crop_lines_from_image(
    image: np.ndarray,
    line_texts: list[tuple[str, str]],
) -> list[tuple[np.ndarray, str, str]]:
    """Crop text lines from a synthetic image (simple fixed layout).

    Args:
        image: Full document image (BGR).
        line_texts: List of (full_text, field_name).

    Returns:
        List of (line_image, text, field_name).
    """
    h, w = image.shape[:2]
    line_h = h // len(line_texts) if line_texts else h
    crops: list[tuple[np.ndarray, str, str]] = []
    for i, (text, fname) in enumerate(line_texts):
        y0 = i * line_h
        y1 = (i + 1) * line_h
        crop = image[y0:y1, :]
        crops.append((crop, text, fname))
    return crops


def main() -> int:
    """CLI entry point for synthetic data generation.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic document images for training"
    )
    parser.add_argument(
        "--document_type",
        default="driver_license_front",
        help="Document type to generate",
    )
    parser.add_argument("--count", type=int, default=500, help="Number of images")
    parser.add_argument(
        "--output",
        default="data/synthetic/driver_license/",
        help="Output directory",
    )
    parser.add_argument("--font", default=None, help="Font file path (default: random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--kanji_boost",
        action="store_true",
        help="Use random kanji from the full CJK pool (instead of fixed dummy lists) "
             "to improve recognition of rare/unknown kanji characters",
    )
    parser.add_argument(
        "--no_blur",
        action="store_true",
        help="Disable blur augmentation",
    )
    parser.add_argument(
        "--no_jpeg",
        action="store_true",
        help="Disable JPEG compression augmentation",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # KEN_ALL.CSVから住所を読み込み
    address_pool = _load_ken_all()

    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating %d synthetic documents (kanji_boost=%s, addresses=%d, fonts=multiple)",
        args.count, args.kanji_boost, len(address_pool),
    )

    labels: dict[str, str] = {}
    for i in range(args.count):
        image, lines = generate_one(
            font_path=args.font,
            kanji_boost=args.kanji_boost,
            address_pool=address_pool,
        )
        crops = crop_lines_from_image(image, lines)
        for j, (crop, text, _fname) in enumerate(crops):
            name = f"{i:05d}_{j}.png"
            cv2.imwrite(str(out / "images" / name), crop)
            labels[name] = text

    with (out / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # データサマリーを保存
    summary = {
        "count": args.count,
        "line_crops": len(labels),
        "surnames": len(_DUMMY_SURNAMES),
        "given_names": len(_DUMMY_GIVEN_NAMES),
        "addresses": len(address_pool),
        "kanji_boost": args.kanji_boost,
        "fonts": len(_find_available_fonts()),
        "address_source": "KEN_ALL.CSV" if address_pool != _DUMMY_ADDRESSES else "fallback",
    }
    with (out / "generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "Generated %d documents (%d line crops) in %s "
        "(surnames=%d, given_names=%d, addresses=%d, fonts=%d)",
        args.count, len(labels), out,
        len(_DUMMY_SURNAMES), len(_DUMMY_GIVEN_NAMES), len(address_pool),
        len(_find_available_fonts()),
    )
    print(f"Generated {len(labels)} line crops → {out}")
    print(f"  Surnames: {len(_DUMMY_SURNAMES)}")
    print(f"  Given names: {len(_DUMMY_GIVEN_NAMES)}")
    print(f"  Addresses: {len(address_pool)}")
    print(f"  Fonts: {len(_find_available_fonts())}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())