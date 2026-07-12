"""Tests for synthetic data generator."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np

from training.generate_synthetic_data import (
    _build_kanji_pool,
    _find_available_fonts,
    _find_font,
    _load_ken_all,
    _random_address,
    _random_era_date,
    _random_kanji_address,
    _random_kanji_name,
    _random_license_number,
    _random_name,
    _apply_distortions,
    _draw_field,
    crop_lines_from_image,
    generate_one,
)


def test_build_kanji_pool():
    """_build_kanji_pool should return a non-empty string of kanji."""
    pool = _build_kanji_pool()
    assert len(pool) > 100
    # Should contain common kanji
    assert "山" in pool or "田" in pool


def test_find_font():
    """_find_font should return a path to an existing font."""
    font = _find_font()
    assert Path(font).exists()


def test_find_available_fonts():
    """_find_available_fonts should return at least one font."""
    fonts = _find_available_fonts()
    assert len(fonts) >= 1
    for f in fonts:
        assert Path(f).exists()


def test_random_license_number():
    """_random_license_number should return 12 digits."""
    num = _random_license_number()
    assert len(num) == 12
    assert num.isdigit()


def test_random_kanji_name():
    """_random_kanji_name should return a string of kanji."""
    pool = _build_kanji_pool()
    name = _random_kanji_name(pool, min_len=2, max_len=4)
    assert 2 <= len(name) <= 4


def test_random_kanji_address():
    """_random_kanji_address should return an address-like string."""
    pool = _build_kanji_pool()
    addr = _random_kanji_address(pool)
    assert "-" in addr
    # Should contain one of the prefecture prefixes
    from training.generate_synthetic_data import _DUMMY_ADDRESSES
    assert any(pref in addr for pref in ["東京都", "大阪府", "神奈川県", "愛知県",
                                          "京都府", "兵庫県", "福岡県", "北海道",
                                          "宮城県", "広島県", "静岡県", "茨城県",
                                          "栃木県", "千葉県", "埼玉県"])


def test_random_era_date():
    """_random_era_date should return a date in era format."""
    for era in ["昭和", "平成", "令和"]:
        date = _random_era_date(era)
        assert era in date
        assert "年" in date
        assert "月" in date
        assert "日" in date


def test_random_era_date_reiwa_range():
    """_random_era_date for 令和 should return year 1-8."""
    for _ in range(100):
        date = _random_era_date("令和")
        year_part = date.replace("令和", "").split("年")[0]
        assert 1 <= int(year_part) <= 8


def test_random_name():
    """_random_name should return surname + given name."""
    name = _random_name()
    assert len(name) >= 2
    # Should start with a known surname
    from training.generate_synthetic_data import _DUMMY_SURNAMES
    assert any(name.startswith(s) for s in _DUMMY_SURNAMES)


def test_random_address():
    """_random_address should return an address from the pool."""
    pool = ["東京都千代田区1-1-1", "大阪府大阪市1-2-3"]
    addr = _random_address(pool)
    assert addr in pool


def test_load_ken_all_fallback():
    """_load_ken_all should fall back to dummy addresses when CSV not found."""
    with patch("training.generate_synthetic_data._KEN_ALL_PATHS", ["/nonexistent/path"]):
        addresses = _load_ken_all()
    assert len(addresses) > 0
    assert "東京都" in addresses[0] or any("東京都" in a for a in addresses)


def test_apply_distortions():
    """_apply_distortions should return same-shape image."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    result = _apply_distortions(img)
    assert result.shape == (100, 200, 3)


def test_apply_distortions_no_blur():
    """_apply_distortions should work with blur disabled."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    result = _apply_distortions(img, enable_blur=False)
    assert result.shape == (100, 200, 3)


def test_apply_distortions_no_jpeg():
    """_apply_distortions should work with JPEG disabled."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    result = _apply_distortions(img, enable_jpeg=False)
    assert result.shape == (100, 200, 3)


def test_generate_one_basic():
    """generate_one should return image and lines."""
    image, lines = generate_one(width=400, height=200)
    assert image.shape == (200, 400, 3)
    assert len(lines) == 8


def test_generate_one_with_address_pool():
    """generate_one should use address from pool when provided."""
    pool = ["東京都千代田区1-1-1"]
    image, lines = generate_one(width=400, height=200, address_pool=pool)
    # Find address line
    addr_lines = [l for l in lines if "住所" in l[0]]
    assert len(addr_lines) == 1
    assert pool[0] in addr_lines[0][0]


def test_generate_one_kanji_boost():
    """generate_one with kanji_boost should use kanji pool for names."""
    image, lines = generate_one(width=400, height=200, kanji_boost=True)
    assert len(lines) == 8


def test_crop_lines_from_image():
    """crop_lines_from_image should return crops for each line."""
    image = np.ones((800, 400, 3), dtype=np.uint8) * 128
    line_texts = [("line1", "name"), ("line2", "address")]
    crops = crop_lines_from_image(image, line_texts)
    assert len(crops) == 2
    assert crops[0][1] == "line1"
    assert crops[1][1] == "line2"


def test_crop_lines_from_image_empty():
    """crop_lines_from_image should handle empty list."""
    image = np.ones((100, 200, 3), dtype=np.uint8) * 128
    crops = crop_lines_from_image(image, [])
    assert len(crops) == 0


def test_find_available_fonts_no_fonts():
    """_find_available_fonts should raise FileNotFoundError when no fonts."""
    with patch("training.generate_synthetic_data._FONT_CANDIDATES", ["/nonexistent/font.ttf"]):
        import pytest
        with pytest.raises(FileNotFoundError):
            _find_available_fonts()


def test_load_ken_all_with_csv(tmp_path):
    """_load_ken_all should parse a mock KEN_ALL.CSV."""
    # KEN_ALL.CSV format: [jis_zip, zip5, pref_kana, city_kana, town_kana,
    #   prefecture, city, town, ...]
    csv_content = (
        "0600001,0600001,ホッカイドウ,サッポロシチュウオウク,キタヒトシジョウニシ,北海道,札幌市中央区,北一条西,1,1,0,0,0,0,0\n"
        "1000001,1000001,トウキョウト,チヨダク,チヨダ,東京都,千代田区,千代田,1,1,0,0,0,0,0\n"
    )
    csv_path = tmp_path / "KEN_ALL.CSV"
    csv_path.write_text(csv_content, encoding="shift_jis")
    with patch("training.generate_synthetic_data._KEN_ALL_PATHS", [str(csv_path)]):
        addresses = _load_ken_all()
    assert len(addresses) == 2
    assert "北海道" in addresses[0]
    assert "東京都" in addresses[1]


def test_load_ken_all_corrupt_csv(tmp_path):
    """_load_ken_all should fall back when CSV is corrupt."""
    csv_path = tmp_path / "KEN_ALL.CSV"
    csv_path.write_text("corrupt data\n", encoding="shift_jis")
    with patch("training.generate_synthetic_data._KEN_ALL_PATHS", [str(csv_path)]):
        addresses = _load_ken_all()
    assert len(addresses) > 0  # should fall back


def test_load_ken_all_empty_rows(tmp_path):
    """_load_ken_all should skip rows with insufficient columns."""
    csv_content = "only,three,columns\n"
    csv_path = tmp_path / "KEN_ALL.CSV"
    csv_path.write_text(csv_content, encoding="shift_jis")
    with patch("training.generate_synthetic_data._KEN_ALL_PATHS", [str(csv_path)]):
        addresses = _load_ken_all()
    assert len(addresses) > 0  # should fall back to dummy


def test_load_ken_all_no_town(tmp_path):
    """_load_ken_all should handle rows with empty town field."""
    csv_content = (
        "1000001,1000001,トウキョウト,チヨダク,,東京都,千代田区,,1,1,0,0,0,0,0\n"
    )
    csv_path = tmp_path / "KEN_ALL.CSV"
    csv_path.write_text(csv_content, encoding="shift_jis")
    with patch("training.generate_synthetic_data._KEN_ALL_PATHS", [str(csv_path)]):
        addresses = _load_ken_all()
    assert len(addresses) == 1
    assert "東京都千代田区" in addresses[0]


def test_load_ken_all_kokonai(tmp_path):
    """_load_ken_all should handle '以下に掲載がない' town field."""
    csv_content = (
        "1000001,1000001,トウキョウト,チヨダク,以下に掲載がない,東京都,千代田区,以下に掲載がない,1,1,0,0,0,0,0\n"
    )
    csv_path = tmp_path / "KEN_ALL.CSV"
    csv_path.write_text(csv_content, encoding="shift_jis")
    with patch("training.generate_synthetic_data._KEN_ALL_PATHS", [str(csv_path)]):
        addresses = _load_ken_all()
    assert len(addresses) == 1
    assert "東京都千代田区" in addresses[0]


def test_load_ken_all_exception(tmp_path):
    """_load_ken_all should handle exceptions gracefully."""
    csv_path = tmp_path / "KEN_ALL.CSV"
    csv_path.write_text("valid content", encoding="shift_jis")
    with patch("training.generate_synthetic_data._KEN_ALL_PATHS", [str(csv_path)]), \
         patch("builtins.open", side_effect=IOError("mock error")):
        addresses = _load_ken_all()
    assert len(addresses) > 0  # should fall back


def test_draw_field():
    """_draw_field should draw text and return correct tuple."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (400, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(_find_font(), 32)

    y_offset, full_text = _draw_field(draw, font, "氏名", "山田太郎", (10, 10))
    assert y_offset == 10
    assert full_text == "氏名 山田太郎"


def test_apply_distortions_all_enabled():
    """_apply_distortions should work with all augmentations enabled."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 128
    result = _apply_distortions(img, enable_blur=True, enable_jpeg=True, enable_perspective=True)
    assert result.shape == (200, 400, 3)


def test_apply_distortions_gaussian_blur():
    """_apply_distortions should apply gaussian blur."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 128
    with patch("random.random", side_effect=[0, 0, 0, 0, 0.2]), \
         patch("random.choice", side_effect=["gaussian", 3]):
        result = _apply_distortions(img, enable_blur=True, enable_jpeg=False)
    assert result.shape == (200, 400, 3)


def test_apply_distortions_motion_blur():
    """_apply_distortions should apply motion blur."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 128
    with patch("random.random", side_effect=[0, 0, 0, 0, 0.2]), \
         patch("random.choice", side_effect=["motion", 5]):
        result = _apply_distortions(img, enable_blur=True, enable_jpeg=False)
    assert result.shape == (200, 400, 3)


def test_random_era_date_unknown_era():
    """_random_era_date should handle unknown eras with fallback range."""
    date = _random_era_date("明治")
    assert "明治" in date


def test_load_ken_all_empty_prefecture(tmp_path):
    """_load_ken_all should skip rows with empty prefecture."""
    csv_content = (
        "1000001,1000001,トウキョウト,チヨダク,チヨダ,,千代田区,千代田,1,1,0,0,0,0,0\n"
    )
    csv_path = tmp_path / "KEN_ALL.CSV"
    csv_path.write_text(csv_content, encoding="shift_jis")
    with patch("training.generate_synthetic_data._KEN_ALL_PATHS", [str(csv_path)]):
        addresses = _load_ken_all()
    # Empty prefecture → skipped, should fall back
    assert len(addresses) > 0


def test_generate_one_with_font():
    """generate_one should accept explicit font path."""
    font = _find_font()
    image, lines = generate_one(width=400, height=200, font_path=font)
    assert image.shape == (200, 400, 3)
    assert len(lines) == 8


def test_main_generates_data(tmp_path):
    """main() should generate images and labels."""
    old_argv = sys.argv
    output_dir = tmp_path / "synthetic"
    sys.argv = [
        "generate_synthetic_data.py",
        "--count", "2",
        "--output", str(output_dir),
        "--seed", "42",
    ]
    try:
        from training.generate_synthetic_data import main
        ret = main()
        assert ret == 0
        # Check files were created
        assert (output_dir / "labels.json").exists()
        assert (output_dir / "generation_summary.json").exists()
        assert (output_dir / "images").exists()
        with open(output_dir / "labels.json") as f:
            labels = json.load(f)
        assert len(labels) == 16  # 2 docs × 8 lines
    finally:
        sys.argv = old_argv


def test_main_with_kanji_boost(tmp_path):
    """main() should work with --kanji_boost."""
    old_argv = sys.argv
    output_dir = tmp_path / "synthetic_boost"
    sys.argv = [
        "generate_synthetic_data.py",
        "--count", "1",
        "--output", str(output_dir),
        "--kanji_boost",
        "--seed", "42",
    ]
    try:
        from training.generate_synthetic_data import main
        ret = main()
        assert ret == 0
        with open(output_dir / "generation_summary.json") as f:
            summary = json.load(f)
        assert summary["kanji_boost"] is True
    finally:
        sys.argv = old_argv


def test_main_with_no_blur_no_jpeg(tmp_path):
    """main() should work with --no_blur and --no_jpeg."""
    old_argv = sys.argv
    output_dir = tmp_path / "synthetic_noaug"
    sys.argv = [
        "generate_synthetic_data.py",
        "--count", "1",
        "--output", str(output_dir),
        "--no_blur",
        "--no_jpeg",
        "--seed", "42",
    ]
    try:
        from training.generate_synthetic_data import main
        ret = main()
        assert ret == 0
    finally:
        sys.argv = old_argv