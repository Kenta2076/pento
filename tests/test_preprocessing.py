from __future__ import annotations

import numpy as np
import pytest

from pento import preprocessing


def test_load_image_logs_basic_stats(tmp_path, caplog):
    pytest.importorskip("PIL.Image", reason="Pillow is required for this test")
    from PIL import Image

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 2), color=(128, 64, 32)).save(image_path)

    with caplog.at_level("INFO"):
        array = preprocessing.load_image(image_path)

    assert array.shape == (2, 4, 3)
    assert array.dtype == np.float32
    assert any("Loaded image" in record.message for record in caplog.records)
