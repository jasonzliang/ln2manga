import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

# make the package importable without an editable install
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ln2manga.config import load_settings  # noqa: E402


def tiny_png(size=(64, 96), color=200) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return tiny_png()


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.paths["data"] = str(tmp_path)        # isolate all I/O under tmp
    return s


def image_response(png: bytes):
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=base64.b64encode(png).decode(), url=None)],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class RecordingImages:
    def __init__(self):
        self.calls = []

    def edit(self, **kw):
        self.calls.append(("edit", kw))
        return image_response(tiny_png())

    def generate(self, **kw):
        self.calls.append(("generate", kw))
        return image_response(tiny_png())


class RecordingClient:
    def __init__(self):
        self.images = RecordingImages()


@pytest.fixture
def recording_client():
    return RecordingClient()
