import io
import zipfile

from PIL import Image

from industrialguard.data.archive_reader import ArchiveReader
from industrialguard.data.discovery import discover_images
from industrialguard.data.validation import validate_records


def image_bytes(color=(0, 0, 0)):
    stream = io.BytesIO()
    Image.new("L", (4, 3), color).save(stream, format="PNG")
    return stream.getvalue()


def test_validates_corruption_and_empty_mask_and_missing_annotation(tmp_path):
    path = tmp_path / "images.zip"
    files = {
        "thing/train/good/ok.png": image_bytes(0),
        "thing/test/dent/001.png": image_bytes(255),
        "thing/ground_truth/dent/001_mask.png": image_bytes(0),
        "thing/test/scratch/002.png": b"broken image",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    names = list(files)
    records, _ = discover_images("synthetic", names, ["thing"])
    result = validate_records(ArchiveReader(path, "zip"), records, names)
    issues = [item["issue"] for item in result["errors"]]
    warnings = [item["issue"] for item in result["warnings"]]
    assert "unreadable_image" in issues
    assert "missing_mask" in issues
    assert "empty_mask" in warnings
    assert result["counts"]["invalid_images"] == 1
    assert result["counts"]["empty_masks"] == 1
    assert result["counts"]["missing_masks"] == 1
