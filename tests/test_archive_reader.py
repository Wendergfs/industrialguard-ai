import io
import tarfile
import zipfile

from PIL import Image

from industrialguard.data.archive_reader import ArchiveReader


def png_bytes(color=(0, 0, 0)):
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(stream, format="PNG")
    return stream.getvalue()


def test_zip_reader_is_read_only_and_lists_files(tmp_path):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("cat/test/good/a.png", png_bytes())
    before = path.read_bytes()
    reader = ArchiveReader(path, "zip")
    assert reader.names() == ["cat/test/good/a.png"]
    assert reader.read_bytes("cat/test/good/a.png") == png_bytes()
    assert reader.test_integrity() is None
    assert path.read_bytes() == before


def test_tar_xz_reader_streams_members(tmp_path):
    path = tmp_path / "fixture.tar.xz"
    payload = png_bytes()
    with tarfile.open(path, "w:xz") as archive:
        info = tarfile.TarInfo("cat/train/good/a.png")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    reader = ArchiveReader(path, "tar.xz")
    assert reader.names() == ["cat/train/good/a.png"]
    assert list(reader.iter_member_bytes()) == [("cat/train/good/a.png", payload)]
    assert reader.test_integrity() is None


def test_invalid_archive_raises(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not a zip archive")
    try:
        ArchiveReader(path, "zip").names()
    except zipfile.BadZipFile:
        pass
    else:
        raise AssertionError("invalid archive should raise BadZipFile")
