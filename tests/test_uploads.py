from io import BytesIO
from zipfile import ZipFile

import pytest

from face_analysis.uploads import records_from_uploads


def upload(name, content):
    file = BytesIO(content)
    file.name = name
    return file


def test_labels_come_only_from_user_filenames_or_folders():
    records = records_from_uploads([upload("Sample_A__01.jpg", b"a"), upload("unknown.jpg", b"b")])
    assert [r["person"] for r in records] == ["Sample A", ""]
    archive = BytesIO()
    with ZipFile(archive, "w") as zf:
        zf.writestr("My_label/01.jpg", b"a")
        zf.writestr("__MACOSX/My_label/._01.jpg", b"ignored")
    records = records_from_uploads([upload("faces.zip", archive.getvalue())])
    assert len(records) == 1
    assert records[0]["person"] == "My label"


def test_bad_zip_does_not_disappear_silently():
    with pytest.raises(ValueError, match="broken.zip"):
        records_from_uploads([upload("broken.zip", b"invalid")])
