from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile, BadZipFile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def infer_person_from_filename(filename: str) -> str:
    stem = PurePosixPath(filename).stem
    if "__" in stem:
        return stem.split("__", 1)[0].replace("_", " ").strip()
    return ""


def records_from_uploads(files) -> list[dict]:
    records: list[dict] = []
    for f in files or []:
        name = f.name
        suffix = PurePosixPath(name).suffix.lower()
        data = f.getvalue()

        if suffix == ".zip":
            try:
                zf = ZipFile(BytesIO(data))
            except BadZipFile as exc:
                raise ValueError(f"Could not read ZIP: {name}") from exc
            with zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    path = PurePosixPath(member.filename)
                    if path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if "__MACOSX" in path.parts or path.name.startswith("._"):
                        continue
                    parts = path.parts
                    person = parts[-2].replace("_", " ").strip() if len(parts) >= 2 else infer_person_from_filename(path.name)
                    try:
                        contents = zf.read(member)
                    except (BadZipFile, RuntimeError, OSError, NotImplementedError) as exc:
                        raise ValueError(f"Could not read {member.filename} from ZIP {name}") from exc
                    records.append({
                        "person": person,
                        "filename": str(path),
                        "bytes": contents,
                    })
        elif suffix in IMAGE_EXTENSIONS:
            records.append({
                "person": infer_person_from_filename(name),
                "filename": name,
                "bytes": data,
            })
    return records
