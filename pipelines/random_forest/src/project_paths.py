"""All active writes stay in this clone and use exclusive creation."""
from pathlib import Path
import json
from .wrmsse_config import ROOT


def local_path(relative):
    value = Path(relative)
    if value.is_absolute() or value.drive:
        raise ValueError("Use a project-relative path")
    if value.parts[:2] == ("data", "raw") and not (ROOT / "data" / "raw").exists():
        shared_root = ROOT.parents[1]
        resolved = (shared_root / value).resolve()
        if not resolved.is_relative_to(shared_root / "data" / "raw") and resolved != (shared_root / "data" / "raw").resolve():
            raise ValueError("Path escapes shared data/raw")
        return resolved
    resolved = (ROOT / value).resolve()
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ValueError("Path escapes the clone")
    if resolved.relative_to(ROOT).parts[0] == "legacy":
        raise ValueError("Legacy artifacts cannot be used as active inputs or outputs")
    return resolved


def write_json_new(relative, value):
    target = local_path(relative)
    if target.relative_to(ROOT).parts[0] not in ("results", "preparation", "data"):
        raise ValueError("Output must be under results, preparation, or data")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Re-check after directory creation to reject junctions and symbolic links.
    target = local_path(relative)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return target
