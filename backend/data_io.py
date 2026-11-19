"""Keep the last complete JSON file when a rebuild is interrupted."""
from pathlib import Path
import json, os, tempfile


def write_json_atomic(path, value):
    path=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf8',dir=path.parent,prefix=path.name+'-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(value,stream,ensure_ascii=False,indent=2)
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
