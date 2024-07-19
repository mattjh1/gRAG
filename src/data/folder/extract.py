from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator

DATE_FORMAT: str = "%Y-%m-%d"


def directory(path: str, glob: str, since: date) -> Generator[Path, Any, None]:
    path_ = Path(path)
    for file in path_.glob(pattern=glob):
        last_modified = file.stat().st_mtime
        last_modified_date = datetime.fromtimestamp(last_modified)
        was_modified = last_modified_date >= since
        if was_modified:
            yield file
