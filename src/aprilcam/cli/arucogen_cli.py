from __future__ import annotations

from typing import Optional, List

from ..arucogen import main as _main


def main(argv: Optional[List[str]] = None) -> int:
    return _main(argv)
