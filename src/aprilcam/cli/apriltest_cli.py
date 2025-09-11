from __future__ import annotations

import argparse
import io
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from ..iohelpers import get_data_dir
from . import aprilcap_cli as aprilcap


@dataclass
class RunResult:
    cmd: str
    ids: Dict[int, Tuple[float, int, int, bool]]  # id -> (pct, hit, total, absent)
    avg_pct: float
    min_pct: float
    covered: List[int]


def parse_summary(stdout: str) -> Dict[int, Tuple[float, int, int, bool]]:
    """Parse lines like 'ID   6:  0.0%  (0/10)  (absent)'. Return map id -> (pct, hit, total, absent)."""
    out: Dict[int, Tuple[float, int, int, bool]] = {}
    rx = re.compile(r"^ID\s+(\d+):\s+([0-9]+\.[0-9])%\s+\((\d+)/(\d+)\)(?:\s+\(absent\))?$")
    # Be a bit more permissive on percent formatting
    rx2 = re.compile(r"^ID\s+(\d+):\s+([0-9]+(?:\.[0-9]+)?)%\s+\((\d+)/(\d+)\)(.*)$")
    for line in stdout.splitlines():
        m = rx.match(line.strip()) or rx2.match(line.strip())
        if not m:
            continue
        tid = int(m.group(1))
        pct = float(m.group(2))
        hit = int(m.group(3))
        tot = int(m.group(4))
        tail = m.group(5) if m.lastindex and m.lastindex >= 5 else ""
        absent = "absent" in (tail or "")
        out[tid] = (pct, hit, tot, absent)
    return out


def score_result(ids: Dict[int, Tuple[float, int, int, bool]], focus_ids: List[int]) -> Tuple[float, float, List[int]]:
    """Compute (avg_pct, min_pct, covered_ids) over the focus_ids present in the summary.
    If an ID is absent from the summary, treat as 0%.
    """
    vals: List[float] = []
    covered: List[int] = []
    for tid in focus_ids:
        if tid in ids:
            pct = ids[tid][0]
            covered.append(tid)
        else:
            pct = 0.0
        vals.append(float(pct))
    if not vals:
        return 0.0, 0.0, covered
    avg = sum(vals) / float(len(vals))
    mn = min(vals)
    return avg, mn, covered


def generate_default_commands() -> List[str]:
    ids = "--ids 0,1,2,3,4,5,6"
    base = [
        f"aprilcap {ids}",
        f"aprilcap {ids} --proc-width 1280",
        f"aprilcap {ids} --proc-width 960",
    ]
    # Smoothing and corner refine variants
    variants = [
        "--quad-sigma 0.0 --corner-refine subpix",
        "--quad-sigma 0.6 --corner-refine subpix",
        "--quad-sigma 0.8 --corner-refine subpix",
        "--quad-sigma 1.2 --corner-refine subpix",
        "--quad-sigma 0.6 --corner-refine contour",
    ]
    # Threshold sets
    thresh = [
        "--april-min-wb-diff 1 --april-min-cluster-pixels 3 --april-max-line-fit-mse 40",
        "--april-min-wb-diff 3 --april-min-cluster-pixels 5 --april-max-line-fit-mse 20",
    ]
    # Enhancements toggles
    enh = [
        "",
        "--sharpen",
        "--clahe",
    ]
    out: List[str] = []
    for b in base:
        for v in variants:
            for t in thresh:
                for e in enh:
                    cmd = " ".join(x for x in [b, v, t, e] if x)
                    out.append(cmd)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq[:60]  # keep it reasonable


def read_commands(path: Path, regen_if_missing: bool) -> List[str]:
    if not path.exists():
        if regen_if_missing:
            cmds = generate_default_commands()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(cmds) + "\n")
            return cmds
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    cmds: List[str] = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        cmds.append(ln)
    return cmds


def run_one(cmdline: str) -> Tuple[int, str]:
    """Run a single aprilcap command line by calling aprilcap.main with captured stdout.
    Returns (exit_code, stdout).
    """
    # Accept lines with or without leading 'aprilcap'
    parts = shlex.split(cmdline)
    if parts and parts[0].lower() == "aprilcap":
        parts = parts[1:]
    # Capture stdout
    buf = io.StringIO()
    import sys
    old = sys.stdout
    rc = 0
    try:
        sys.stdout = buf
        rc = int(aprilcap.main(parts))
    except SystemExit as e:
        rc = int(e.code or 0)
    except Exception:
        rc = 1
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="apriltest",
        description=(
            "Batch runner for aprilcap parameter sweeps. Reads commands from a file, runs them, and ranks the results.\n"
            "By default uses data/testcommands.txt; if missing, a sensible set is generated."
        ),
    )
    p.add_argument("--commands", type=str, default="testcommands.txt", help="Path under data/ or absolute to the commands file")
    p.add_argument("--regen", action="store_true", help="Regenerate the commands file with defaults before running")
    p.add_argument("--top", type=int, default=10, help="Show top N results (default 10)")
    p.add_argument("--focus-ids", type=str, default="0,1,2,3,4,5,6", help="IDs to score, comma-separated (default 0..6)")
    p.add_argument("--dry-run", action="store_true", help="Only show what would run; don't execute")
    args = p.parse_args(argv)

    data_dir = get_data_dir()
    cmd_path = Path(args.commands)
    if not cmd_path.is_absolute():
        cmd_path = data_dir / cmd_path

    cmds = read_commands(cmd_path, regen_if_missing=True if args.regen or not cmd_path.exists() else False)
    if not cmds:
        print(f"No commands found in {cmd_path}. Use --regen to create defaults.")
        return 2

    print(f"Running {len(cmds)} command(s) from {cmd_path}")
    if args.dry_run:
        for i, c in enumerate(cmds, 1):
            print(f"[{i:03d}] {c}")
        return 0

    focus_ids = []
    for part in (args.focus_ids or "").split(','):
        part = part.strip()
        if part:
            try:
                focus_ids.append(int(part))
            except Exception:
                pass
    if not focus_ids:
        focus_ids = [0, 1, 2, 3, 4, 5, 6]

    results: List[RunResult] = []
    for i, c in enumerate(cmds, 1):
        print(f"[{i:03d}/{len(cmds):03d}] {c}")
        rc, out = run_one(c)
        if rc != 0:
            print(f"  -> exited with {rc}; skipping")
            continue
        parsed = parse_summary(out)
        if not parsed:
            # Try list mode parse fallback: if a user put --list, skip scoring
            print("  -> no summary lines found; skipping")
            continue
        avg, mn, covered = score_result(parsed, focus_ids)
        results.append(RunResult(cmd=c, ids=parsed, avg_pct=avg, min_pct=mn, covered=covered))
        print(f"  -> avg {avg:.1f}%  min {mn:.1f}%  covered {len(covered)}/{len(focus_ids)}")

    if not results:
        print("No successful runs with summary output to score.")
        return 1

    # Sort by avg desc, then min desc, then covered count desc
    results.sort(key=lambda r: (r.avg_pct, r.min_pct, len(r.covered)), reverse=True)

    topn = max(1, int(args.top))
    print("\nLeaderboard:")
    for rank, r in enumerate(results[:topn], 1):
        print(f"#{rank:02d}  avg:{r.avg_pct:5.1f}%  min:{r.min_pct:5.1f}%  ids:{len(r.covered)}  {r.cmd}")

    # Write CSV results under data/
    csv_path = get_data_dir() / "apriltest_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("rank,avg_pct,min_pct,covered,cmd\n")
        for rank, r in enumerate(results, 1):
            f.write(f"{rank},{r.avg_pct:.2f},{r.min_pct:.2f},{len(r.covered)},\"{r.cmd}\"\n")
    print(f"\nSaved full results to {csv_path}")
    return 0
