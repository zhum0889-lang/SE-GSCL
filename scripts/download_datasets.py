"""Download the public datasets selected for SE-GSCL experiments.

Paderborn is hosted as direct RAR archives. HUSTbearing is hosted as a
public Google Drive folder and is downloaded with gdown.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PADERBORN_BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter"
PADERBORN_PILOT_IDS = ("K006", "KA01", "KI01")
PADERBORN_ALL_IDS = (
    "K001",
    "K002",
    "K003",
    "K004",
    "K005",
    "K006",
    "KA01",
    "KA03",
    "KA04",
    "KA05",
    "KA06",
    "KA07",
    "KA08",
    "KA09",
    "KA15",
    "KA16",
    "KA22",
    "KA30",
    "KB23",
    "KB24",
    "KB27",
    "KI01",
    "KI03",
    "KI04",
    "KI05",
    "KI07",
    "KI08",
    "KI14",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
)
HUST_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1UMOvyfstYJRyR0rPw0OfH-tIjJg2_0aN?usp=sharing"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("paderborn", "hust", "all"),
        default="all",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default="pilot",
        help="Paderborn pilot downloads one healthy, one outer-race, and one inner-race archive.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Destination root.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract Paderborn RAR files when 7z is installed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without transferring files.",
    )
    return parser.parse_args()


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "SE-GSCL-dataset-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as exc:
        if exc.code == 416 and partial.exists():
            print(f"Range resume rejected for {partial.name}; restarting download.")
            partial.unlink()
            return download_url(url, destination)
        raise

    status = getattr(response, "status", 200)
    if offset and status != 206:
        offset = 0
        partial.unlink(missing_ok=True)
    content_length = int(response.headers.get("Content-Length", "0"))
    total = offset + content_length if content_length else 0
    mode = "ab" if offset else "wb"
    downloaded = offset
    last_report = 0.0

    print(f"Downloading {destination.name}")
    with partial.open(mode) as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_report >= 2:
                if total:
                    print(
                        f"  {downloaded / 1024**2:.1f} / {total / 1024**2:.1f} MiB "
                        f"({downloaded / total:.0%})"
                    )
                else:
                    print(f"  {downloaded / 1024**2:.1f} MiB")
                last_report = now
    if total and downloaded != total:
        raise IOError(
            f"Incomplete download for {destination.name}: "
            f"received {downloaded} of {total} bytes. Partial data kept at {partial}."
        )
    partial.replace(destination)
    print(f"Saved: {destination}")


def test_archive(extractor: str, archive: Path, *, show_error: bool = False) -> bool:
    result = subprocess.run(
        [extractor, "t", str(archive)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0 and show_error:
        output_tail = "\n".join(result.stdout.splitlines()[-20:])
        print(f"Archive integrity test failed: {archive}\n{output_tail}")
    return result.returncode == 0


def download_paderborn(data_root: Path, mode: str, dry_run: bool) -> Path:
    destination = data_root / "Paderborn" / "archives"
    bearing_ids = PADERBORN_PILOT_IDS if mode == "pilot" else PADERBORN_ALL_IDS
    extractor = shutil.which("7zz") or shutil.which("7z")
    if extractor:
        print(f"Archive validator: {extractor}")
    print(f"Paderborn mode={mode}; archives={len(bearing_ids)}")
    for bearing_id in bearing_ids:
        url = f"{PADERBORN_BASE_URL}/{bearing_id}.rar"
        target = destination / f"{bearing_id}.rar"
        if target.exists() and extractor and not test_archive(extractor, target):
            invalid = target.with_suffix(target.suffix + ".invalid")
            invalid.unlink(missing_ok=True)
            target.replace(invalid)
            print(f"Archive failed validation; preserved as {invalid}")
        if target.exists():
            print(f"Already present: {target}")
        elif dry_run:
            print(f"Would download: {url}")
        else:
            download_url(url, target)
            if extractor and not test_archive(extractor, target, show_error=True):
                invalid = target.with_suffix(target.suffix + ".invalid")
                invalid.unlink(missing_ok=True)
                target.replace(invalid)
                raise RuntimeError(
                    f"Downloaded archive failed integrity validation and was preserved "
                    f"as {invalid}. Check its size and file signature before retrying."
                )

    manifest = {
        "dataset": "Paderborn University Bearing Data Center",
        "source": f"{PADERBORN_BASE_URL}/",
        "mode": mode,
        "bearing_ids": list(bearing_ids),
        "pilot_label_map": {
            "K006": "healthy",
            "KA01": "outer_race_fault",
            "KI01": "inner_race_fault",
        },
    }
    if not dry_run:
        write_manifest(data_root / "Paderborn" / "source_manifest.json", manifest)
    return destination


def download_hust(data_root: Path, dry_run: bool) -> Path:
    destination = data_root / "HUSTbearing"
    if dry_run:
        print(f"Would download Google Drive folder: {HUST_FOLDER_URL}")
        print(f"Destination: {destination}")
        return destination

    try:
        import gdown
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "HUSTbearing requires gdown. Install implementation/requirements.txt "
            "and rerun this command."
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HUSTbearing into {destination}")
    files = gdown.download_folder(
        url=HUST_FOLDER_URL,
        output=str(destination),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    if not files:
        raise RuntimeError("Google Drive returned no HUSTbearing files.")
    write_manifest(
        destination / "source_manifest.json",
        {
            "dataset": "HUSTbearing dataset (Zhao, Zio, and Shen)",
            "source": HUST_FOLDER_URL,
            "downloaded_files": len(files),
        },
    )
    return destination


def extract_paderborn(archive_dir: Path, dry_run: bool) -> None:
    extractor = shutil.which("7zz") or shutil.which("7z")
    if not extractor:
        raise SystemExit("--extract requires 7-Zip (7z or 7zz) on PATH.")
    print(f"Archive extractor: {extractor}")
    archives = sorted(archive_dir.glob("*.rar"))
    if not archives:
        raise FileNotFoundError(f"No Paderborn RAR archives found under {archive_dir}")
    if not dry_run:
        invalid_archives = [
            archive
            for archive in archives
            if not test_archive(extractor, archive, show_error=True)
        ]
        if invalid_archives:
            names = ", ".join(archive.name for archive in invalid_archives)
            raise RuntimeError(
                f"Refusing extraction because archive integrity checks failed: {names}. "
                "Rerun the downloader so corrupt archives are replaced."
            )
    output_dir = archive_dir.parent / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        command = [extractor, "x", "-y", f"-o{output_dir}", str(archive)]
        if dry_run:
            print("Would run:", " ".join(command))
        else:
            subprocess.run(command, check=True)


def write_manifest(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    print(f"Data root: {data_root}")
    paderborn_dir = None
    if args.dataset in {"paderborn", "all"}:
        paderborn_dir = download_paderborn(data_root, args.mode, args.dry_run)
    if args.dataset in {"hust", "all"}:
        download_hust(data_root, args.dry_run)
    if args.extract and paderborn_dir is not None:
        extract_paderborn(paderborn_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
