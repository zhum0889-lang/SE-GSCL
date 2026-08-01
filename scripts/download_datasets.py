"""Download the public datasets selected for SE-GSCL experiments.

Paderborn is hosted as direct RAR archives. HUSTbearing is hosted as a
public Google Drive folder and is downloaded with gdown. The multi-domain
bearing dataset used by Risca et al. is downloaded through Mendeley Data's
anonymous public ZIP endpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
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
MENDELEY_ZIP_URL = "https://data.mendeley.com/public-api/zip/{dataset_id}/download/{version}"
MULTIDOMAIN_SUBSETS = (
    {
        "subset": 1,
        "dataset_id": "53vtnjy6c6",
        "version": 1,
        "bearing": "6204_deep_groove_ball",
        "doi": "10.17632/53vtnjy6c6.1",
    },
    {
        "subset": 2,
        "dataset_id": "7trwzz77xh",
        "version": 1,
        "bearing": "N204_NJ204_cylindrical_roller",
        "doi": "10.17632/7trwzz77xh.1",
    },
    {
        "subset": 3,
        "dataset_id": "2cygy6y4rk",
        "version": 1,
        "bearing": "30204_tapered_roller",
        "doi": "10.17632/2cygy6y4rk.1",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("paderborn", "hust", "multidomain", "risca", "all"),
        default="all",
        help="Dataset to download. 'risca' is an alias for 'multidomain'.",
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default="pilot",
        help=(
            "Pilot downloads three representative Paderborn archives and Mendeley "
            "subset 1; full downloads all Paderborn archives and all three Mendeley subsets."
        ),
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
        help="Extract downloaded Paderborn RAR and Mendeley ZIP archives.",
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


def find_archive_tool() -> tuple[str, str] | None:
    unrar = shutil.which("unrar")
    if unrar:
        return "unrar", unrar
    unar = shutil.which("unar")
    if unar and shutil.which("lsar"):
        return "unar", unar
    seven_zip = shutil.which("7zz") or shutil.which("7z")
    if seven_zip:
        return "7zip", seven_zip
    return None


def test_archive(
    archive_tool: tuple[str, str],
    archive: Path,
    *,
    show_error: bool = False,
) -> bool:
    backend, executable = archive_tool
    if backend == "unrar":
        command = [executable, "t", "-idq", str(archive)]
    elif backend == "unar":
        command = [shutil.which("lsar") or "lsar", str(archive)]
    else:
        command = [executable, "t", str(archive)]
    result = subprocess.run(
        command,
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
    archive_tool = find_archive_tool()
    if archive_tool:
        print(f"Archive validator: {archive_tool[0]} ({archive_tool[1]})")
    print(f"Paderborn mode={mode}; archives={len(bearing_ids)}")
    for bearing_id in bearing_ids:
        url = f"{PADERBORN_BASE_URL}/{bearing_id}.rar"
        target = destination / f"{bearing_id}.rar"
        if target.exists() and archive_tool and not test_archive(archive_tool, target):
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
            if archive_tool and not test_archive(archive_tool, target, show_error=True):
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


def inspect_zip_archive(path: Path) -> dict[str, int]:
    """Validate the ZIP directory and return counts without rereading all payloads."""
    try:
        with zipfile.ZipFile(path) as archive:
            files = [entry for entry in archive.infolist() if not entry.is_dir()]
            mat_files = [entry for entry in files if entry.filename.lower().endswith(".mat")]
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Invalid ZIP archive: {path}: {exc}") from exc
    if not files:
        raise RuntimeError(f"ZIP archive contains no files: {path}")
    if not mat_files:
        raise RuntimeError(f"ZIP archive contains no MAT records: {path}")
    return {"zip_entries": len(files), "mat_files": len(mat_files)}


def download_multidomain(data_root: Path, mode: str, dry_run: bool) -> Path:
    dataset_root = data_root / "MultiDomainBearing"
    archive_dir = dataset_root / "archives"
    selected = MULTIDOMAIN_SUBSETS[:1] if mode == "pilot" else MULTIDOMAIN_SUBSETS
    manifest_subsets: list[dict[str, object]] = []
    print(f"MultiDomainBearing mode={mode}; subsets={len(selected)}")

    for metadata in selected:
        dataset_id = str(metadata["dataset_id"])
        version = int(metadata["version"])
        filename = f"subset{metadata['subset']}_{metadata['bearing']}.zip"
        target = archive_dir / filename
        url = MENDELEY_ZIP_URL.format(dataset_id=dataset_id, version=version)

        validation: dict[str, int] | None = None
        if target.exists():
            try:
                validation = inspect_zip_archive(target)
            except RuntimeError:
                invalid = target.with_suffix(target.suffix + ".invalid")
                invalid.unlink(missing_ok=True)
                target.replace(invalid)
                print(f"Archive failed validation; preserved as {invalid}")

        if target.exists():
            print(f"Already present: {target}")
        elif dry_run:
            print(f"Would download: {url}")
            print(f"Destination: {target}")
        else:
            download_url(url, target)
            try:
                validation = inspect_zip_archive(target)
            except RuntimeError as exc:
                invalid = target.with_suffix(target.suffix + ".invalid")
                invalid.unlink(missing_ok=True)
                target.replace(invalid)
                raise RuntimeError(
                    f"Downloaded Mendeley archive failed validation and was preserved as {invalid}."
                ) from exc

        manifest_entry: dict[str, object] = {
            **metadata,
            "source": f"https://data.mendeley.com/datasets/{dataset_id}/{version}",
            "download_url": url,
            "archive": str(target.relative_to(dataset_root)),
        }
        if target.exists():
            if validation is None:
                validation = inspect_zip_archive(target)
            manifest_entry.update(validation)
            manifest_entry["bytes"] = target.stat().st_size
        manifest_subsets.append(manifest_entry)

    if not dry_run:
        write_manifest(
            dataset_root / "source_manifest.json",
            {
                "dataset": "Multi-domain vibration dataset with compound machine faults",
                "mode": mode,
                "license": "CC BY 4.0",
                "subsets": manifest_subsets,
            },
        )
    return archive_dir


def extract_zip_archives(archive_dir: Path, dry_run: bool) -> None:
    archives = sorted(archive_dir.glob("*.zip"))
    if not archives:
        if dry_run:
            print(f"Would extract downloaded Mendeley ZIP archives under {archive_dir}")
            return
        raise FileNotFoundError(f"No Mendeley ZIP archives found under {archive_dir}")
    output_root = archive_dir.parent / "extracted"
    for archive_path in archives:
        validation = inspect_zip_archive(archive_path)
        output_dir = output_root / archive_path.stem
        print(
            f"Extracting {archive_path.name}: {validation['mat_files']} MAT files "
            f"into {output_dir}"
        )
        if dry_run:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_root = output_dir.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                resolved_target = (output_dir / entry.filename).resolve()
                if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
                    raise RuntimeError(
                        f"Unsafe path in ZIP archive {archive_path.name}: {entry.filename}"
                    )
            archive.extractall(output_dir)


def extract_paderborn(archive_dir: Path, dry_run: bool) -> None:
    archive_tool = find_archive_tool()
    if not archive_tool:
        raise SystemExit("--extract requires unrar, unar/lsar, 7zz, or 7z on PATH.")
    backend, executable = archive_tool
    print(f"Archive extractor: {backend} ({executable})")
    archives = sorted(archive_dir.glob("*.rar"))
    if not archives:
        raise FileNotFoundError(f"No Paderborn RAR archives found under {archive_dir}")
    if not dry_run:
        invalid_archives = [
            archive
            for archive in archives
            if not test_archive(archive_tool, archive, show_error=True)
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
        if backend == "unrar":
            command = [executable, "x", "-o+", "-idq", str(archive), f"{output_dir}/"]
        elif backend == "unar":
            command = [executable, "-f", "-o", str(output_dir), str(archive)]
        else:
            command = [executable, "x", "-y", f"-o{output_dir}", str(archive)]
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
    multidomain_dir = None
    if args.dataset in {"paderborn", "all"}:
        paderborn_dir = download_paderborn(data_root, args.mode, args.dry_run)
    if args.dataset in {"hust", "all"}:
        download_hust(data_root, args.dry_run)
    if args.dataset in {"multidomain", "risca", "all"}:
        multidomain_dir = download_multidomain(data_root, args.mode, args.dry_run)
    if args.extract and paderborn_dir is not None:
        extract_paderborn(paderborn_dir, args.dry_run)
    if args.extract and multidomain_dir is not None:
        extract_zip_archives(multidomain_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
