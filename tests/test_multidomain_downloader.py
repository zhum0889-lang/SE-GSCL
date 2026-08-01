from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "download_datasets",
    ROOT / "scripts" / "download_datasets.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load scripts/download_datasets.py")
DOWNLOADS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADS)


class MultiDomainDownloaderTests(unittest.TestCase):
    def test_inspect_zip_counts_mat_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "subset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("8kHz/600/H_IR_8_6204_600.mat", b"mat-data")
                archive.writestr("README.txt", b"metadata")

            summary = DOWNLOADS.inspect_zip_archive(archive_path)
            self.assertEqual(summary, {"zip_entries": 2, "mat_files": 1})

    def test_inspect_zip_rejects_archive_without_mat_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "empty.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("README.txt", b"metadata")

            with self.assertRaisesRegex(RuntimeError, "no MAT records"):
                DOWNLOADS.inspect_zip_archive(archive_path)

    def test_extract_zip_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_dir = Path(temp_dir) / "archives"
            archive_dir.mkdir()
            archive_path = archive_dir / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("valid.mat", b"mat-data")
                archive.writestr("../escaped.mat", b"unsafe")

            with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
                DOWNLOADS.extract_zip_archives(archive_dir, dry_run=False)
            self.assertFalse((Path(temp_dir) / "escaped.mat").exists())


if __name__ == "__main__":
    unittest.main()
