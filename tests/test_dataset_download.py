import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import py7zr

from ISM_diffusion.dataset_download import (
    resolve_training_data, resolve_training_split, validate_archive_members,
)


class DatasetDownloadChecks(unittest.TestCase):
    def test_local_directory_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("ISM_diffusion.dataset_download.download_model_file") as download:
                self.assertEqual(resolve_training_data(directory), Path(directory).resolve())
                with self.assertRaises(FileNotFoundError):
                    resolve_training_data(Path(directory) / "missing")
                download.assert_not_called()

    def test_archive_extraction_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.npz"
            source.write_bytes(b"small archive fixture")
            archive = root / "RGTSYN.7z"
            with py7zr.SevenZipFile(archive, "w") as writer:
                writer.write(source, "RGTSYN/00000.npz")
                writer.write(source, "RGTSYN/00001.npz")
            with patch("ISM_diffusion.dataset_download.SAMPLE_COUNT", 2), \
                    patch("ISM_diffusion.dataset_download.download_model_file", return_value=archive) as download:
                result = resolve_training_data(storage_dir=root / "extracted", cache_dir=root / "cache", revision="commit")
                download.assert_called_once_with("RGTSYN.7z", cache_dir=root / "cache", revision="commit")
                self.assertEqual((result / "RGTSYN/00000.npz").read_bytes(), source.read_bytes())
                self.assertTrue((result / "ready.json").is_file())
                with patch.object(py7zr.SevenZipFile, "extractall", side_effect=AssertionError("should reuse cache")):
                    self.assertEqual(resolve_training_data(storage_dir=root / "extracted"), result)

    def test_unsafe_archive_members_are_rejected(self):
        for name, symlink in [("../escape.npz", False), ("/escape.npz", False),
                              ("C:/escape.npz", False), ("..\\escape.npz", False), ("link.npz", True)]:
            entry = SimpleNamespace(filename=name, is_file=True, is_directory=False, is_symlink=symlink)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_archive_members([entry])

    def test_failed_extraction_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "broken.7z"
            archive.write_bytes(b"incomplete download")
            with patch("ISM_diffusion.dataset_download.download_model_file", return_value=archive):
                with self.assertRaises(py7zr.Bad7zFile):
                    resolve_training_data(storage_dir=root / "extracted")
            self.assertEqual(list((root / "extracted").rglob("ready.json")), [])
            self.assertEqual(list((root / "extracted").glob(".rgtsyn-*")), [])

    def test_split_is_disjoint_persistent_and_restored_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            for index in range(4000):
                (root / f"{index:05d}.npz").touch()
            storage = Path(directory) / "cache"
            path, split = resolve_training_split(root, storage_dir=storage, seed=2026)
            self.assertEqual([len(split[k]) for k in ("train", "val", "test")], [3600, 200, 200])
            self.assertEqual(len(set(split["train"] + split["val"] + split["test"])), 4000)
            self.assertEqual(resolve_training_split(root, storage_dir=storage, seed=2026), (path, split))
            checkpoint = Path(directory) / "run/checkpoints/last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            with self.assertRaises(FileNotFoundError):
                resolve_training_split(root, storage_dir=storage, resume=checkpoint)
            saved_split = checkpoint.parent.parent / "split.json"
            saved_split.write_text(json.dumps(split), encoding="utf-8")
            resumed_path, resumed_split = resolve_training_split(root, storage_dir=storage, seed=999, resume=checkpoint)
            self.assertEqual(resumed_path, saved_split)
            self.assertEqual(resumed_split, split)
            (root / split["train"][0]).unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_training_split(root, path=saved_split)

    def test_training_cli_needs_no_data_or_split_argument(self):
        from train import parse_args
        with patch.object(sys, "argv", ["train.py"]):
            args = parse_args()
        self.assertIsNone(args.data)
        self.assertIsNone(args.split)
        self.assertEqual(args.data_dir, Path("data/modelscope"))


if __name__ == "__main__":
    unittest.main()
