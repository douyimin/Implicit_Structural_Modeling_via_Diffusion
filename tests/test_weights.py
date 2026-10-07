import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from ISM_diffusion.weights import MODEL_FILENAME, MODEL_ID, resolve_checkpoint


class WeightDownloadChecks(unittest.TestCase):
    def test_explicit_local_path_does_not_import_modelscope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / MODEL_FILENAME
            path.write_bytes(b"local checkpoint")
            with patch.dict(sys.modules, {"modelscope": None}):
                self.assertEqual(resolve_checkpoint(path), path.resolve())
                with self.assertRaises(FileNotFoundError):
                    resolve_checkpoint(Path(directory) / "missing.ckpt")
                path.write_bytes(b"")
                with self.assertRaises(FileNotFoundError):
                    resolve_checkpoint(path)

    def test_download_uses_expected_repository_filename_revision_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / MODEL_FILENAME
            checkpoint.write_bytes(b"cached checkpoint")
            api = ModuleType("modelscope.hub.file_download")
            api.model_file_download = Mock(return_value=str(checkpoint))
            modules = {"modelscope": ModuleType("modelscope"),
                       "modelscope.hub": ModuleType("modelscope.hub"),
                       "modelscope.hub.file_download": api}
            with patch.dict(sys.modules, modules):
                result = resolve_checkpoint(cache_dir=directory, revision="test-commit")
            self.assertEqual(result, checkpoint.resolve())
            api.model_file_download.assert_called_once_with(
                model_id=MODEL_ID, file_path=MODEL_FILENAME, revision="test-commit",
                cache_dir=str(Path(directory).resolve()))

    def test_download_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            api = ModuleType("modelscope.hub.file_download")
            api.model_file_download = Mock(side_effect=ConnectionError("download unavailable"))
            with patch.dict(sys.modules, {"modelscope": ModuleType("modelscope"),
                                         "modelscope.hub": ModuleType("modelscope.hub"),
                                         "modelscope.hub.file_download": api}):
                with self.assertRaises(ConnectionError):
                    resolve_checkpoint(cache_dir=directory)

    def test_inference_and_training_accept_automatic_download(self):
        inference = importlib.import_module("inference")
        train = importlib.import_module("train")
        with patch.object(sys, "argv", ["inference.py"]):
            args = inference.parse_args()
            self.assertIsNone(args.checkpoint)
            self.assertEqual(args.model_revision, "master")
        with patch.object(sys, "argv", ["train.py", "--data", "data", "--split", "split.json"]):
            args = train.parse_args()
            self.assertIsNone(args.init)
            self.assertIsNone(args.resume)
        with patch.object(sys, "argv", ["train.py", "--data", "data", "--split", "split.json",
                                        "--resume", "last.ckpt"]):
            args = train.parse_args()
            self.assertEqual(args.resume, Path("last.ckpt"))
            self.assertIsNone(args.init)


if __name__ == "__main__":
    unittest.main()
