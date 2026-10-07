"""Cached ModelScope training data extraction and persistent sample splits."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import tempfile

from filelock import FileLock

from ISM_diffusion.weights import download_model_file


DATASET_FILENAME = "RGTSYN.7z"
SAMPLE_COUNT = 4000


def sample_files(root):
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.npz"))
    if len(files) != SAMPLE_COUNT:
        raise ValueError(f"Expected {SAMPLE_COUNT} NPZ samples in {root}, found {len(files)}")
    return files


def validate_archive_members(entries):
    """Reject links and paths that could escape the private extraction directory."""
    names = set()
    for entry in entries:
        path = PurePosixPath(entry.filename.replace("\\", "/"))
        if (path.is_absolute() or ".." in path.parts or ":" in str(path)
                or entry.is_symlink or not (entry.is_file or entry.is_directory)):
            raise ValueError(f"Unsupported archive member: {entry.filename}")
        normalized = str(path).casefold()
        if normalized in names:
            raise ValueError(f"Duplicate archive member: {entry.filename}")
        names.add(normalized)
    count = sum(e.is_file and e.filename.endswith(".npz") for e in entries)
    if count != SAMPLE_COUNT:
        raise ValueError(f"Expected {SAMPLE_COUNT} NPZ samples in the archive, found {count}")


def resolve_training_data(path=None, *, storage_dir=Path("data/modelscope"), cache_dir=None, revision="master"):
    if path is not None:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Training data directory does not exist: {root}")
        return root

    archive_path = download_model_file(DATASET_FILENAME, cache_dir=cache_dir, revision=revision)
    stat = archive_path.stat()
    signature = dict(path=str(archive_path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    storage = Path(storage_dir).expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    destination = storage / f"RGTSYN-{key}"
    with FileLock(str(storage / f"RGTSYN-{key}.lock")):
        if (destination / "ready.json").is_file():
            sample_files(destination)
            return destination
        if destination.exists():
            raise RuntimeError(f"Incomplete extraction at {destination}; select a new --data-dir")
        try:
            import py7zr
        except ImportError as error:
            raise RuntimeError("Automatic extraction requires py7zr; install requirements.txt or use --data") from error
        # Publish only a fully extracted dataset. Interrupted/failed attempts cannot
        # be mistaken for a complete cache by another training process.
        with tempfile.TemporaryDirectory(prefix=".rgtsyn-", dir=storage) as temporary:
            staging = Path(temporary)
            print(f"Extracting {DATASET_FILENAME} to {destination} (about 8.4 GB)")
            with py7zr.SevenZipFile(archive_path, "r") as archive:
                validate_archive_members(archive.list())
                archive.extractall(path=staging)
            sample_files(staging)
            (staging / "ready.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
            staging.rename(destination)
    return destination


def resolve_training_split(root, path=None, *, storage_dir=Path("data/modelscope"), seed=2026, resume=None):
    from ISM_diffusion.data import read_split

    if path is None and resume is not None:
        # Checkpoints from train.py live in RUN/checkpoints; the saved split is
        # RUN/split.json. Never silently replace that partition during resume.
        path = Path(resume).resolve().parent.parent / "split.json"
        if not path.is_file():
            raise FileNotFoundError("Resume split.json was not found beside the run; supply --split explicitly")
    if path is not None:
        path = Path(path).expanduser().resolve()
        split = read_split(path)
    else:
        files = sample_files(root)
        identity = json.dumps(dict(root=str(root), files=files, seed=seed), sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()[:16]
        folder = Path(storage_dir).expanduser().resolve() / "splits"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"split-{key}.json"
        with FileLock(str(path.with_suffix(".lock"))):
            if not path.exists():
                random.Random(seed).shuffle(files)
                split = dict(seed=seed, train=files[:3600], val=files[3600:3800], test=files[3800:])
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(split, indent=2), encoding="utf-8")
                temporary.replace(path)
            split = read_split(path)
    if [len(split[k]) for k in ("train", "val", "test")] != [3600, 200, 200]:
        raise ValueError("The paper recipe requires a 3600/200/200 split")
    for group in ("train", "val", "test"):
        for name in split[group]:
            sample = (root / name).resolve()
            if not sample.is_relative_to(root) or not sample.is_file():
                raise FileNotFoundError(f"Split sample missing or outside data directory: {name}")
    return path, split
