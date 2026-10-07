"""Resolve local weights and download published ModelScope assets."""
from pathlib import Path


MODEL_ID = "douyimin/ISMdiffusion"
MODEL_FILENAME = "ISM_PretrainedModel.ckpt"


def add_download_arguments(parser):
    parser.add_argument("--cache-dir", type=Path,
                        help="ModelScope cache directory (default: ModelScope's configured cache)")
    parser.add_argument("--model-revision", default="master",
                        help="ModelScope branch or commit for automatic download (default: master)")


def resolve_checkpoint(path=None, *, cache_dir=None, revision="master"):
    """Explicit local paths never trigger a download; the SDK reuses cached files."""
    if path is not None:
        local_path = Path(path).expanduser().resolve()
        if not local_path.is_file() or local_path.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint is missing or empty: {local_path}")
        return local_path

    return download_model_file(MODEL_FILENAME, cache_dir=cache_dir, revision=revision)


def download_model_file(filename, *, cache_dir=None, revision="master"):
    """Download one published file using the SDK cache and a process lock."""

    try:
        from filelock import FileLock
        from modelscope.hub.file_download import model_file_download
    except ImportError as error:
        raise RuntimeError(
            "Automatic download requires ModelScope. Run 'python -m pip install -r requirements.txt', "
            "or supply local weights/data with --checkpoint, --init and --data."
        ) from error

    cache = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    lock_dir = cache if cache is not None else Path.home() / ".cache" / "ism-diffusion"
    lock_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resolving {MODEL_ID}/{filename} from ModelScope (revision={revision})")
    # Lightning/torchrun workers can reach this before distributed initialization.
    # A shared lock prevents concurrent transfers into the same SDK cache.
    with FileLock(str(lock_dir / "ism-modelscope-download.lock")):
        downloaded = model_file_download(
            model_id=MODEL_ID, file_path=filename, revision=revision,
            cache_dir=str(cache) if cache is not None else None,
        )
    local_path = Path(downloaded).resolve()
    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise FileNotFoundError(f"Downloaded file is missing or empty: {local_path}")
    return local_path
