import importlib.util
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_dust3r_roots() -> list[Path]:
    project_root = _project_root()
    candidates = [
        project_root / "submodules" / "dust3r",
        Path("/home/isaac/code/InstantSplat/submodules/dust3r"),
    ]
    env_root = None
    try:
        import os

        env_root = os.environ.get("INSTANTSPLAT_ROOT")
    except Exception:
        env_root = None
    if env_root:
        candidates.insert(0, Path(env_root) / "submodules" / "dust3r")
    return candidates


def ensure_runtime_paths() -> Path:
    if importlib.util.find_spec("dust3r") is not None and importlib.util.find_spec(
        "models.croco"
    ) is not None:
        return _project_root()

    for dust3r_root in _candidate_dust3r_roots():
        croco_root = dust3r_root / "croco"
        if not dust3r_root.exists() or not croco_root.exists():
            continue
        for path in (dust3r_root, croco_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        return _project_root()

    raise ModuleNotFoundError(
        "Could not locate the dust3r/croco runtime required by feature_3dgs.ttt3r. "
        "Expected either a local 'submodules/dust3r' checkout or an "
        "InstantSplat checkout at '/home/isaac/code/InstantSplat'."
    )
