"""Small workflow entry points shared by the CLI, notebook and model lifecycle."""
from pathlib import Path
from .data import load_manifest, prepare
from .validation import validate_config, validate_cache_contract


def load_cache(config, *, prepare_if_missing=False):
    """Open a matching cache, optionally preparing it when no completed cache exists."""
    validate_config(config)
    cache = Path(config["data"]["cache"])
    if prepare_if_missing and not (cache / "manifest.json").is_file():
        manifest = prepare(config)
    else:
        manifest = load_manifest(cache)
    validate_cache_contract(manifest, config)
    return manifest
