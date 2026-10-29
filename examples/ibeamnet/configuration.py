"""Load TOML paths; shared workflow rules live in validation.py."""
from pathlib import Path
from .validation import validate_config


def load_config(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    path = Path(path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    validate_config(config)
    config["source_configuration"] = str(path)
    for section, key in (("data", "source"), ("data", "cache"),
                         ("training", "output"), ("export", "output")):
        config[section][key] = (path.parent / config[section][key]).resolve()
    return config
