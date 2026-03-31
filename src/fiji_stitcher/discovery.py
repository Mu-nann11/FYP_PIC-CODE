from pathlib import Path

DEFAULT_CHANNELS = ("DAPI", "HER2", "PR", "ER")


def _channels_from_config(config):
    try:
        channels = config.get("LOADER", {}).get("CHANNELS", None)
    except Exception:
        channels = None

    if not channels:
        return DEFAULT_CHANNELS

    out = [str(c).strip() for c in channels if str(c).strip()]
    return tuple(out) if out else DEFAULT_CHANNELS


def get_all_level1_directories(config):
    root = Path(config["DEFAULT_ROOT_DIR"])
    raw_name = str(config.get("RAW_DATA_DIR_NAME", "Raw_Data")).strip()
    channels = _channels_from_config(config)

    candidate_roots = [
        root / raw_name,
        root,
    ]

    found = []
    seen = set()

    def _has_channel_dir(p: Path) -> bool:
        return any((p / ch).is_dir() for ch in channels)

    def _has_tif_files(p: Path) -> bool:
        return any(p.glob("*.tif")) or any(p.glob("*.tiff"))

    def _add(p: Path):
        resolved = str(p.resolve())
        if resolved not in seen:
            found.append(resolved)
            seen.add(resolved)

    for base in candidate_roots:
        if not base.exists() or not base.is_dir():
            continue

        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue

            if _has_channel_dir(p) or _has_tif_files(p):
                _add(p)
                continue

            for child in sorted(p.iterdir()):
                if not child.is_dir():
                    continue
                if _has_channel_dir(child) or _has_tif_files(child):
                    _add(child)
                    continue

                for grandchild in sorted(child.iterdir()):
                    if not grandchild.is_dir():
                        continue
                    if _has_channel_dir(grandchild) or _has_tif_files(grandchild):
                        _add(grandchild)

    return found
