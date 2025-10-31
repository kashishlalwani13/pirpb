# ==========================================================
# config.py  |  PIR/PB Project Configuration Loader
# ==========================================================
# Loads all experiment settings from configs/config.yaml
# Keeps the pipeline reproducible and parameter-driven.
# ----------------------------------------------------------

from pathlib import Path
from pydantic import BaseModel
import yaml


class Settings(BaseModel):
    data_dir: Path
    serp_dir: Path
    results_dir: Path
    logs_dir: Path
    engine: str
    top_k: int
    cosine_min: float
    embed_model: str
    rbo_p: float


def load_settings(cfg_path: str = "configs/config.yaml") -> "Settings":
    """Read YAML configuration and return a typed Settings object."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return Settings(
        data_dir=Path(cfg["io"]["data_dir"]),
        serp_dir=Path(cfg["io"]["serp_dir"]),
        results_dir=Path(cfg["io"]["results_dir"]),
        logs_dir=Path(cfg["io"]["logs_dir"]),
        engine=cfg["search"]["engine"],
        top_k=cfg["search"]["top_k"],
        cosine_min=cfg["paraphrase"]["cosine_min"],
        embed_model=cfg["embedding"]["model"],
        rbo_p=cfg["metrics"]["rbo_p"],
    )


if __name__ == "__main__":
    s = load_settings()
    print("✅ Config loaded:", s)
