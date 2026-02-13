from borsa.config import AppConfig


def test_config_paths_are_initialized() -> None:
    cfg = AppConfig()
    assert cfg.cache_dir.name == "cache"
