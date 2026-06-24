import logging


def setup_logging(level: str = "INFO") -> None:
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
        force=True,
    )
    logging.getLogger("__main__").setLevel(numeric_level)
    logging.getLogger("src").setLevel(numeric_level)
