from tacyolo.model import TacticalYOLO

__all__ = ["TacticalYOLO", "main"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "main":
        from tacyolo.cli import main

        return main
    raise AttributeError(name)
