"""Piper history-aware inference; no hardware control."""

def __getattr__(name):
    if name == 'PiperPolicy':
        from .policy import PiperPolicy
        return PiperPolicy
    raise AttributeError(name)

__all__ = ['PiperPolicy']
