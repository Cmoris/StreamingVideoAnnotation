"""Audio model wrappers with lazy imports for optional dependencies."""

from importlib import import_module


_EXPORTS = {
    "WhisperASR": (".asr_model", "WhisperASR"),
    "QwenASR": (".asr_model", "QwenASR"),
    "VAD": (".vad_model", "VAD"),
    "VAP": (".vap_model", "VAP"),
    "Qwen3OmniCaptioner": (".sed_model", "Qwen3OmniCaptioner"),
    "EnvDetect": (".env_model", "EnvDetect"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
