import json, os

_PATH = os.environ.get("EVA_SETTINGS_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "settings.json")
try:
    with open(_PATH, "r", encoding="utf-8") as _f:
        _CFG = json.load(_f)
except Exception as _e:
    print(f"[SETTINGS] could not read {_PATH}: {_e!r} - using built-in defaults", flush=True)
    _CFG = {}


def get(path, default=None):
    """get('tts.model', 'bulbul:v3') -> value from settings.json, else default."""
    node = _CFG
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    if node is None and default is not None:
        return default
    return node