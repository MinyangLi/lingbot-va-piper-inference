"""Load the included LingBot runtime without installing a separate Git checkout."""
from pathlib import Path
import sys


def add_runtime_path():
    root = Path(__file__).parent / '_vendor' / 'lingbot_va'
    # Upstream uses absolute imports such as modules/utils/configs. Fail explicitly
    # if another package has already claimed one of those module namespaces.
    for name in ('modules', 'utils', 'configs', 'distributed', 'wan_va'):
        existing = sys.modules.get(name)
        file = getattr(existing, '__file__', None)
        if existing is not None and (file is None or root not in Path(file).resolve().parents):
            raise RuntimeError(f'Runtime namespace conflict: {name}; use a dedicated policy process')
    for path in (root, root/'wan_va'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root
