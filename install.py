from importlib import metadata
from packaging.version import parse


def get_installed_version(package):
    try:
        return parse(metadata.version(package))
    except Exception:
        return None


if not get_installed_version('OpenCC'):
    from launch import run_pip
    run_pip('install OpenCC', 'Civitai: OpenCC')
