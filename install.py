from importlib import metadata


def get_installed_version(package):
    try:
        return tuple(map(int, metadata.version(package).split('.')))
    except Exception:
        return (0,)


if get_installed_version('python-socketio') <= (5, 7, 2):
    from launch import run_pip
    run_pip('install python-socketio', 'Civitai: python-socketio')

if get_installed_version('OpenCC'):
    from launch import run_pip
    run_pip('install OpenCC', 'Civitai: OpenCC')
