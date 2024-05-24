import json
import os
import shutil
import tempfile
import time
from typing import List
import requests
import glob

from tqdm import tqdm
from modules import shared, sd_models, sd_vae, hashes, ui_extra_networks
from modules.paths import models_path

base_url = shared.cmd_opts.civitai_endpoint
user_agent = 'CivitaiLink:Automatic1111'
download_chunk_size = 8192


# endregion

# region Utils
def log(message):
    """Log a message to the console."""
    print(f'Civitai: {message}')


def download_file(url, dest, on_progress=None):
    if os.path.exists(dest):
        log(f'File already exists: {dest}')

    log(f'Downloading: "{url}" to {dest}\n')

    response = requests.get(url, stream=True, headers={"User-Agent": user_agent})
    total = int(response.headers.get('content-length', 0))
    start_time = time.time()

    dest = os.path.expanduser(dest)
    dst_dir = os.path.dirname(dest)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        current = 0
        with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=download_chunk_size):
                current += len(data)
                pos = f.write(data)
                bar.update(pos)
                if on_progress is not None:
                    should_stop = on_progress(current, total, start_time)
                    if should_stop:
                        raise Exception('Download cancelled')
        f.close()
        shutil.move(f.name, dest)
    except OSError as e:
        print(f"Could not write the preview file to {dst_dir}")
        print(e)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


# region API
def req(endpoint, method='GET', data=None, params=None, headers=None):
    """Make a request to the Civitai API."""
    if headers is None:
        headers = {}
    headers['User-Agent'] = user_agent
    api_key = shared.opts.data.get("civitai_api_key", None)
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'
    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data)
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
    if params is None:
        params = {}
    response = requests.request(method, base_url + endpoint, data=data, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f'Error: {response.status_code} {response.text}')
    return response.json()


def get_models(query, creator, tag, file_type, page=1, page_size=20, sort='Most Downloaded', period='AllTime'):
    """Get a list of models from the Civitai API."""
    response = req('/models', params={
        'query': query,
        'username': creator,
        'tag': tag,
        'type': file_type,
        'sort': sort,
        'period': period,
        'page': page,
        'pageSize': page_size,
    })
    return response


def get_all_by_hash(file_hashes: List[str]):
    response = req(f"/model-versions/by-hash", method='POST', data=file_hashes)
    return response


def get_model_version(_id):
    """Get a model version from the Civitai API."""
    response = req('/model-versions/' + _id)
    return response


def get_model_version_by_hash(file_hash: str):
    response = req(f"/model-versions/by-hash/{file_hash}")
    return response


def get_creators(query, page=1, page_size=20):
    """Get a list of creators from the Civitai API."""
    response = req('/creators', params={
        'query': query,
        'page': page,
        'pageSize': page_size
    })
    return response


def get_tags(query, page=1, page_size=20):
    """Get a list of tags from the Civitai API."""
    response = req('/tags', params={
        'query': query,
        'page': page,
        'pageSize': page_size
    })
    return response


def get_lora_dir():
    return shared.cmd_opts.lora_dir


def get_locon_dir():
    try:
        return shared.cmd_opts.lyco_dir or get_lora_dir()
    except AttributeError:
        return get_lora_dir()


def get_model_dir():
    return shared.cmd_opts.ckpt_dir or sd_models.model_path


def get_automatic_type(file_type: str):
    if file_type == 'Hypernetwork':
        return 'hypernet'
    return file_type.lower()


def get_automatic_name(file_type: str, filename: str, folder: str):
    abspath = os.path.abspath(filename)
    if abspath.startswith(folder):
        fullname = abspath.replace(folder, '')
    else:
        fullname = os.path.basename(filename)

    if fullname.startswith("\\") or fullname.startswith("/"):
        fullname = fullname[1:]

    if file_type == 'Checkpoint':
        return fullname
    return os.path.splitext(fullname)[0]


def has_preview(filename: str):
    ui_extra_networks.allowed_preview_extensions()
    preview_exts = ui_extra_networks.allowed_preview_extensions()
    preview_exts = [*preview_exts, *["preview." + x for x in preview_exts]]
    for ext in preview_exts:
        if os.path.exists(os.path.splitext(filename)[0] + '.' + ext):
            return True
    return False


def has_info(filename: str):
    return os.path.isfile(os.path.splitext(filename)[0] + '.json')


def get_resources_in_folder(file_type, folder, exts=None, exts_exclude=None):
    if exts_exclude is None:
        exts_exclude = []
    if exts is None:
        exts = []
    _resources = []
    os.makedirs(folder, exist_ok=True)

    candidates = []
    for ext in exts:
        candidates += glob.glob(os.path.join(folder, '**/*.' + ext), recursive=True)
    for ext in exts_exclude:
        candidates = [x for x in candidates if not x.endswith(ext)]

    folder = os.path.abspath(folder)
    automatic_type = get_automatic_type(file_type)

    cmd_opts_no_hashing = shared.cmd_opts.no_hashing
    shared.cmd_opts.no_hashing = False
    try:
        for filename in sorted(candidates):
            if os.path.isdir(filename):
                continue

            name = os.path.splitext(os.path.basename(filename))[0]
            automatic_name = get_automatic_name(file_type, filename, folder)
            file_hash = hashes.sha256(filename, f"{automatic_type}/{automatic_name}")

            _resources.append({'type': file_type, 'name': name, 'hash': file_hash, 'path': filename, 'hasPreview': has_preview(filename), 'hasInfo': has_info(filename)})
    finally:
        shared.cmd_opts.no_hashing = cmd_opts_no_hashing
    return _resources


resources = []


def load_resource_list(types=None):
    global resources
    if types is None:
        types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint', 'VAE', 'Controlnet', 'Upscaler']

    if 'LORA' in types:
        resources = [r for r in resources if r['type'] != 'LORA']
        resources += get_resources_in_folder('LORA', get_lora_dir(), ['pt', 'safetensors', 'ckpt'])
    if 'LoCon' in types:
        resources = [r for r in resources if r['type'] != 'LoCon']
        resources += get_resources_in_folder('LoCon', get_locon_dir(), ['pt', 'safetensors', 'ckpt'])
    if 'Hypernetwork' in types:
        resources = [r for r in resources if r['type'] != 'Hypernetwork']
        resources += get_resources_in_folder('Hypernetwork', shared.cmd_opts.hypernetwork_dir, ['pt', 'safetensors', 'ckpt'])
    if 'TextualInversion' in types:
        resources = [r for r in resources if r['type'] != 'TextualInversion']
        resources += get_resources_in_folder('TextualInversion', shared.cmd_opts.embeddings_dir, ['pt', 'bin', 'safetensors'])
    if 'Checkpoint' in types:
        resources = [r for r in resources if r['type'] != 'Checkpoint']
        resources += get_resources_in_folder('Checkpoint', get_model_dir(), ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt'])
    if 'Controlnet' in types:
        resources = [r for r in resources if r['type'] != 'Controlnet']
        resources += get_resources_in_folder('Controlnet', os.path.join(models_path, "ControlNet"), ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt'])
    if 'Upscaler' in types:
        resources = [r for r in resources if r['type'] != 'Upscaler']
        resources += get_resources_in_folder('Upscaler', os.path.join(models_path, "ESRGAN"), ['safetensors', 'ckpt', 'pt'])
    if 'VAE' in types:
        resources = [r for r in resources if r['type'] != 'VAE']
        resources += get_resources_in_folder('VAE', get_model_dir(), ['vae.pt', 'vae.safetensors', 'vae.ckpt'])
        resources += get_resources_in_folder('VAE', sd_vae.vae_path, ['pt', 'safetensors', 'ckpt'])

    return resources


def get_model_by_hash(file_hash: str):
    if found := [info for info in sd_models.checkpoints_list.values() if file_hash == info.sha256 or file_hash == info.shorthash or file_hash == info.hash]:
        return found[0]


def update_resource_preview(file_hash: str, preview_url: str):
    file_hash = file_hash.lower()
    for resource in [resource for resource in load_resource_list([]) if file_hash == resource['hash']]:
        # download image and save to resource['path'] - ext + '.preview.png'
        download_file(preview_url, f'{os.path.splitext(resource["path"])[0]}.preview{os.path.splitext(preview_url)[1]}')
