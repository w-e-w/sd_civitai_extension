from datetime import datetime
from pathlib import Path
from typing import List
from tqdm import tqdm
import gradio as gr
import threading
import filetype
import requests
import json
import time
import glob
import io
import os
import re

from modules import shared, sd_models, sd_vae, hashes, ui_extra_networks, errors, cache
from modules.paths import models_path

base_url = shared.cmd_opts.civitai_endpoint
user_agent = 'CivitaiLink:Automatic1111'
download_chunk_size = 8192
bar_format = '{l_bar}{bar:25}{r_bar}{bar:-10b}'

image_extensions = ['.jpeg', '.png', '.jpg', '.gif', '.webp', '.avif']
preview_extensions = image_extensions + ['.mp4', '.webm', '.mov']
civil_ai_api_cache = cache.cache('civil_ai_api_sha256')
input_lock = threading.Lock()


def ask_valid_user_input(message, valid_input=('y', 'n')):
    while True:
        with input_lock:
            tqdm.write(message, end='')
            user_input = input().strip().lower()
            if user_input in valid_input:
                return user_input


# endregion
# region Utils
def log(message):
    """Log a message to the console."""
    tqdm.write(f'Civitai: {message}')


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


def get_all_by_hash_with_cache(file_hashes: List[str]):
    """"Un-finished function"""

    # cached_info_hashes = [file_hash for file_hash in file_hashes if file_hash in metadata_cache_dict]
    missing_info_hashes = [file_hash for file_hash in file_hashes if file_hash not in civil_ai_api_cache]
    new_results = []
    try:
        for i in range(0, len(missing_info_hashes), 100):
            batch = missing_info_hashes[i:i + 100]
            new_results.extend(get_all_by_hash(batch))

    except Exception as e:
        errors.report('Failed to fetch info from Civitai', exc_info=True)
        raise e

    new_results = sorted(new_results, key=lambda x: datetime.fromisoformat(x['createdAt'].rstrip('Z')), reverse=True)

    found_info_hashes = set()
    for new_metadata in new_results:
        for file in new_metadata['files']:
            file_hash = file['hashes']['SHA256'].lower()
            found_info_hashes.add(file_hash)
    for file_hash in set(missing_info_hashes) - found_info_hashes:
        if file_hash not in civil_ai_api_cache:
            civil_ai_api_cache[file_hash] = None
    return new_results


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
    lora_dir = get_lora_dir()
    if 'LORA' in types:
        resources = [r for r in resources if r['type'] != 'LORA']
        resources += get_resources_in_folder('LORA', lora_dir, ['pt', 'safetensors', 'ckpt'])
    if 'LoCon' in types:
        lycoris_dir = get_locon_dir()
        if lora_dir != lycoris_dir:
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


modified_url_re = re.compile(r'/width=\d+/')
re_uuid_v4 = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/).*')


IMG_CONTENT_TYPE_MAP = {
    'image/x-icon': '.ico',
}


def get_request_stream(url):
    response = None
    while True:
        for i in range(3):
            response = requests.get(url, stream=True, headers={"User-Agent": user_agent})
            if response.status_code == 200:
                return response
            time.sleep(1)

        if response.status_code != 200:
            if ask_valid_user_input('Press Enter to retry, Enter "s" to skip: ', valid_input=('', 's')) == 's':
                return response
            tqdm.write(f"Retrying with new URL: {url}")


def test_image_type(image_path):
    try:
        return f'.{filetype.guess(image_path).extension}'
    except Exception as e:
        pass


def download_image_auto_file_type(url, dest, total_pbar: tqdm = None):
    dest = Path(dest)

    original_true_url = re_uuid_v4.sub(r'\1original=true', url)
    if total_pbar is not None:
        total_pbar.set_postfix_str(f'{original_true_url} -> {dest.with_suffix("")}')
    response = get_request_stream(original_true_url)
    if response.status_code != 200:
        log(f'Failed to download {original_true_url} {response.status_code}')
        return

    content_type = response.headers.get('Content-Type', '')
    file_extension = IMG_CONTENT_TYPE_MAP.get(content_type, f'.{content_type.rpartition("/")[2]}')
    dest = dest.with_suffix(file_extension)
    total = int(response.headers.get('content-length', 0))
    with io.BytesIO() as file_like_object:
        try:
            current = 0
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, dynamic_ncols=True, bar_format=bar_format, leave=False) as bar:
                for data in response.iter_content(chunk_size=download_chunk_size):
                    current += len(data)
                    file_like_object.write(data)
                    bar.update(len(data))  # Update with the length of the data written

        except Exception as e:
            log(f'Failed to download {original_true_url} {e}')

        try:
            real_img_type = test_image_type(file_like_object)
            if real_img_type is not None:
                dest = dest.with_suffix(real_img_type)

            if dest.exists():
                override_choice = ask_valid_user_input(f"File already exists: {str(dest)} overwrite Y/N?: ")
                if override_choice != 'y':
                    return
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(file_like_object.getvalue())
            if dest.suffix not in preview_extensions:
                message = f'Warning: Not unexpected file type {str(dest)}\nPress Enter to continue'
                gr.Warning(message)
        except Exception as e:
            log(f'Failed to download {original_true_url} {e}')
