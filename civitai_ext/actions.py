from concurrent.futures import ThreadPoolExecutor
from . import lib as civitai, opencc_utils
from datetime import datetime
from pathlib import Path
import json
from modules import shared, errors
from tqdm import tqdm
import gradio as gr
import os

previewable_types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']


def load_previews():
    nsfw_previews = shared.opts.civitai_nsfw_previews

    civitai.log(f'Check resources for missing preview images')
    resources = civitai.load_resource_list()
    resources = [r for r in resources if r['type'] in previewable_types]

    # get all resources that are missing previews
    missing_previews = [r for r in resources if r['hasPreview'] is False]
    civitai.log(f'Found {len(missing_previews)} resources missing preview images')
    hashes = [r['hash'] for r in missing_previews]

    # split hashes into batches of 100 and fetch into results
    results = []
    try:
        for i in range(0, len(hashes), 100):
            batch = hashes[i:i + 100]
            results.extend(civitai.get_all_by_hash(batch))
    except Exception:
        errors.report('Failed to fetch preview images from Civitai', exc_info=True)
        return

    if not results:
        civitai.log('No preview images found on Civitai')
        return

    results = sorted(results, key=lambda x: datetime.fromisoformat(x['createdAt'].rstrip('Z')), reverse=True)

    civitai.log(f'Found {len(results)} hash matches')

    # update the resources with the new preview
    updated = 0
    for r in tqdm(results):
        if r is None:
            continue

        for file in r['files']:
            if 'hashes' not in file or 'SHA256' not in file['hashes']:
                continue
            file_hash = file['hashes']['SHA256']
            if file_hash.lower() not in hashes:
                continue
            images = r['images']
            if not nsfw_previews:
                images = [i for i in images if i['nsfw'] is False]
            if not images:
                continue
            image_url = images[0]['url']
            civitai.update_resource_preview(file_hash, image_url)
            updated += 1

    civitai.log(f'Updated {updated} preview images')


def run_load_previews():
    with ThreadPoolExecutor() as executor:
        executor.submit(load_previews)
    gr.Info('Finished fetching preview images from Civitai')


actionable_types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']
base_model_version = {
    'SD 1': 'SD1',
    'SD 2': 'SD2',
    'SDXL': 'SDXL',
    # 'Pony': 'SDXL',
    'SD 3': 'SD3',
}


def load_info():
    civitai.log('Check resources for missing info files')
    resources = civitai.load_resource_list()
    resources = [r for r in resources if r['type'] in actionable_types]

    # get all resources that have no info files
    missing_info = [r for r in resources if r['hasInfo'] is False]
    civitai.log(f'Found {len(missing_info)} resources missing info files')
    hashes = [r['hash'] for r in missing_info]

    # split hashes into batches of 100 and fetch into results
    results = []
    try:
        for i in range(0, len(hashes), 100):
            batch = hashes[i:i + 100]
            results.extend(civitai.get_all_by_hash(batch))

    except Exception:
        errors.report('Failed to fetch info from Civitai', exc_info=True)
        return

    if not results:
        civitai.log('No info found on Civitai')
        return

    results = sorted(results, key=lambda x: datetime.fromisoformat(x['createdAt'].rstrip('Z')), reverse=True)

    civitai.log(f'Found {len(results)} hash matches')

    cc = opencc_utils.converter()

    # update the resources with the new info
    updated = 0
    for r in tqdm(results):
        if r is None:
            continue

        for file in r['files']:
            if 'hashes' not in file or 'SHA256' not in file['hashes']:
                continue
            file_hash = file['hashes']['SHA256']
            if file_hash.lower() not in hashes:
                continue

            sd_version = base_model_version.get(r['baseModel'])

            trained_words = [strip for s in r['trainedWords'] if (strip := s.strip().strip(','))]

            notes = ''
            if (model_id := r.get('modelId')) and (sub_id := r.get('id')):
                notes += f'https://civitai.com/models/{model_id}?modelVersionId={sub_id}\n'
            if trained_words:
                notes += '\n'.join(trained_words) + '\n'

            about_this_version = r.get('description')
            if about_this_version is not None:
                if version_description := about_this_version.strip():
                    notes += f'\nAbout this version:\n'
                    notes += version_description + '\n'

            data = {
                'description': cc.convert(r.get('model', {}).get('name', '')),
                'activation text': ', '.join(trained_words),
                # 'preferred weight': 0.8,
                'notes': notes,
                'civitai_metadata': r
            }
            if sd_version:
                data['sd version'] = sd_version

            if not (matches := [resource for resource in missing_info if file_hash.lower() == resource['hash']]):
                continue

            for resource in matches:
                Path(resource['path']).with_suffix('.json').write_text(json.dumps(data, indent=4, ensure_ascii=False))

            updated += 1

    civitai.log(f'Updated {updated} info files')


def run_get_load_info():
    with ThreadPoolExecutor() as executor:
        executor.submit(load_info)
    gr.Info('Finished fetching info from Civitai')


def run_get_info():
    with ThreadPoolExecutor() as executor:
        executor.submit(load_info)
        executor.submit(load_previews)
    gr.Info('Finished fetching info and preview images from Civitai')


def get_all_missing_previews():
    for resource in civitai.load_resource_list():
        if resource['hasInfo']:
            model_path = Path(resource['path'])
            model_info_path = model_path.with_suffix('.json')
            model_info = json.loads(model_info_path.read_text())
            civitai_metadata = model_info.get('civitai_metadata')
            if not civitai_metadata:
                continue
            for i, image in enumerate(civitai_metadata.get('images', [])):
                url_ext = os.path.splitext(image['url'])[1].lower()
                dest = model_path.with_stem(f'{model_path.stem}.preview.{i}').with_suffix(url_ext)
                if not dest.exists():
                    for ext in civitai.image_extensions:
                        if url_ext == ext:
                            continue
                        if model_path.with_stem(f'{model_path.stem}.preview.{i}').with_suffix(ext).exists():
                            break
                    else:
                        yield image['url'], dest


def re_download_preview_from_cache():
    gr.Info('Scanning for missing preview images')
    if missing_images_url_dest := set(get_all_missing_previews()):
        for url, dest in tqdm(missing_images_url_dest):
            # civitai.download_image(url, dest)
            civitai.download_image_auto_file_type(url, dest)
        gr.Info('Finished fetching preview images from Civitai')
    else:
        gr.Info('No missing preview images found')
