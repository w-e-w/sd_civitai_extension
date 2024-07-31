from . import lib as civitai
from pathlib import Path
import threading
import json
from modules import shared, errors
from tqdm import tqdm

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
    threading.Thread(target=load_previews).start()


actionable_types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']


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

    civitai.log(f'Found {len(results)} hash matches')

    from opencc import OpenCC
    cc = OpenCC('s2tw')

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

            if 'SD 1' in r['baseModel']:
                sd_version = 'SD1'
            elif 'SD 2' in r['baseModel']:
                sd_version = 'SD2'
            elif 'SDXL' in r['baseModel']:
                sd_version = 'SDXL'
            else:
                sd_version = 'unknown'

            trained_words = [strip for s in r['trainedWords'] if (strip := s.strip()).strip(',')]

            notes = ''
            if (model_id := r.get('modelId')) and (sub_id := r.get('id')):
                notes += f'https://civitai.com/models/{model_id}?modelVersionId={sub_id}\n'
            if trained_words:
                notes += '\n'.join(trained_words) + '\n'

            data = {
                'description': cc.convert(r.get('model', {}).get('name', '')),
                'sd version': sd_version,
                'activation text': ', '.join(trained_words),
                # 'preferred weight': 0.8,
                'notes': notes,
                'civitai_metadata': r
            }

            if not (matches := [resource for resource in missing_info if file_hash.lower() == resource['hash']]):
                continue

            for resource in matches:
                Path(resource['path']).with_suffix('.json').write_text(json.dumps(data, indent=4, ensure_ascii=False))

            updated += 1

    civitai.log(f'Updated {updated} info files')


def run_get_load_info():
    threading.Thread(target=load_info).start()


def run_get_info():
    run_load_previews()
    run_get_load_info()
