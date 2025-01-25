from . import lib as civitai, opencc_utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules import errors, images
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import gradio as gr
import threading
import shutil
import glob
import json
import re
import os


previewable_types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']
actionable_types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']
base_model_version = {
    # 'SD 1': 'SD1',
    # 'SD 1.5': 'SD1',
    # 'SD 2': 'SD2',
    # 'SDXL': 'SDXL',
    # 'Pony': 'SDXL',
    # 'Illustrious': 'SDXL',
    'SD 3': 'SD3',
}
lock = threading.Lock()


def show_started():
    gr.Info('Civitai: Started')


def show_finished():
    gr.Info('Civitai: Finished')


def load_info():
    show_started()
    with lock:
        load_info_inner()
    show_finished()


def load_info_inner():
    civitai.log('Check resources for missing info files')
    resources = civitai.load_resource_list()
    resources = [r for r in resources if r['type'] in actionable_types]

    # get all resources that have no info files
    missing_info = [r for r in resources if r['hasInfo'] is False]
    civitai.log(f'Found {len(missing_info)} resources missing info files')
    hashes = [r['hash'] for r in missing_info]

    # # split hashes into batches of 100 and fetch into results
    # results = []
    # try:
    #     for i in range(0, len(hashes), 100):
    #         batch = hashes[i:i + 100]
    #         results.extend(civitai.get_all_by_hash(batch))
    #
    #
    #
    # except Exception:
    #     errors.report('Failed to fetch info from Civitai', exc_info=True)
    #     return
    results = civitai.get_all_by_hash_with_cache(hashes)

    if not results:
        civitai.log('No info found on Civitai')
        return

    # results = sorted(results, key=lambda x: datetime.fromisoformat(x['createdAt'].rstrip('Z')), reverse=True)

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

            description = f"{r.get('model', {}).get('name', '')}\n{r.get('name', '')}"
            data = {
                'description': cc.convert(description),
                'activation text': ', '.join([prompt.strip() for prompts in trained_words for prompt in prompts.split(',')]),
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


def run_get_info():
    show_started()
    with lock:
        load_info_inner()
        load_previews_v2_inner()
    show_finished()


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
                    for ext in civitai.preview_extensions:
                        if url_ext == ext:
                            continue
                        if model_path.with_stem(f'{model_path.stem}.preview.{i}').with_suffix(ext).exists():
                            break
                    else:
                        yield image['url'], dest


def re_download_preview_from_cache():
    if missing_images_url_dest := set(get_all_missing_previews()):
        with ThreadPoolExecutor(max_workers=10) as executor:
            with tqdm(total=len(missing_images_url_dest)) as pbar:
                futures = [
                    executor.submit(civitai.download_image_auto_file_type, url, dest, pbar)
                    for url, dest in missing_images_url_dest
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    finally:
                        pbar.update(1)
        gr.Info('Finished fetching preview images from Civitai')


def select_preview(image_list):
    for img_path in image_list:
        try:
            geninfo, items = images.read_info_from_image(Image.open(img_path))
            if geninfo:
                return img_path
        except Exception:
            errors.report(f'Error reading image {img_path}', exc_info=True)

    return image_list[0]


def load_previews_v2():
    show_started()
    with lock:
        load_previews_v2_inner()
    show_finished()


def load_previews_v2_inner():
    re_download_preview_from_cache()

    # nsfw_previews = shared.opts.civitai_nsfw_previews

    resources = civitai.load_resource_list()
    resources = [r for r in resources if r['type'] in previewable_types]

    # get all resources that are missing previews
    missing_previews = [r for r in resources if r['hasPreview'] is False]

    civitai.log(f'Found {len(missing_previews)} resources missing preview images')

    paths = [Path(r['path']) for r in missing_previews]
    for path in paths:
        matching_files = [file for file in path.parent.glob(f'{glob.escape(path.stem)}.*')]
        file_pattern = re.compile(f'^{re.escape(path.stem)}\\.preview\\.[0-9]+\\.[^.]+$')
        matching_files = list(filter(lambda x: x.suffix.lower() in civitai.image_extensions and file_pattern.match(x.name), matching_files))
        if matching_files:
            matching_files = sorted(matching_files, key=lambda x: int(x.stem.split('.')[-1]))
            img = select_preview(matching_files)
            preview_path = img.with_stem(img.stem.rpartition(".")[0])
            if not preview_path.exists():
                shutil.copy(img, preview_path)
