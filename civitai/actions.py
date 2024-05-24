from . import lib as civitai
from pathlib import Path
import threading
import json
from modules import shared

previewable_types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']


def load_previews():
    nsfw_previews = shared.opts.data.get('civitai_nsfw_previews', True)

    civitai.log(f"Check resources for missing preview images")
    resources = civitai.load_resource_list()
    resources = [r for r in resources if r['type'] in previewable_types]

    # get all resources that are missing previews
    missing_previews = [r for r in resources if r['hasPreview'] is False]
    civitai.log(f"Found {len(missing_previews)} resources missing preview images")
    hashes = [r['hash'] for r in missing_previews]

    # split hashes into batches of 100 and fetch into results
    results = []
    try:
        for i in range(0, len(hashes), 100):
            batch = hashes[i:i + 100]
            results.extend(civitai.get_all_by_hash(batch))
    except:
        civitai.log("Failed to fetch preview images from Civitai")
        return

    if len(results) == 0:
        civitai.log("No preview images found on Civitai")
        return

    civitai.log(f"Found {len(results)} hash matches")

    # update the resources with the new preview
    updated = 0
    for r in results:
        if (r is None): continue

        for file in r['files']:
            if not 'hashes' in file or not 'SHA256' in file['hashes']: continue
            hash = file['hashes']['SHA256']
            if hash.lower() not in hashes: continue
            images = r['images']
            if (nsfw_previews is False): images = [i for i in images if i['nsfw'] is False]
            if (len(images) == 0): continue
            image_url = images[0]['url']
            civitai.update_resource_preview(hash, image_url)
            updated += 1

    civitai.log(f"Updated {updated} preview images")


def run_load_previews():
    threading.Thread(target=load_previews).start()


actionable_types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']


def load_info():
    civitai.log("Check resources for missing info files")
    resources = civitai.load_resource_list()
    resources = [r for r in resources if r['type'] in actionable_types]

    # get all resources that have no info files
    missing_info = [r for r in resources if r['hasInfo'] is False]
    civitai.log(f"Found {len(missing_info)} resources missing info files")
    hashes = [r['hash'] for r in missing_info]

    # split hashes into batches of 100 and fetch into results
    results = []
    try:
        for i in range(0, len(hashes), 100):
            batch = hashes[i:i + 100]
            results.extend(civitai.get_all_by_hash(batch))
    except:
        civitai.log("Failed to fetch info from Civitai")
        return

    if len(results) == 0:
        civitai.log("No info found on Civitai")
        return

    civitai.log(f"Found {len(results)} hash matches")

    # update the resources with the new info
    updated = 0
    for r in results:
        if (r is None):
            continue

        for file in r['files']:
            if not 'hashes' in file or not 'SHA256' in file['hashes']:
                continue
            file_hash = file['hashes']['SHA256']
            if file_hash.lower() not in hashes:
                continue

            if "SD 1" in r['baseModel']:
                sd_version = "SD1"
            elif "SD 2" in r['baseModel']:
                sd_version = "SD2"
            elif "SDXL" in r['baseModel']:
                sd_version = "SDXL"
            else:
                sd_version = "unknown"
            data = {
                "description": "",
                "sd version": sd_version,
                "activation text": ", ".join(r['trainedWords']),
                "preferred weight": 0.8,
                "notes": "",
                "html": r['description'],
            }

            matches = [resource for resource in missing_info if file_hash.lower() == resource['hash']]
            if len(matches) == 0:
                continue

            for resource in matches:
                Path(resource['path']).with_suffix(".json").write_text(json.dumps(data, indent=4))
            updated += 1

    civitai.log(f"Updated {updated} info files")


def run_get_load_info():
    threading.Thread(target=load_info).start()


def run_get_info():
    run_load_previews()
    run_get_load_info()
