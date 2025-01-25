from fastapi.exceptions import HTTPException
from starlette.responses import FileResponse
from modules import cache, shared, sd_models
from fastapi import FastAPI
from pathlib import Path
import gradio as gr
import threading
import asyncio
import json
import re
import os


parse_model_re = re.compile(r'models/(\d+)(?:.*modelVersionId=(\d+))?')
mhtml_cache = cache.cache('civilai_mhtml')
metadata_id_cache = cache.cache('civilai_metadata_id')
lock = threading.Lock()


class MhtmlDB:

    snapshot_content_location = 'Snapshot-Content-Location: '

    def __init__(self):
        self.path_dirs = set()
        self.full_url_db = None
        self.full_model_id_db = None
        self.model_id_db = None
        self.metadata_mhtml_dict = None
        self.loaded = None

    def add_dir(self, path_dir):
        if path_dir:
            self.path_dirs.add(Path(path_dir))

    def reload(self):
        self.full_url_db = {}
        self.full_model_id_db = {}
        self.model_id_db = {}
        self.metadata_mhtml_dict = {}
        self.loaded = False
        self.get_metadata_mhtml_dict()

    @staticmethod
    def get_snapshot_content_location(mhtml_path: Path):
        with open(mhtml_path, 'r') as f:
            for line in f:
                if MhtmlDB.snapshot_content_location in line:
                    url = line[len(MhtmlDB.snapshot_content_location):-1]
                    return url

    def get_mhtml_dict(self):
        for path_dir in self.path_dirs:
            for mhtml in path_dir.rglob('*.mhtml'):
                url, model_id, version_id = None, None, None
                mhtml_mtime = mhtml.stat().st_mtime
                cache_mhtml = mhtml_cache.get(str(mhtml))

                if cache_mhtml and cache_mhtml['mtime'] == mhtml_mtime:
                    url, model_id, version_id = cache_mhtml['url'], cache_mhtml['model_id'], cache_mhtml['version_id']
                    if url:
                        self.full_url_db[url] = mhtml
                    if model_id and version_id:
                        self.full_model_id_db[model_id, version_id] = mhtml
                    if model_id:
                        self.model_id_db[model_id] = mhtml
                    continue

                url = self.get_snapshot_content_location(mhtml)
                if url and (match := parse_model_re.search(url)):
                    model_id, version_id = match.groups()
                    if model_id:
                        self.model_id_db[model_id] = mhtml
                        if version_id:
                            self.full_model_id_db[model_id, version_id] = mhtml

                mhtml_cache[str(mhtml)] = {'mtime': mhtml_mtime, 'url': url, 'model_id': model_id, 'version_id': version_id}

    def get_mhtml(self, url=None, model_id=None, version_id=None):
        if model_id:
            model_id = str(model_id)
        if version_id:
            version_id = str(version_id)
        if url and (res := self.full_url_db.get(url)):
            return res
        if model_id and version_id and (res := self.full_model_id_db.get((model_id, version_id))):
            return res
        if model_id and (res := self.model_id_db.get(model_id)):
            return res

    @staticmethod
    def get_id_from_json(path_json: Path):
        cache_metadata_id = metadata_id_cache.get(str(path_json))
        cache_mtime = path_json.stat().st_mtime
        if cache_metadata_id and cache_metadata_id['mtime'] == cache_mtime:
            model_id = cache_metadata_id['model_id']
            version_id = cache_metadata_id['version_id']
        else:
            try:
                metadata = json.loads(path_json.read_text())
                if 'civitai_metadata' not in metadata:
                    return None, None
                civitai_metadata = metadata['civitai_metadata']
                model_id = civitai_metadata.get('modelId')
                version_id = civitai_metadata.get('id')
            except json.JSONDecodeError:
                model_id, version_id = None, None
            metadata_id_cache[str(path_json)] = {'mtime': cache_mtime, 'model_id': model_id, 'version_id': version_id}
        return model_id, version_id

    def get_epub_from_json(self, path_json: Path):
        model_id, version_id = self.get_id_from_json(path_json)
        res = self.get_mhtml(model_id=model_id, version_id=version_id)
        return res

    def load(self):
        with lock:
            if self.loaded:
                return
            self.get_mhtml_dict()
            for path_dir in self.path_dirs:
                for path_json in path_dir.rglob('*.json'):
                    if res := self.get_epub_from_json(path_json):
                        self.metadata_mhtml_dict[str(path_json.with_suffix(''))] = str(res)
            self.loaded = True

    def get_metadata_mhtml_dict(self):
        if self.loaded:
            return self.metadata_mhtml_dict
        self.load()
        return self.metadata_mhtml_dict


mhtml_db = MhtmlDB()

mhtml_db.add_dir(sd_models.model_path)
mhtml_db.add_dir(shared.cmd_opts.ckpt_dir)
mhtml_db.add_dir(shared.cmd_opts.lora_dir)
mhtml_db.add_dir(shared.cmd_opts.embeddings_dir)
mhtml_db.add_dir(shared.cmd_opts.hypernetwork_dir)


async def preload_mhtml_db():
    mhtml_db.reload()

asyncio.run(preload_mhtml_db())

pass

# if __name__ == '__main__':

#     models_dir = Path(r'B:\Stable Diffusion\Models')
#     mhtml_db = MhtmlDB(models_dir)
#     for j in models_dir.rglob('*.json'):
#         pass
#         r = mhtml_db.get_epub_from_json(j)
#


def add_api(_: gr.Blocks, app: FastAPI):
    @app.get("/civitai/get-mhtml_dict")
    async def get_mhtml_dict():
        return mhtml_db.get_metadata_mhtml_dict()

    @app.get("/civitai/get-mhtml")
    async def get_mhtml(filename: str):
        return FileResponse(
            filename,
            headers={
                "Accept-Ranges": "bytes",
                "Content-type": 'multipart/related; type="text/html"; boundary="----MultipartBoundary--01bt8bpZk5svUyn5d5gztLgMbzlnFT9BvAPkdbOybF----"',
            },
        )
