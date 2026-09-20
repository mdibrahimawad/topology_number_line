"""Immutable scientific runtime contract, independent of worker topology."""
import inspect
from pathlib import Path

from .dataset import package_versions
from .extract import infer_final_equals, extraction_dependency
from .storage import digest, fingerprint, read_json, write_json

LEGACY_SOURCES = {
    'extract.py': '3bd1bd30d7d855f20726abe60f1d948b568a429ba680df4d80803a42031eebcc',
    'dataset.py': 'a0f1c2b40165ee721e1861a2d3c494fa31f086d66dff3be918975b1f7345a30a',
    'storage.py': '984d0da5b29bd1bc8864e97a257882518b49f7cb21d17a253c33cc6e2291f34e',
    '__init__.py': '62b26ad6fdb0ea684370997007287e010387df2e668dd91fb65d3ea9e2129f98',
}


def ensure_extraction_contract(store, read_only=False):
    versions = package_versions(['torch', 'transformers', 'tokenizers', 'numpy'])
    contract = dict(extraction=extraction_dependency(store), packages=versions,
                    inference_source=inspect.getsource(infer_final_equals),
                    extraction_source=digest(Path(__file__).with_name('extract.py')),
                    dataset_source=digest(Path(__file__).with_name('dataset.py')),
                    config_source=digest(Path(__file__).with_name('__init__.py')))
    dep = fingerprint(contract)
    if 'extraction_contract' in store.manifest['artifacts']:
        if not store.valid('extraction_contract', dep):
            raise ValueError('Incompatible or corrupt extraction runtime contract')
        return dep
    if read_only:
        raise ValueError('Coordinator must prepare extraction runtime contract before fan-out')
    if 'extraction_runtime' in store.manifest['artifacts']:
        store.require('extraction_runtime')
        old = read_json(store.root / 'extraction_runtime.json')
        legacy_dep = fingerprint(dict(extraction=extraction_dependency(store), versions=versions, code=LEGACY_SOURCES))
        if (old.get('versions') != versions or old.get('source_hashes') != LEGACY_SOURCES
            or store.manifest['artifacts']['extraction_runtime']['dependency'] != legacy_dep):
            raise ValueError('Legacy extraction runtime is incompatible; refusing to mix representations')
    path = store.root / 'extraction_contract.json'
    write_json(path, contract)
    store.finish('extraction_contract', [path], dep)
    return dep


def warm_model_cache(store, token=None):
    """Coordinator only, called ONLY in an explicitly launched cloud extraction stage."""
    from huggingface_hub import snapshot_download
    cfg = store.manifest['config']
    return snapshot_download(cfg['model_id'], revision=cfg['model_revision'], token=token,
        allow_patterns=['*.json', '*.py', '*.bin', '*.safetensors', '*.model', '*.txt'])
