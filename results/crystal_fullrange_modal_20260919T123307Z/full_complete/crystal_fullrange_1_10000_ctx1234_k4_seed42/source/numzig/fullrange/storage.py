"""Atomic, checksummed checkpoints. A commit callback is Modal Volume.commit remotely."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic(path, lambda f: f.write(json.dumps(value, indent=2, sort_keys=True,
                                            allow_nan=False).encode()))


def write_npz(path, **arrays):
    atomic(path, lambda f: np.savez(f, **arrays))


def read_json(path):
    return json.loads(Path(path).read_text())


class Store:
    def __init__(self, root, config=None, commit=lambda: None):
        self.root = Path(root)
        self.commit = commit
        self.path = self.root / 'manifest.json'
        if self.path.exists():
            self.manifest = read_json(self.path)
            if config is not None and self.manifest['config'] != config:
                raise ValueError('Incompatible experiment configuration; use a separate output directory')
            if fingerprint(self.manifest['config']) != self.manifest['config_fingerprint']:
                raise ValueError('Manifest configuration fingerprint is invalid')
        else:
            if config is None:
                raise FileNotFoundError('No experiment manifest; prepare dataset first')
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError('Refusing to overwrite an unrelated nonempty output directory')
            self.manifest = dict(config=config, config_fingerprint=fingerprint(config), artifacts={}, stages={})
            self.save()
            self.commit()

    def save(self):
        write_json(self.path, self.manifest)

    def output_path(self, name):
        return self.root / name

    def log(self, message):
        print(message, flush=True)
        with (self.root / 'progress.log').open('a') as f:
            f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {message}\n')

    def valid(self, key, dependency=None):
        item = self.manifest['artifacts'].get(key)
        if not item:
            return False
        if dependency is not None and item['dependency'] != dependency:
            return False
        try:
            good = all((self.root / p).is_file() and digest(self.root / p) == sha
                       for p, sha in item['files'].items())
        except OSError:
            good = False
        if good and item['status'] == 'prepared':
            # Data survived interruption between the two commits; adopt without recomputation.
            self.commit()
            item['status'] = 'complete'
            self.save()
            self.commit()
            self.log(f'[RECOVER] {key}: validated durable data; no recomputation')
        return good and item['status'] == 'complete'

    def finish(self, key, paths, dependency, metadata=None):
        files = {str(Path(p).relative_to(self.root)): digest(p) for p in paths}
        item = dict(status='prepared', dependency=dependency, files=files, metadata=metadata or {})
        self.manifest['artifacts'][key] = item
        self.save()
        self.commit()  # Data and hashes durable BEFORE the completion marker is written.
        item['status'] = 'complete'
        self.save()
        self.commit()  # Failure stops the stage; the next invocation validates/adopts it.
        self.log(f'[COMPLETE] {key}')

    def stage(self, name, status):
        self.manifest['stages'][name] = status
        self.save()
        self.commit()

    def artifact_fingerprint(self, key):
        return fingerprint(self.manifest['artifacts'][key]['files'])

    def require(self, key):
        if not self.valid(key):
            raise ValueError(f'Missing or invalid {key}; resume its producing stage (inference is never implicit)')
        return self.manifest['artifacts'][key]
