"""Download the completed graph archive and verify it before extraction."""
from pathlib import Path
import hashlib
import json
import os
import tarfile

import modal

destination = Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/taskrules')
remote = '/taskrules_20260920_v3'
volume = modal.Volume.from_name('numberline-zigzag-results', environment_name='main')
receipt_bytes = b''.join(volume.read_file(remote + '/package_receipt.json'))
receipt = json.loads(receipt_bytes)
archive = destination / 'lightweight.tar.gz'
temporary = archive.with_suffix('.tar.gz.partial')
digest = hashlib.sha256()
with temporary.open('wb') as stream:
    for block in volume.read_file(remote + '/lightweight.tar.gz'):
        stream.write(block)
        digest.update(block)
assert digest.hexdigest() == receipt['archive_sha256'], 'Archive checksum mismatch'
assert temporary.stat().st_size == receipt['bytes'], 'Archive byte count mismatch'
os.replace(temporary, archive)
root = destination / 'run'
root.mkdir(exist_ok=True)
with tarfile.open(archive) as package:
    members = package.getmembers()
    assert len(members) == receipt['files'], 'Archive file count mismatch'
    for member in members:
        assert member.isfile(), 'Only regular files expected'
        assert (root / member.name).resolve().is_relative_to(root.resolve()), 'Unsafe archive path'
    package.extractall(root, members=members, filter='data')
(root / 'package_receipt.json').write_bytes(receipt_bytes)
print(json.dumps(dict(archive_bytes=receipt['bytes'], archive_files=receipt['files'],
                     archive_sha256=receipt['archive_sha256'], verified=True, extracted=str(root)), indent=2))
