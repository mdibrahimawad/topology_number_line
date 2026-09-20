"""Bounded parallel reads and a single atomic, portable lightweight archive."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import os
from pathlib import Path
import stat
import tarfile


def export_archive(root, workers=16):
    root = Path(root)
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 16:
        raise ValueError('workers must be an integer from 1 through 16')
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Archive root must be a real directory')
    archive = root/'lightweight.tar.gz'
    partial = root/'lightweight.tar.gz.partial'
    paths = []
    # os.walk does not descend into symlink directories. Include their paths so
    # the same threaded regular-file check rejects them instead of omitting them.
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        parent = Path(directory)
        dirs[:] = sorted(d for d in dirs if d != 'hidden' and not d.endswith('.partial'))
        paths.extend(parent/d for d in dirs if (parent/d).is_symlink())
        for name in files:
            if name in ('hidden.npz','lightweight.tar.gz','package_receipt.json') or name.endswith('.partial'):
                continue
            paths.append(parent/name)
    paths.sort(key=lambda p:p.relative_to(root).as_posix())

    def read(path):
        original = path.lstat()
        if not stat.S_ISREG(original.st_mode):
            raise ValueError(f'Archive input is not a regular file: {path}')
        # O_NOFOLLOW also closes the final-component symlink race after lstat.
        flags = os.O_RDONLY | getattr(os,'O_NOFOLLOW',0) | getattr(os,'O_NONBLOCK',0)
        with os.fdopen(os.open(path,flags),'rb') as stream:
            current = os.fstat(stream.fileno())
            if not stat.S_ISREG(current.st_mode) or (current.st_dev,current.st_ino)!=(original.st_dev,original.st_ino):
                raise ValueError(f'Archive input changed while opening: {path}')
            data = stream.read()
            final = os.fstat(stream.fileno())
        if len(data)!=original.st_size or final.st_mtime_ns!=original.st_mtime_ns or final.st_size!=original.st_size:
            raise ValueError(f'Archive input changed while reading: {path}')
        return path.relative_to(root).as_posix(),data

    # A wave is fully consumed before the next is submitted: at most workers
    # file payloads are retained, even when compression is slower than reads.
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool, open(partial,'wb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=3,mtime=0,filename='') as zipped:
                with tarfile.open(fileobj=zipped,mode='w|') as tar:
                    completed = 0
                    for start in range(0,len(paths),workers):
                        for name,data in pool.map(read,paths[start:start+workers]):
                            info = tarfile.TarInfo(name)
                            info.size=len(data);info.mode=0o644;info.mtime=0
                            with io.BytesIO(data) as payload:
                                tar.addfile(info,payload)
                            del data
                            completed+=1
                            if completed%256==0:
                                print(f'[EXPORT] {completed}/{len(paths)} files',flush=True)
            raw.flush();os.fsync(raw.fileno())
        sha = hashlib.sha256()
        with open(partial,'rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                sha.update(block)
        size=partial.stat().st_size
        os.replace(partial,archive)
        return dict(archive_sha256=sha.hexdigest(),files=len(paths),bytes=size,
            hidden_states='Retained remotely in model/hidden chunks; excluded from lightweight archive')
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
