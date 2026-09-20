import hashlib
from pathlib import Path
import tarfile
import pytest
from export import export_archive


def test_content_exclusions_metadata_and_repeat(tmp_path):
    (tmp_path/'gallery').mkdir();(tmp_path/'hidden').mkdir()
    wanted={'index.html':b'<p>Results</p>','gallery/plot.json':b'{"scores": [1,2,3]}'}
    for name,value in wanted.items():(tmp_path/name).write_bytes(value)
    for name in ('hidden/data.npz','hidden.npz','incomplete.partial','package_receipt.json'):
        (tmp_path/name).write_bytes(b'excluded')
    result=export_archive(tmp_path,workers=2);archive=tmp_path/'lightweight.tar.gz'
    assert result['files']==2 and result['bytes']==archive.stat().st_size
    assert result['archive_sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames())==sorted(wanted)
        for member in tar:
            assert member.mode==0o644 and member.mtime==0 and member.isfile()
            assert tar.extractfile(member).read()==wanted[member.name]
    before=archive.read_bytes()
    assert export_archive(tmp_path,workers=1)['archive_sha256']==result['archive_sha256']
    assert archive.read_bytes()==before


@pytest.mark.parametrize('directory',[False,True])
def test_symlink_rejected_and_existing_archive_preserved(tmp_path,directory):
    (tmp_path/'ok.txt').write_text('ok');export_archive(tmp_path)
    archive=tmp_path/'lightweight.tar.gz';before=archive.read_bytes()
    target=tmp_path/'target'
    target.mkdir() if directory else target.write_text('do not follow')
    (tmp_path/'link').symlink_to(target,target_is_directory=directory)
    with pytest.raises(ValueError,match='not a regular file'):export_archive(tmp_path)
    assert archive.read_bytes()==before and not (tmp_path/'lightweight.tar.gz.partial').exists()


def test_fifo_rejected_without_blocking(tmp_path):
    import os
    os.mkfifo(tmp_path/'pipe')
    with pytest.raises(ValueError,match='not a regular file'):export_archive(tmp_path)


def test_parallel_reader_has_bounded_waves(tmp_path,monkeypatch):
    import export
    for i in range(35):(tmp_path/f'{i:03d}.txt').write_text(str(i))
    real=export.ThreadPoolExecutor;waves=[]
    class Observed(real):
        def map(self,function,items):
            items=list(items);waves.append(len(items));return super().map(function,items)
    monkeypatch.setattr(export,'ThreadPoolExecutor',Observed)
    assert export_archive(tmp_path,workers=16)['files']==35
    assert waves==[16,16,3]
