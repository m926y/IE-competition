"""Local reproducibility records; never put data values or access tokens in Git."""
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
from importlib.metadata import version

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def record_run(member,inputs,outputs):
    root=Path(__file__).resolve().parents[2]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=root/'runs'/member/stamp; directory.mkdir(parents=True)
    record={'created_at_utc':stamp,'inputs_sha256':{str(p.relative_to(root)):sha256(p) for p in inputs},
            'outputs_sha256':{str(p.relative_to(root)):sha256(p) for p in outputs if p.is_file()},
            'source_sha256':{str(p.relative_to(root)):sha256(p) for p in (root/'src').rglob('*.py')},
            'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','duckdb','pyarrow']}}
    (directory/'run_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return directory
