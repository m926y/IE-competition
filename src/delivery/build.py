"""Build an allowlisted four-file competition delivery, without external writes."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import zipfile

import pandas as pd
from .report import build_report
from .validate import validate, allowed_archive_name

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def build():
    folder = ROOT / 'submission/final'
    folder.mkdir(parents=True, exist_ok=True)
    audit = ROOT / 'outputs/delivery'
    audit.mkdir(parents=True, exist_ok=True)
    sources = {'forecast_result.csv': ROOT / 'outputs/prediction/forecast_result.csv',
               'safety_scores.csv': ROOT / 'outputs/scoring/safety_scores.csv'}
    for name, source in sources.items():
        shutil.copyfile(source, folder / name)
    build_report(folder / 'method_report.pdf')
    members = {}
    for tree in ['src', 'tests']:
        for path in sorted((ROOT / tree).rglob('*.py')):
            members[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for path in sorted((ROOT / 'src').rglob('README.md')):
        members[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    docs = ['SUBMISSION_README.md', 'final_optimization.md']
    for name in docs:
        members['docs/' + name] = (ROOT / 'docs' / name).read_bytes()
    members['README.md'] = (ROOT / 'docs/SUBMISSION_README.md').read_bytes()
    members['requirements.txt'] = (ROOT / 'requirements.txt').read_bytes()
    # Lock installed environment versions, never local URLs, tokens or pip config.
    distributions = {d.metadata['Name']: d.version for d in metadata.distributions()
                     if d.metadata['Name'].lower() not in {'pip', 'setuptools', 'wheel'}}
    members['requirements-lock.txt'] = ('\n'.join(f'{n}=={v}' for n, v in sorted(distributions.items())) + '\n').encode()
    for config in sorted((ROOT / 'configs/submission').glob('*.json')):
        members['configs/submission/' + config.name] = config.read_bytes()
    for name, source in [('prediction_evaluation', 'outputs/prediction/evaluation.json'),
                         ('prediction_robustness', 'outputs/prediction/robustness.json'),
                         ('scoring_evaluation', 'outputs/scoring/evaluation.json')]:
        members['evidence/' + name + '.json'] = (ROOT / source).read_bytes()
    if not all(allowed_archive_name(n) for n in members):
        raise ValueError('Non-whitelisted package file')
    manifest = {'python': platform.python_version(), 'platform': platform.system(),
                'files': {n: hashlib.sha256(data).hexdigest() for n, data in sorted(members.items())},
                'data_policy': 'No competition raw data, cleaned tables, vehicle features, model binaries or access links'}
    members['MANIFEST.json'] = encoded(manifest)
    with zipfile.ZipFile(folder / 'source_code.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    ids = pd.read_csv(ROOT / 'data/processed/cleaned/vehicle_profile.csv', dtype={'gpsno': str}).gpsno
    result = validate(folder, ids)
    (audit / 'acceptance.json').write_bytes(encoded(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
