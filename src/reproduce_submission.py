"""Reproduce official outputs with frozen B configuration; never launch a new B search."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', action='store_true', help='Build cleaned tables from original gzip files')
    parser.add_argument('--rebuild-features', action='store_true')
    parser.add_argument('--package', action='store_true', help='Also rebuild PDF and four-file folder')
    args = parser.parse_args()
    def command(module, *options):
        subprocess.run([sys.executable, '-m', module, *options], check=True, cwd=ROOT)
    if args.clean:
        command('src.data.clean_data')
    if args.rebuild_features or not (ROOT / 'outputs/features/feature_quality.json').exists():
        command('src.features.pipeline', '--overwrite')
    command('src.features.audit_delivery')
    from src.prediction.optimize import reproduce_frozen
    reproduce_frozen()
    command('src.prediction.robustness')
    command('src.scoring.enhanced')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'], check=True, cwd=ROOT)
    if args.package:
        command('src.delivery.build')


if __name__ == '__main__':
    run()
