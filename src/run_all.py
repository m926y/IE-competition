"""Rebuild A, train B, score C locally: python -m src.run_all."""
import argparse
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-features',action='store_true',help='Reuse already QA-corrected A outputs')
    args=parser.parse_args()
    modules=[] if args.reuse_features else [('src.features.pipeline',['--overwrite'])]
    modules += [('src.features.audit_delivery',[]),('src.prediction.integrated',[]),('src.scoring.integrated',[])]
    for module,options in modules:
        subprocess.run([sys.executable,'-m',module,*options],check=True)

if __name__=='__main__':
    main()
