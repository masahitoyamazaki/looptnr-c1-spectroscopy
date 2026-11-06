"""Optional isolated output workspace for the existing scientific pipeline."""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

def prepare_run():
    p=argparse.ArgumentParser(description='Analyze saved CFT and draw this figure. No tensor measurement or RG is run.')
    p.add_argument('--cft-dir',type=Path,help='Complete replacement CFT dataset; requires --output-dir.')
    p.add_argument('--output-dir',type=Path,help='Empty/new directory receiving an independent analysis bundle.')
    args=p.parse_args()
    if args.cft_dir and not args.output_dir:p.error('--cft-dir requires --output-dir to preserve the published inputs.')
    if not args.output_dir:return False
    out=args.output_dir.expanduser().resolve()
    if out==ROOT or ROOT in out.parents or out in ROOT.parents:p.error('--output-dir must be outside this figure bundle and its ancestors.')
    if out.exists() and any(out.iterdir()):p.error('--output-dir must be empty or new.')
    external=args.cft_dir.expanduser().resolve() if args.cft_dir else None
    if external:
        if not external.is_dir():p.error('--cft-dir is not a directory.')
        if external==out or out in external.parents:p.error('--output-dir must not contain the input CFT directory.')
        meta=json.loads((ROOT/'provenance.json').read_text())
        expected=[r['path'][len('data/cft/'):] for r in meta['inputs'] if r['path'].startswith('data/cft/')]
        if not expected:p.error('This schematic figure has no CFT input.')
        missing=[s for s in expected if not (external/s).is_file()]
        if missing:p.error(f'Incomplete CFT dataset: {len(missing)} required files missing; first entries: {missing[:8]}. A single-point smoke test is not a full figure dataset.')
    shutil.copytree(ROOT,out,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.DS_Store','.deps'))
    for name in ['data/processed','figures']:
        dest=out/name
        if dest.exists():shutil.rmtree(dest)
        dest.mkdir(parents=True)
    if external:
        shutil.rmtree(out/'data/cft');shutil.copytree(external,out/'data/cft')
        meta=json.loads((out/'provenance.json').read_text())
        (out/'provenance/published_inputs.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        meta['analysis_mode']='external_cft'
        meta['external_cft_directory']=str(external)
        meta['inputs']=[r for r in meta['inputs'] if not r['path'].startswith('data/cft/')]
        meta['inputs'] += [dict(path=str(f.relative_to(out)),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                           for f in sorted((out/'data/cft').rglob('*')) if f.is_file()]
        (out/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    subprocess.run([sys.executable,'-B',str(out/'scripts/reproduce.py')],cwd=out,check=True)
    print(f'Analysis and figure outputs: {out}')
    return True
