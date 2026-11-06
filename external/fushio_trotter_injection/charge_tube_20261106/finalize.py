"""Aggregate completed readouts and copy each series beside its checkpoints."""
import hashlib,json,shutil,tomllib
from pathlib import Path
import toml
H=Path(__file__).resolve().parent;ROOT=H.parents[5]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
copies=[];(H/'series').mkdir(exist_ok=True)
for directory in sorted((H/'data').iterdir()):
    if not directory.is_dir():continue
    files=sorted(directory.glob('*.toml'));assert len(files)==9
    values=[tomllib.loads(p.read_text()) for p in files]
    assert [v['step'] for v in values]==list(range(0,17,2))
    assert all(v['status']=='measured' for v in values)
    cfg=values[0]['config'];aggregate=dict(config=cfg,h=values[0]['h'],status='complete',shape=[1,4,1],measurements=values,
        source_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in files])
    source=H/'series'/f"{cfg['id']}.toml";source.write_text(toml.dumps(aggregate))
    # Retain the original RG6..16 CFT copy; publish the extension as a new readout version.
    dest=ROOT/cfg['checkpoint_path']/'CFT/tube=1_4_1/measurement=charged_note13_early_20261106/results.toml'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(dest)==sha(source),'Refuse overwrite of different CFT data'
    else:shutil.copy2(source,dest)
    copies.append(dict(source=str(source.relative_to(ROOT)),destination=str(dest.relative_to(ROOT)),sha256=sha(source),measurements=9,drive_uploaded=False))
(H/'control/local_cft_copies.json').write_text(json.dumps(copies,indent=2)+'\n')
print('saved local series',len(copies))
