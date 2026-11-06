"""Local input verification and regression against saved scientific results."""
from pathlib import Path
import csv, hashlib, json, math, platform
ROOT=Path(__file__).resolve().parents[1]
EXTERNAL=json.loads((ROOT/'provenance.json').read_text()).get('analysis_mode')=='external_cft'
COMPARISONS={}
def checked(name, operation):
    try:
        operation();COMPARISONS[name]={'matched':True}
    except AssertionError as error:
        if not EXTERNAL:raise
        COMPARISONS[name]={'matched':False,'first_difference':str(error)}
def verify_inputs():
    p=ROOT/'provenance.json'
    if not p.exists():
        raise FileNotFoundError('Missing provenance.json')
    for row in json.loads(p.read_text())['inputs']:
        path=ROOT/row['path']
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'], path
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    (ROOT/'figures').mkdir(parents=True,exist_ok=True)
def compare(a,b,where='',ignore=()):
    if isinstance(a,dict):
        assert (set(a)-set(ignore))==(set(b)-set(ignore)),where
        for k in a:
            if k not in ignore:compare(a[k],b[k],where+'/'+k,ignore)
    elif isinstance(a,list):
        assert len(a)==len(b),(where,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,where+f'/{i}',ignore)
    else:
        try:x,y=float(a),float(b)
        except (TypeError,ValueError):
            assert a==b,(where,a,b);return
        assert (math.isnan(x) and math.isnan(y)) or math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12),(where,a,b)
def check_csv(name,ignore=()):
    def read(p):
        with p.open(newline='') as f:return list(csv.DictReader(f))
    checked(name,lambda:compare(read(ROOT/'data/processed'/name),read(ROOT/'data/reference'/name),name,ignore))
def check_json(name,ignore=()):
    checked(name,lambda:compare(json.loads((ROOT/'data/processed'/name).read_text()),json.loads((ROOT/'data/reference'/name).read_text()),name,ignore))
def finish(details):
    import numpy,matplotlib
    report=dict(passed=all(r['matched'] for r in COMPARISONS.values()),analysis_completed=True,analysis_mode='external_cft' if EXTERNAL else 'published_inputs',reference_comparisons=COMPARISONS,reference_rtol=1e-10,reference_atol=1e-12,**details,
                environment=dict(python=platform.python_version(),numpy=numpy.__version__,matplotlib=matplotlib.__version__))
    (ROOT/'data/processed/validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
