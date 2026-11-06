"""Regenerate the CSV inventory while preserving the PDF project exclusion."""
import contextlib,hashlib,io,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[5]
S=ROOT/'research_note/calculation_table/support';script=S/'build_table.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def generate(name,source):
    target=H/'control'/name/'support'
    target.mkdir(parents=True,exist_ok=True)
    for p in S.iterdir():
        if p.is_file():shutil.copy2(p,target/p.name)
    code=source.replace('OUT = Path(__file__).resolve().parent',f'OUT = Path({str(target)!r})')
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):exec(compile(code,str(script),'exec'),{'__file__':str(script),'__name__':'__main__'})
    return target,buf.getvalue()
before,oldlog=generate('table_baseline',(H/'control/build_table_before.py').read_text())
after,newlog=generate('table_updated',script.read_text())
assert (before.parent/'main.tex').read_bytes()==(after.parent/'main.tex').read_bytes(),'Unexpected PDF change: inspect before copying'
import csv
old=list(csv.DictReader((before/'table_rows.csv').open()));new=list(csv.DictReader((after/'table_rows.csv').open()))
added=[r for r in new if r not in old];removed=[r for r in old if r not in new]
assert not removed,removed
assert len(added)==3,added
assert all('charge_tube_20261106' in r['evidence'] and r['tube']=='[1,4,1]' for r in added)
changes=[]
for p in sorted(after.glob('*.csv')):
    dest=S/p.name
    if p.read_bytes()==dest.read_bytes():continue
    changes.append(dict(path=str(dest.relative_to(ROOT)),before=sha(dest),after=sha(p)))
    shutil.copy2(p,dest)
record=dict(added_CFT_series=3,removed_rows=0,added_rows=added,changed_csv=changes,
    PDF_exclusion_preserved=True,PDF_modified=False,main_tex_modified=False,before_log=oldlog,after_log=newlog)
(H/'control/calculation_table_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('CFT series added:',len(added),'CSV files updated:',len(changes),'PDF unchanged')
