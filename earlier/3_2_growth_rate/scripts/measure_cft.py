#!/usr/bin/env python3
"""Measure saved checkpoints listed in measurement_plan.csv; never run RG."""
from pathlib import Path
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, tempfile
try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def julia_command(given):
    if given: return given
    # Resolve Juliaup's selected binary without writing its launcher lock file.
    config = Path.home()/'.julia/juliaup/juliaup.json'
    if config.is_file():
        d = json.loads(config.read_text())
        channel = d.get('InstalledChannels', {}).get(d.get('Default'), {})
        version = d.get('InstalledVersions', {}).get(channel.get('Version'), {})
        binary = version.get('BinaryPath')
        if binary and (config.parent/binary).is_file(): return str((config.parent/binary).resolve())
    command = shutil.which('julia')
    if not command: raise SystemExit('Julia is missing. Install the version stated in environment/README.md or pass --julia.')
    return command

def write_csv(path, rows, fields=None):
    if not rows: return
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)

def export_records(out, plan):
    """Export the same input schemas used by the figure's saved-data analysis."""
    records = []
    for row in plan:
        path = out/row['output_record']
        if path.is_file():
            records.append((row, tomllib.loads(path.read_text())))
    logs, g2, obsrows, spectra, sectors = {}, [], [], [], []
    for row, d in records:
        o = d['observables']; profile = row['profile']
        if profile == 'o2_log':
            if o['status'] != 'ok': continue
            k = float(row['K']); km = o['K_m']
            line = f"fushio χ = 40  c{row['physical_c']} (K = {k:.4f})  step {int(row['rg_step']):2d}  c = {o['central_charge']:.5f}  K_m = {km:.5f}  ({(km-k)/k:+.2e})  cft {o['seconds']:.1f} s"
            logs.setdefault(row['analysis_file'], []).append((int(row['line']), line))
        elif profile == 'o2_g2':
            if o['status'] != 'ok': continue
            values = d['g2']
            g2.append(dict(K=float(row['K']), vertex_c=row['physical_c'], step=int(row['rg_step']),
                           central_charge=o['central_charge'], x_m1=o['x_m1'],
                           **values, checkpoint_sha256=o['checkpoint_sha256'], seconds=o['seconds'], source=o['source']))
        elif profile == 'u1_wide':
            r = {**o, 'result':row['output_record']}
            obsrows.append(r)
            for item in d.get('spectra', []):
                spectra.append(dict(K_target=o['K_target'], vertex_c=o['vertex_c'], rg_step=o['rg_step'],
                                    symmetry=o['symmetry'], tube='wide', **item, measurement_status=o['status']))
            for item in d.get('sectors', []):
                sectors.append(dict(K_target=o['K_target'], vertex_c=o['vertex_c'], rg_step=o['rg_step'], **item))
    for name, lines in logs.items():
        (out/name).write_text('\n'.join(line for _, line in sorted(lines))+'\n')
    if g2:
        fields = ['K','vertex_c','step','central_charge','x_m1']+[f'{p}_{i}' for p in ['even','odd'] for i in range(1,7)]+['checkpoint_sha256','seconds','source']
        write_csv(out/'spectra.csv', g2, fields)
    if obsrows:
        write_csv(out/'observables.csv', obsrows)
        write_csv(out/'spectra.csv', spectra); write_csv(out/'sectors.csv', sectors)
    return len(records)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor-root', type=Path, required=True, help='Local Six_vertex directory matching catalog_path.')
    p.add_argument('--output-dir', type=Path, required=True, help='New CFT directory, outside this figure bundle.')
    select = p.add_mutually_exclusive_group()
    select.add_argument('--record-id', action='append', help='ID from measurement_plan.csv; repeatable.')
    select.add_argument('--all', action='store_true', help='Measure every listed record. Missing inputs stop the run.')
    p.add_argument('--preflight', action='store_true', help='Check all selected inputs and load the Julia environment; no measurement or output writes.')
    p.add_argument('--julia', help='Julia executable; defaults to the installed Juliaup default or PATH.')
    args = p.parse_args()
    if not args.preflight and not (args.all or args.record_id): p.error('Choose --record-id or --all; use --preflight to check inputs first.')
    tensor_root, out = args.tensor_root.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if out == ROOT or ROOT in out.parents or out in ROOT.parents:
        p.error('--output-dir must be outside the source figure bundle and its ancestors.')
    if out == tensor_root or tensor_root in out.parents or out in tensor_root.parents:
        p.error('--output-dir must be separate from the tensor library.')
    plan = list(csv.DictReader((ROOT/'measurement_plan.csv').open()))
    selected = [r for r in plan if not args.record_id or r['record_id'] in args.record_id]
    unknown = set(args.record_id or [])-{r['record_id'] for r in selected}
    if unknown: p.error(f'Unknown record ID: {sorted(unknown)}')
    problems, verified, candidates = [], 0, 0
    hashes = {}
    for r in selected:
        if not r['catalog_path']:
            problems.append(dict(record_id=r['record_id'], issue='no_checkpoint_mapping', cft_record=r['cft_record'])); continue
        path = (tensor_root/r['catalog_path']).resolve()
        if tensor_root not in path.parents:
            problems.append(dict(record_id=r['record_id'], issue='path_outside_tensor_root')); continue
        if not path.is_file():
            problems.append(dict(record_id=r['record_id'], issue='missing_checkpoint', path=str(path))); continue
        if path not in hashes: hashes[path] = sha(path)
        if hashes[path] != r['catalog_sha256']:
            problems.append(dict(record_id=r['record_id'], issue='checkpoint_hash_mismatch', path=str(path))); continue
        verified += 1; candidates += r['match_method'] != 'checkpoint_sha256'
        r['checkpoint'] = str(path)
        if (out/r['output_record']).exists():
            problems.append(dict(record_id=r['record_id'], issue='output_exists', path=str(out/r['output_record'])))
    for r in json.loads((ROOT/'measurement_environment.json').read_text())['files']:
        file = ROOT/r['path']
        if not file.is_file() or sha(file) != r['sha256']:
            problems.append(dict(issue='environment_or_kernel_hash_mismatch', path=r['path']))
    report = dict(selected=len(selected), verified_checkpoints=verified, parameter_only_candidates=candidates,
                  problems=problems, tensor_root=str(tensor_root), output_dir=str(out))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if problems: raise SystemExit(2)
    command = [julia_command(args.julia), '--startup-file=no', '--threads=1', '--project='+str(ROOT/'environment')]
    # Julia loads the local TNRKit environment before any output directory is created.
    subprocess.run(command+['-e','using TNRKit, TensorKit, Serialization, TOML; println("ENVIRONMENT_OK Julia ", VERSION)'], check=True)
    if args.preflight: return
    if out.exists() and any(out.iterdir()):
        p.error('--output-dir must be empty; use a new directory for each invocation.')
    out.mkdir(parents=True, exist_ok=True)
    for r in selected:
        (out/r['output_record']).parent.mkdir(parents=True, exist_ok=True)
    job = '\n'.join(['output_dir = '+json.dumps(str(out))]+[
        '\n[[records]]\n'+'\n'.join(k+' = '+json.dumps(v,ensure_ascii=False) for k,v in row.items()) for row in selected])+'\n'
    (out/'measurement_job.toml').write_text(job)
    with (out/'measurement.log').open('w') as log:
        proc = subprocess.Popen(command+[str(ROOT/'scripts/measure_kernel.jl'),str(out/'measurement_job.toml')],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line,end='',flush=True); log.write(line); log.flush()
        code = proc.wait()
    count = export_records(out, selected)
    statuses = {}
    for row in selected:
        file = out/row['output_record']
        if file.is_file():
            status = tomllib.loads(file.read_text())['observables']['status']
            statuses[status] = statuses.get(status,0)+1
    report.update(completed_records=count, measurement_status_counts=statuses, process_exit_code=code,
                  complete_selected_scope=count==len(selected) and code==0,
                  full_plan_selected=len(selected)==len(plan),
                  note='Complete means all selected attempts were recorded, not that all sectors converged. Reanalysis requires the complete figure dataset; a one-record smoke test cannot regenerate the full figure.')
    (out/'measurement_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if code or count != len(selected): raise SystemExit(code or 1)
    print(f'Wrote {count} CFT record(s) to {out}. No RG was run.')

if __name__ == '__main__': main()
