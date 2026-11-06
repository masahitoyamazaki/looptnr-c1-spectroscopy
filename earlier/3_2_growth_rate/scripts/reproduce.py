from run_bundle import prepare_run
if prepare_run():
    raise SystemExit(0)

"""Reproduce this figure from bundled CFT records; no RG or eigensolver is run."""
import sys,os,tempfile
from pathlib import Path
# Use a writable cache without depending on the caller's home directory.
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"paper_figure_mpl"))
os.environ.setdefault("XDG_CACHE_HOME",str(Path(tempfile.gettempdir())/"paper_figure_cache"))
sys.dont_write_bytecode=True
from bundle import verify_inputs
verify_inputs()
# Refit from the full-precision levels (o2_figure.py); skipped when data/cft has no o2_levels.csv
from o2_figure import main as main_o2
main_o2()
from analyze import main
main()
