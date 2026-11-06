# Unperturbed six-vertex LoopTNR baseline with a single public code version (Yamazaki, 2026-11-06).
# Registered TNRKit 0.7.0 (environment/ in this folder), default LoopParameters(),
# Cho's physical_tensor / measure_loop (Cho/scripts/symmetry_validation/common.jl, unchanged;
# fixed tau = i/2 readout of the [sqrt2, 2sqrt2, 0] tube; eigenvalue phases are kept).
#
# Usage:  julia --project=<env> run.jl SYM K CHI NSTEPS OUTDIR
#   SYM = CU1 (O(2)) or U1;  the condition must be reserved in local_tnr_data/Six_vertex/_requests
#   (owner Yamazaki) before running.
# Output in OUTDIR/<SYM>_K<K>_chi<CHI>/:
#   step_XXX.jls   checkpoint (scheme, normalization history, metadata)
#   levels.csv     step, sector, rank, x, phase, lambda_real, lambda_imag, residual, converged
#   steps.csv      step, rg_seconds, sweeps, final_cost, normalization, central_charge, status, ...
using TNRKit, TensorKit, LinearAlgebra, Printf, Serialization
include(joinpath(@__DIR__, "common.jl"))
using .SymmetryValidation

const SYM = ARGS[1]
const K = parse(Float64, ARGS[2])
const CHI = parse(Int, ARGS[3])
const NSTEPS = parse(Int, ARGS[4])
const OUT = joinpath(ARGS[5], @sprintf("%s_K%.4f_chi%d", SYM, K, CHI))
const SECTORS = SYM == "CU1" ?
    [CU1Irrep(0, 0), CU1Irrep(0, 1), CU1Irrep(1, 2), CU1Irrep(2, 2), CU1Irrep(3, 2)] :
    [U1Irrep(q) for q in (0, 1, -1, 2, -2, 3, -3)]
const NH = 8

isdir(OUT) && !isempty(readdir(OUT)) && error("output folder not empty: $OUT")
mkpath(OUT)
atomic_serialize(path, x) = (serialize(path * ".tmp", x); mv(path * ".tmp", path; force = true))

# One standard LoopTNR step with the sweep count and cost recorded (as in Cho's core.jl).
struct RecordedSweeping{C} <: TNRKit.stopcrit
    base::C
    history::Vector{Tuple{Int, Float64}}
end
(c::RecordedSweeping)(n::Int, cost) = (push!(c.history, (n, Float64(real(last(cost))))); c.base(n, cost))
function advance!(s)
    trace = Tuple{Int, Float64}[]
    p = LoopParameters(sweeping = RecordedSweeping(LoopParameters().sweeping, trace))
    norm = 0.0
    t = @elapsed TNRKit.LoggingExtras.withlevel(; verbosity = 0) do
        TNRKit.step!(s, truncrank(CHI), p, 0)
        norm = Float64(finalize!(s))
    end
    all(x -> isfinite(x[2]), trace) || error("nonfinite loop cost")
    return (; seconds = t, normalization = norm,
        sweeps = isempty(trace) ? 0 : last(trace)[1], cost = isempty(trace) ? 0.0 : last(trace)[2])
end

open(joinpath(OUT, "levels.csv"), "w") do io
    println(io, "step,sector,rank,x,phase,lambda_real,lambda_imag,residual,converged,requested")
end
open(joinpath(OUT, "steps.csv"), "w") do io
    println(io, "step,rg_seconds,sweeps,final_cost,normalization,central_charge,max_residual,status,measure_seconds,dims_A,peak_rss_GB")
end

@printf "%s K=%.4f chi=%d nsteps=%d  Julia threads %d  BLAS threads %d\n" SYM K CHI NSTEPS Threads.nthreads() BLAS.get_num_threads()
flush(stdout)
s = LoopTNR(physical_tensor(SYM, K))
history = [Float64(finalize!(s))]
lowc = 0
for n in 0:NSTEPS
    global lowc
    rg = n == 0 ? (; seconds = 0.0, normalization = history[1], sweeps = 0, cost = 0.0) : advance!(s)
    n > 0 && push!(history, rg.normalization)
    atomic_serialize(joinpath(OUT, @sprintf("step_%03d.jls", n)),
        (; scheme = s, normalization_history = copy(history), symmetry = SYM, K, chi = CHI, rg_step = n,
           vertex_c = round(2sin(pi / (4K)); digits = 12), loop_parameters = repr(LoopParameters()),
           code = "Yamazaki/runs/cpu_baseline_20261106/run.jl"))
    m = nothing
    tm = @elapsed try
        m = measure_loop(s; nh = NH, selected = SECTORS)
    catch err
        # After the RG has broken down, a sector can vanish and the readout fails; stop cleanly.
        @printf "step %3d  readout failed (%s): stop\n" n sprint(showerror, err)[1:min(end, 120)]
    end
    m === nothing && break
    res = maximum(r["residual"] for r in m["spectra"])
    status = any(r["converged"] < r["requested"] for r in m["spectra"]) ? "partial_convergence" :
             res > 1e-8 ? "residual_failure" : "ok"
    open(joinpath(OUT, "levels.csv"), "a") do io
        for r in m["spectra"]
            @printf(io, "%d,\"%s\",%d,%.16e,%.16e,%.16e,%.16e,%.3e,%d,%d\n", n, r["sector"], r["rank"], r["x"],
                r["phase"], r["lambda_real"], r["lambda_imag"], r["residual"], r["converged"], r["requested"])
        end
    end
    dimsA = join([dim(space(s.TA, i)) for i in 1:4], ";")
    open(joinpath(OUT, "steps.csv"), "a") do io
        @printf(io, "%d,%.2f,%d,%.3e,%.16e,%.12f,%.3e,%s,%.2f,%s,%.2f\n", n, rg.seconds, rg.sweeps, rg.cost,
            rg.normalization, m["central_charge"], res, status, tm, dimsA, Sys.maxrss() / 2^30)
    end
    @printf "step %3d  rg %7.1f s  meas %6.1f s  sweeps %2d  cost %.1e  c %.5f  %s\n" n rg.seconds tm rg.sweeps rg.cost m["central_charge"] status
    flush(stdout)
    lowc = abs(m["central_charge"] - 1) > 0.9 ? lowc + 1 : 0
    lowc >= 3 && (println("|c - 1| > 0.9 for 3 steps: stop"); break)
end
println("done")
