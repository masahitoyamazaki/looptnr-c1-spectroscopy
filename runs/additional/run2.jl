# Follow-up runs B1-B5 (Yamazaki, 2026-11-06); same environment as ../cpu_baseline_20261106.
# Generalizes ../cpu_baseline_20261106/run.jl with options given as environment variables:
#   (SYM = C: charge conjugation only, added for runs B7)
#   STAGGER_H  (default 0)    B1: staggered vertex weight c_A = c_v (1 + h), c_B = c_v (1 - h)
#                              on the two LoopTNR sublattices (U(1), C kept; one-site translation broken)
#   STAGGER_AB (default 0)    B9 (added 2026-11-06): staggered a/b weights, (a, b) = (1 + h, 1 - h) on
#                              sublattice A and (1 - h, 1 + h) on B, c_v unchanged.  Odd under one-site
#                              translation AND under reflection about a vertex (a <-> b), like the
#                              dimerization V_d; STAGGER_H is even under that reflection.
#   STEP_OVERRIDE (optional, C7) "n:chi:sweeps": use bond dimension chi and sweep cap sweeps only for the RG
#                              step that produces step n (all other steps use CHI and SWEEPS)
#   SWEEPS     (default 20)   loop-optimization sweep cap        } B3/B4; the convergence function
#   CONV       (default 1e-9) loop-optimization cost tolerance   } is the one of LoopParameters()
#   EF_RTOL    (default 1e-14) entanglement-filtering relative cutoff
#   FROM       (optional)     checkpoint step_XXX.jls to continue from (B3 branches); its rg_step is kept
# Usage:  julia --project=<env> run2.jl SYM K CHI NSTEPS OUTDIR TAG
# Output: OUTDIR/TAG/{step_XXX.jls, levels.csv, steps.csv} as in run.jl.  Conditions are reserved in
# local_tnr_data/Six_vertex/_requests (owner Yamazaki) before running.
using TNRKit, TensorKit, LinearAlgebra, Printf, Serialization
include(joinpath(@__DIR__, "common.jl"))
using .SymmetryValidation

const SYM, K, CHI, NSTEPS = ARGS[1], parse(Float64, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4])
const OUT = joinpath(ARGS[5], ARGS[6])
const H = parse(Float64, get(ENV, "STAGGER_H", "0"))
const HAB = parse(Float64, get(ENV, "STAGGER_AB", "0"))
H != 0 && HAB != 0 && error("STAGGER_H and STAGGER_AB are exclusive")
const SWEEPS = parse(Int, get(ENV, "SWEEPS", "20"))
const CONV = parse(Float64, get(ENV, "CONV", "1e-9"))
const EF_RTOL = parse(Float64, get(ENV, "EF_RTOL", "1e-14"))
const FROM = get(ENV, "FROM", "")
const OV = let v = get(ENV, "STEP_OVERRIDE", "")
    isempty(v) ? nothing : (x -> (parse(Int, x[1]), parse(Int, x[2]), parse(Int, x[3])))(split(v, ":"))
end
# SYM = CU1 (O(2)), U1, or C (charge conjugation only: Z2-graded dense tensor, both C sectors read out)
const SECTORS = SYM == "CU1" ? [CU1Irrep(0, 0), CU1Irrep(0, 1), CU1Irrep(1, 2), CU1Irrep(2, 2), CU1Irrep(3, 2)] :
                SYM == "U1" ? [U1Irrep(q) for q in (0, 1, -1, 2, -2, 3, -3)] : nothing
const NH = SYM == "C" ? 16 : 8

const DEFAULT = LoopParameters()
loop_parameters(trace = nothing; sweeps = SWEEPS) = begin
    crit = TNRKit.maxiter(sweeps) & TNRKit.convcrit(CONV, DEFAULT.sweeping.crits[2].f)
    LoopParameters(sweeping = trace === nothing ? crit : RecordedSweeping(crit, trace),
                   truncentanglement = trunctol(; rtol = EF_RTOL))
end

isdir(OUT) && !isempty(readdir(OUT)) && error("output folder not empty: $OUT")
mkpath(OUT)
atomic_serialize(path, x) = (serialize(path * ".tmp", x); mv(path * ".tmp", path; force = true))

struct RecordedSweeping{C} <: TNRKit.stopcrit
    base::C
    history::Vector{Tuple{Int, Float64}}
end
(c::RecordedSweeping)(n::Int, cost) = (push!(c.history, (n, Float64(real(last(cost))))); c.base(n, cost))
function advance!(s, n = -1)
    trace = Tuple{Int, Float64}[]
    chi, sw = (OV !== nothing && OV[1] == n) ? (OV[2], OV[3]) : (CHI, SWEEPS)
    p = loop_parameters(trace; sweeps = sw)
    norm = 0.0
    t = @elapsed TNRKit.LoggingExtras.withlevel(; verbosity = 0) do
        TNRKit.step!(s, truncrank(chi), p, 0)
        norm = Float64(finalize!(s))
    end
    all(x -> isfinite(x[2]), trace) || error("nonfinite loop cost")
    return (; seconds = t, normalization = norm,
        sweeps = isempty(trace) ? 0 : last(trace)[1], cost = isempty(trace) ? 0.0 : last(trace)[2])
end

function initial_scheme()
    if !isempty(FROM)
        c = deserialize(FROM)
        return c.scheme, copy(c.normalization_history), c.rg_step
    end
    if HAB != 0
        s = LoopTNR(physical_tensor(SYM, K; a = 1 + HAB, b = 1 - HAB), physical_tensor(SYM, K; a = 1 - HAB, b = 1 + HAB))
    elseif H == 0
        s = LoopTNR(physical_tensor(SYM, K))
    else
        cv = round(2sin(pi / (4K)); digits = 12)
        s = LoopTNR(physical_tensor(SYM, K; vertex_c = cv * (1 + H)), physical_tensor(SYM, K; vertex_c = cv * (1 - H)))
    end
    return s, [Float64(finalize!(s))], 0
end

open(joinpath(OUT, "levels.csv"), "w") do io
    println(io, "step,sector,rank,x,phase,lambda_real,lambda_imag,residual,converged,requested")
end
open(joinpath(OUT, "steps.csv"), "w") do io
    println(io, "step,rg_seconds,sweeps,final_cost,normalization,central_charge,max_residual,status,measure_seconds,dims_A,peak_rss_GB")
end

s, history, n0 = initial_scheme()
@printf "%s K=%.4f chi=%d nsteps=%d h=%g hab=%g sweeps=%d conv=%g ef_rtol=%g from=%s (step %d)  BLAS threads %d\n" SYM K CHI NSTEPS H HAB SWEEPS CONV EF_RTOL (isempty(FROM) ? "-" : FROM) n0 BLAS.get_num_threads()
println("loop_parameters = ", repr(loop_parameters()))
OV === nothing || println("STEP_OVERRIDE: step ", OV[1], " uses chi=", OV[2], " sweeps=", OV[3])
flush(stdout)
lowc = 0
for n in n0:NSTEPS
    global lowc
    rg = n == n0 ? (; seconds = 0.0, normalization = history[end], sweeps = 0, cost = 0.0) : advance!(s, n)
    n > n0 && push!(history, rg.normalization)
    atomic_serialize(joinpath(OUT, @sprintf("step_%03d.jls", n)),
        (; scheme = s, normalization_history = copy(history), symmetry = SYM, K, chi = CHI, rg_step = n,
           vertex_c = round(2sin(pi / (4K)); digits = 12), stagger_h = H, stagger_ab = HAB, from = FROM,
           loop_parameters = repr(loop_parameters()), code = "Yamazaki/runs/followup_B_20261106/run2.jl"))
    m = nothing
    tm = @elapsed try
        m = measure_loop(s; nh = NH, selected = SECTORS)
    catch err
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
    @printf "step %3d  rg %7.1f s  meas %6.1f s  sweeps %3d  cost %.1e  c %.5f  %s\n" n rg.seconds tm rg.sweeps rg.cost m["central_charge"] status
    flush(stdout)
    lowc = abs(m["central_charge"] - 1) > 0.9 ? lowc + 1 : 0
    lowc >= 3 && (println("|c - 1| > 0.9 for 3 steps: stop"); break)
end
println("done")
