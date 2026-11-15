# Mid-RG compensation (K. Cho's projected-gradient correction, v2 directions P3_3c / P3_4c / P3_5c),
# rerun in one environment for chi = 24, 32, 40 so that the chi dependence and the cost can be compared
# with the uncompensated main runs on the same machine.
#
# Usage:
#   julia --project=<env> run.jl K chi dir interval start_step max_step outdir
#     K          Luttinger parameter (0.5 = BKT point, run with vertex_c = 2)
#     dir        P3_3c | P3_4c | P3_5c | none   (none: uncompensated reference with the same diagnostics)
#     interval   correction every `interval` steps from start_step on
#     start_step first correction (the run is generated from scratch up to this step)
#     max_step   last step
#
# Differences from Cho's run_phase3_deep_bkt_and_p3_5c.jl (2d521d2b): chi is an argument, the state is
# generated from scratch at that chi, the state is evaluated at EVERY step (before the correction), the
# correction is evaluated after it as well, the triplet levels are stored individually, and the wall-clock
# times of the RG step, the correction, and the evaluation are recorded separately.  The helper files in
# cho_src/ are verbatim copies of Cho/scripts/midrg_compensation_20261112 at 2d521d2b.
include("cho_src/common.jl")
include("cho_src/bond.jl")
include("cho_src/p3_directions.jl")
using .SymmetryValidation: physical_tensor, measure_loop, atomic_toml
using Serialization, Printf, SHA, TOML, LinearAlgebra

const SELECTED = [CU1Irrep(0,0), CU1Irrep(0,1), CU1Irrep(1,2), CU1Irrep(2,2), CU1Irrep(3,2)]

function loop_params()
    default = LoopParameters()
    crit = TNRKit.maxiter(20) & TNRKit.convcrit(1e-9, default.sweeping.crits[2].f)
    return LoopParameters(sweeping = crit, truncentanglement = trunctol(; rtol = 1e-14))
end

function compute_direction_with_pre(s::LoopTNR, dir_key::Symbol)
    A, B = s.TA, s.TB
    ge, re = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=true, rotated=false)
    go, ro = log_eigen_gradient_2x2(A, B, CU1Irrep(0,1); flux=true, rotated=false)
    gv, _  = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=false, rotated=false)
    delta_pi_pre = log(re["abs_lambda"] / ro["abs_lambda"]) / (2pi)
    g_delta = pairscale(1.0 / (2pi), pairsub(ge, go))
    g_c = pairscale(1.0 / (4pi), pairsub(pairscale(2.0, gv), pairadd(ge, go)))
    cA = (A, zero(B)); cB = (zero(A), B); ref_pair = (A, B)
    if dir_key == :P3_3c
        D, _ = project_direction(g_delta, [cA, cB, g_c], ref_pair)
        return D, pairdot(g_delta, D), delta_pi_pre
    end
    gb, _ = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=false, rotated=true)
    g_aniso = pairsub(gb, gv)
    if dir_key == :P3_4c
        D, _ = project_direction(g_delta, [cA, cB, g_c, g_aniso], ref_pair)
        return D, pairdot(g_delta, D), delta_pi_pre
    end
    gd_vac, _ = log_eigen_gradient_diag(A, B, CU1Irrep(0,0); rank=1, rotated=false)
    g01_raw, _ = log_eigen_gradient_diag(A, B, CU1Irrep(1,2); rank=1, rotated=false)
    g_x01 = pairscale(1.0 / pi, pairsub(gd_vac, g01_raw))
    D, _ = project_direction(g_delta, [cA, cB, g_c, g_aniso, g_x01], ref_pair)
    return D, pairdot(g_delta, D), delta_pi_pre
end

# all levels are stored; the assignment below follows Cho's identify_q01_levels
function evaluate_state(s::LoopTNR, K::Real)
    even = tube_sector(s, CU1Irrep(0,0); flux=true, nlevels=2)
    odd  = tube_sector(s, CU1Irrep(0,1); flux=true, nlevels=2)
    vac0 = tube_sector(s, CU1Irrep(0,0); flux=false, nlevels=1)
    s_rot = rotate90(s)
    vac90 = tube_sector(s_rot, CU1Irrep(0,0); flux=false, nlevels=1)
    la = abs(vac0[1].lambda); lb = abs(vac90[1].lambda)
    le = abs(even[1].lambda); lo = abs(odd[1].lambda)
    m0  = measure_loop(s; nh=5, selected=SELECTED)
    lev(q) = sort([r["x"] for r in m0["spectra"] if r["sector"] == string(q)])
    q00, q01, q12, q22, q32 = lev(CU1Irrep(0,0)), lev(CU1Irrep(0,1)), lev(CU1Irrep(1,2)), lev(CU1Irrep(2,2)), lev(CU1Irrep(3,2))
    x01 = q12[1]
    eplus = length(q00) >= 2 ? q00[2] : NaN
    # lowest C-odd neutral levels: for K < 0.8 the C-odd e=+-1 state is the lowest, for K >= 1.1 the current pair is
    if K >= 1.1
        j1, j2, eminus = get(q01, 1, NaN), get(q01, 2, NaN), get(q01, 3, NaN)
    elseif K >= 0.8
        pairs = ((1, 2, 3), (1, 3, 2), (2, 3, 1))
        a, b, c = pairs[argmin([abs(q01[i] - q01[j]) for (i, j, _) in pairs])]
        j1, j2, eminus = q01[a], q01[b], q01[c]
    else
        eminus, j1, j2 = get(q01, 1, NaN), get(q01, 2, NaN), get(q01, 3, NaN)
    end
    xe = (eplus + eminus) / 2
    return Dict{String,Any}(
        "delta_pi" => log(le / lo) / (2pi), "xbar_pi" => (2log(la) - log(le) - log(lo)) / (4pi),
        "aniso_2x2" => log(lb / la), "c_raw" => m0["central_charge"],
        "X" => 1 - 4K * x01, "x01" => x01, "x02" => get(q22, 1, NaN), "x03" => get(q32, 1, NaN),
        "eplus" => eplus, "eminus" => eminus, "xJ1" => j1, "xJ2" => j2,
        "Kn" => 0.5 * sqrt(xe / x01), "scale" => 2sqrt(x01 * xe),
        "q00" => q00[1:min(5, end)], "q01" => q01[1:min(5, end)], "q12" => q12[1:min(5, end)],
        "q22" => q22[1:min(3, end)], "q32" => q32[1:min(3, end)])
end

function main()
    K = parse(Float64, ARGS[1]); chi = parse(Int, ARGS[2]); dir = Symbol(ARGS[3])
    interval = parse(Int, ARGS[4]); start_step = parse(Int, ARGS[5]); max_step = parse(Int, ARGS[6])
    outdir = ARGS[7]
    BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "4")))
    tag = @sprintf("K%.2f_chi%d_%s_int%d", K, chi, dir, interval)
    out = joinpath(outdir, tag * ".toml")
    T = K == 0.5 ? physical_tensor("CU1", 0.5; vertex_c = 2.0) : physical_tensor("CU1", K)
    s = LoopTNR(T); finalize!(s)
    rec = Dict{String,Any}("K" => K, "chi" => chi, "direction" => string(dir), "interval" => interval,
        "start_step" => start_step, "max_step" => max_step,
        "julia" => string(VERSION), "tnrkit" => string(pkgversion(TNRKit)), "tensorkit" => string(pkgversion(TensorKit)),
        "blas_threads" => BLAS.get_num_threads(), "host" => "workstation",
        "source_sha256" => Dict(f => bytes2hex(sha256(read(joinpath(@__DIR__, f))))
                                for f in ("run.jl", "cho_src/common.jl", "cho_src/bond.jl", "cho_src/p3_directions.jl", "cho_src/twist_measure.jl")))
    steps = Dict{String,Any}[]
    n_e = -1; status = "max_step"
    for n in 0:max_step
        t_rg = 0.0
        if n > 0
            t_rg = @elapsed TNRKit.LoggingExtras.withlevel(; verbosity=0) do
                TNRKit.step!(s, truncrank(chi), loop_params(), 0)
            end
            finalize!(s)
        end
        n < 4 && continue
        try
            t_ev = @elapsed (f = evaluate_state(s, K))
            f["step"] = n; f["rg_seconds"] = t_rg; f["eval_seconds"] = t_ev
            if abs(f["X"]) <= 0.1; n_e = n; end
            if dir != :none && n >= start_step && (n - start_step) % interval == 0
                t_fb = @elapsed begin
                    D, kappa, dpre = compute_direction_with_pre(s, dir)
                    alpha = clamp(-dpre / kappa, -0.05, 0.05)
                    s = apply_direction(s, D, alpha; do_finalize = true)
                end
                g = evaluate_state(s, K)
                f["corrected"] = true; f["feedback_seconds"] = t_fb; f["alpha"] = alpha; f["kappa"] = kappa
                f["post"] = g
            end
            push!(steps, f)
            @printf("%s n=%3d X=%+.2e x01=%.5f e-=%.5f e+=%.5f Kn=%.5f dpi=%+.2e aniso=%+.1e c=%.5f  t_rg=%.1fs t_fb=%.1fs\n",
                    tag, n, f["X"], f["x01"], f["eminus"], f["eplus"], f["Kn"], f["delta_pi"], f["aniso_2x2"], f["c_raw"],
                    t_rg, get(f, "feedback_seconds", 0.0))
            flush(stdout)
            if !isfinite(f["c_raw"]) || f["c_raw"] < 0.5 || abs(f["X"]) > 0.25
                status = "stopped_threshold"; break
            end
        catch e
            rec["error"] = sprint(showerror, e); status = "stopped_on_error"
            @printf("%s n=%3d ERROR %s\n", tag, n, rec["error"]); flush(stdout); break
        end
        if n % 4 == 0
            rec["steps"] = steps; rec["n_e"] = n_e; rec["status"] = "running"; atomic_toml(out, rec)
        end
    end
    rec["steps"] = steps; rec["n_e"] = n_e; rec["status"] = status
    atomic_toml(out, rec)
    println(tag, " done: status=", status, " n_e=", n_e)
end

main()
