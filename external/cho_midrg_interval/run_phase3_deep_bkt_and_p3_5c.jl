# Phase 3: Full P3_5c multi-step (K=1.5, K=1.0) & Deep BKT K=0.5 (RG12 -> RG64, interval=2 vs 4)
# Single-process sequential runner (--heap-size-hint=3200M) with fast pre-delta extraction.
include("common.jl")
include("bond.jl")
include("p3_directions.jl")
using .SymmetryValidation: physical_tensor, measure_loop, atomic_toml, provenance, ROOT
using Serialization, Printf, SHA, Random, TOML

const SELECTED = [CU1Irrep(0,0), CU1Irrep(0,1), CU1Irrep(1,2), CU1Irrep(2,2), CU1Irrep(3,2)]
const CHI_REF = 24

function loop_params(trace=nothing)
    default = LoopParameters()
    f = trace === nothing ? default.sweeping.crits[2].f :
        (steps::Int, data) -> (push!(trace, (steps, data[end])); default.sweeping.crits[2].f(steps, data))
    crit = TNRKit.maxiter(20) & TNRKit.convcrit(1e-9, f)
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
    cA = (A, zero(B))
    cB = (zero(A), B)
    ref_pair = (A, B)
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

function identify_q01_levels(q01::Vector{Float64}, K_target::Real)
    if K_target >= 1.1
        j1 = length(q01) >= 1 ? q01[1] : NaN
        j2 = length(q01) >= 2 ? q01[2] : NaN
        eminus = length(q01) >= 3 ? q01[3] : NaN
        return j1, j2, eminus
    elseif K_target >= 0.8 && length(q01) >= 3
        # At K=1.0, two current states J and C-odd e_- are near x=1; pair the closest two as J1, J2.
        pairs = ((1, 2, 3), (1, 3, 2), (2, 3, 1))
        best_i, best_j, rem_k = pairs[argmin([abs(q01[a] - q01[b]) for (a, b, _) in pairs])]
        return q01[best_i], q01[best_j], q01[rem_k]
    else
        eminus = length(q01) >= 1 ? q01[1] : NaN
        j1 = length(q01) >= 2 ? q01[2] : NaN
        j2 = length(q01) >= 3 ? q01[3] : NaN
        return j1, j2, eminus
    end
end

function evaluate_state(s::LoopTNR, K_target::Real)
    even = tube_sector(s, CU1Irrep(0,0); flux=true, nlevels=2)
    odd  = tube_sector(s, CU1Irrep(0,1); flux=true, nlevels=2)
    vac0 = tube_sector(s, CU1Irrep(0,0); flux=false, nlevels=1)
    s_rot = rotate90(s)
    vac90 = tube_sector(s_rot, CU1Irrep(0,0); flux=false, nlevels=1)

    la = abs(vac0[1].lambda)
    lb = abs(vac90[1].lambda)
    le = abs(even[1].lambda)
    lo = abs(odd[1].lambda)

    delta_pi = log(le / lo) / (2pi)
    xbar_pi  = (2*log(la) - log(le) - log(lo)) / (4pi)
    aniso_2x2 = log(lb / la)

    m0  = measure_loop(s; nh=5, selected=SELECTED)
    m90 = measure_loop(s_rot; nh=1, selected=[CU1Irrep(0,0)])
    log_ld_over_la  = -pi / 12.0 * m0["central_charge"]
    log_ldp_over_lb = -pi / 12.0 * m90["central_charge"]
    log_ldp_over_la = log_ldp_over_lb + aniso_2x2
    aniso_diag = log_ldp_over_la - log_ld_over_la

    metric = fit_4vacuum_metric(aniso_2x2, log_ld_over_la, log_ldp_over_la)

    level_list(m, q) = sort([r["x"] for r in m["spectra"] if r["sector"] == string(q)])
    q00 = level_list(m0, CU1Irrep(0,0))
    q01 = level_list(m0, CU1Irrep(0,1))
    q12 = level_list(m0, CU1Irrep(1,2))
    q22 = level_list(m0, CU1Irrep(2,2))
    q32 = level_list(m0, CU1Irrep(3,2))

    x01 = q12[1]
    x02 = isempty(q22) ? NaN : q22[1]
    x03 = isempty(q32) ? NaN : q32[1]
    eplus = length(q00) >= 2 ? q00[2] : NaN
    j1, j2, eminus = identify_q01_levels(q01, K_target)

    xe = (eplus + eminus) / 2.0
    xJ_mean = (j1 + j2) / 2.0
    xJ_split = abs(j2 - j1)
    kn_em = 0.5 * sqrt(xe / x01)
    kn_e  = xe / xJ_mean
    kn_m  = xJ_mean / (4.0 * x01)
    scale = 2.0 * sqrt(x01 * xe)
    eps_q = max(abs(x02 / (4.0 * x01) - 1.0), abs(x03 / (9.0 * x01) - 1.0))

    x_triplet   = (2.0 * x01 + eminus) / 3.0
    x_triplet_2 = 0.5 * (x01 + eminus)
    x_singlet   = eplus

    return Dict(
        "delta_pi" => delta_pi, "xbar_pi" => xbar_pi,
        "aniso_2x2" => aniso_2x2, "aniso_diag" => aniso_diag,
        "c_raw" => m0["central_charge"], "c_raw_90" => m90["central_charge"],
        "c_metric" => metric["c_metric"], "tau0_real" => metric["tau0_real"], "tau0_imag" => metric["tau0_imag"],
        "X" => 1.0 - 4.0 * K_target * x01, "x01" => x01, "x02" => x02, "x03" => x03,
        "eplus" => eplus, "eminus" => eminus, "xe" => xe,
        "xJ_mean" => xJ_mean, "xJ_split" => xJ_split,
        "Kn" => kn_em, "Ke" => kn_e, "Km" => kn_m,
        "scale" => scale, "eps_q" => eps_q,
        "x_triplet" => x_triplet, "x_triplet_2" => x_triplet_2, "x_singlet" => x_singlet,
        "q01_top3" => q01[1:min(3, length(q01))],
        "q00_top3" => q00[1:min(3, length(q00))]
    )
end

function feedback_step_fast!(s::LoopTNR, dir_key::Symbol, K_target::Real; max_rel_update::Float64=0.05)
    D, kappa, delta_pi_pre = compute_direction_with_pre(s, dir_key)
    raw_alpha = -delta_pi_pre / kappa
    alpha = clamp(raw_alpha, -max_rel_update, max_rel_update)
    s_after = apply_direction(s, D, alpha; do_finalize=true)
    f_after = evaluate_state(s_after, K_target)
    return s_after, delta_pi_pre, kappa, f_after, alpha
end

function run_feedback_loop(s_start::LoopTNR, start_step::Int, max_step::Int, K_target::Real;
                           dir_key::Symbol, interval::Int=4, first_intervention_step::Int=start_step)
    s = deepcopy(s_start)
    finalize!(s)
    traj = Dict{String,Any}[]
    n_e = max_step
    status = "completed"
    err_msg = ""

    for n in start_step:max_step
        try
            sweep_trace = Tuple{Int,Float64}[]
            sec = 0.0
            if n > start_step
                p_step = loop_params(sweep_trace)
                sec = @elapsed TNRKit.LoggingExtras.withlevel(; verbosity=0) do
                    TNRKit.step!(s, truncrank(CHI_REF), p_step, 0)
                end
                finalize!(s)
            end
            if n >= first_intervention_step && (n - first_intervention_step) % interval == 0
                s, delta_pre, kappa, f_post, alpha_tot = feedback_step_fast!(s, dir_key, K_target)
                f_post["step"] = n
                f_post["rg_seconds"] = sec
                f_post["sweeps"] = isempty(sweep_trace) ? 0 : sweep_trace[end][1]
                f_post["cost_end"] = isempty(sweep_trace) ? 0.0 : sweep_trace[end][2]
                f_post["alpha_applied"] = alpha_tot
                f_post["kappa_delta"] = kappa
                f_post["delta_pi_pre"] = delta_pre
                push!(traj, f_post)
                @printf("  [K=%.2f, %s, int=%d, n=%3d]: delta_pre=%+.5f -> post=%+.2e | upd=%+.4e | Kn=%.5f, x01=%.5f, x_t2=%.5f, x_s=%.5f, eps_q=%.5f, aniso=%+.5f, c_raw=%.5f\n",
                        K_target, dir_key, interval, n, delta_pre, f_post["delta_pi"], alpha_tot,
                        f_post["Kn"], f_post["x01"], f_post["x_triplet_2"], f_post["x_singlet"],
                        f_post["eps_q"], f_post["aniso_2x2"], f_post["c_raw"])
                flush(stdout)
                if !isfinite(f_post["c_raw"]) || f_post["c_raw"] < 0.5 || abs(f_post["X"]) > 0.25
                    n_e = n - 1
                    status = "stopped_threshold"
                    @printf("  --> Stopped early at n=%d (c_raw=%.4f, X=%.4f)\n", n, f_post["c_raw"], f_post["X"])
                    flush(stdout)
                    break
                end
            end
        catch e
            n_e = n - 1
            status = "stopped_on_error"
            err_msg = sprint(showerror, e)
            @printf("  --> Caught error at n=%d: %s\n", n, err_msg)
            flush(stdout)
            break
        end
    end
    GC.gc(false)
    return Dict("trajectory" => traj, "n_e" => n_e, "interval" => interval,
                "direction" => string(dir_key), "status" => status, "error" => err_msg)
end

function main()
    BLAS.set_num_threads(1)
    out_path = joinpath(ROOT, "Cho/output/midrg_compensation_20261112/phase3_deep_bkt_and_p3_5c.toml")
    prev = isfile(out_path) ? TOML.parsefile(out_path) : Dict{String,Any}()
    results = Dict{String,Any}("provenance" => provenance())

    # 1. K=1.5 Full Observable Comparison: P3_3c vs P3_4c vs P3_5c (RG24 -> RG64)
    if haskey(prev, "K1p5_full") && all(haskey(prev["K1p5_full"], string(d)) for d in (:P3_3c, :P3_4c, :P3_5c))
        println("=== 1. K=1.5 Full Observable Comparison: loaded saved K1p5_full from TOML ===")
        flush(stdout)
        results["K1p5_full"] = prev["K1p5_full"]
    else
        println("=== 1. K=1.5 Full Observable Comparison: P3_3c vs P3_4c vs P3_5c (RG24 -> RG64) ===")
        flush(stdout)
        s24_k15 = (x -> x isa LoopTNR ? x : x.scheme)(deserialize(joinpath(ROOT, "Cho/output/midrg_compensation_20261112/step_024.jls")))
        k15_full = Dict{String,Any}()
        for dkey in (:P3_3c, :P3_4c, :P3_5c)
            println("--- K=1.5 Multi-step: ", dkey, " (RG24 -> RG64) ---")
            flush(stdout)
            k15_full[string(dkey)] = run_feedback_loop(s24_k15, 24, 64, 1.5; dir_key=dkey, interval=4, first_intervention_step=24)
        end
        results["K1p5_full"] = k15_full
        atomic_toml(out_path, results)
    end

    # 2. K=1.0 P3_5c Multi-Step (RG16 -> RG40)
    println("\n=== 2. K=1.0 P3_5c Multi-Step (RG16 -> RG40) ===")
    flush(stdout)
    T_k10 = physical_tensor("CU1", 1.0)
    s16_k10 = LoopTNR(T_k10); finalize!(s16_k10)
    traj_k10_init = Dict{String,Any}[]
    for n in 0:16
        if n > 0
            TNRKit.LoggingExtras.withlevel(; verbosity=0) do
                TNRKit.step!(s16_k10, truncrank(CHI_REF), loop_params(), 0)
            end
            finalize!(s16_k10)
        end
        if n % 4 == 0 && n >= 4
            f = evaluate_state(s16_k10, 1.0); f["step"] = n; push!(traj_k10_init, f)
        end
    end
    GC.gc(false)
    k10_p35c = Dict{String,Any}("init_0_to_16" => traj_k10_init,
        "P3_5c" => run_feedback_loop(s16_k10, 16, 40, 1.0; dir_key=:P3_5c, interval=4, first_intervention_step=16))
    results["K1p0_p3_5c"] = k10_p35c
    atomic_toml(out_path, results)

    # 3. Deep BKT K=0.5 (c_v=2.0) (RG12 -> RG64), interval=4 and interval=2
    println("\n=== 3. Deep BKT K=0.5 (c=2.0) Extension (RG12 -> RG64): P3_4c, P3_3c, P3_5c (int4 & int2) ===")
    flush(stdout)
    T_bkt = physical_tensor("CU1", 0.5; vertex_c=2.0)
    s12_bkt = LoopTNR(T_bkt); finalize!(s12_bkt)
    traj_bkt_init = Dict{String,Any}[]
    for n in 0:12
        if n > 0
            TNRKit.LoggingExtras.withlevel(; verbosity=0) do
                TNRKit.step!(s12_bkt, truncrank(CHI_REF), loop_params(), 0)
            end
            finalize!(s12_bkt)
        end
        if n >= 4
            f = evaluate_state(s12_bkt, 0.5); f["step"] = n; push!(traj_bkt_init, f)
            if n % 4 == 0
                @printf("  [BKT init, n=%2d]: delta=%+.5f, x01=%.5f, x_t2=%.5f, x_t3=%.5f, x_s=%.5f, Kn=%.5f\n",
                        n, f["delta_pi"], f["x01"], f["x_triplet_2"], f["x_triplet"], f["x_singlet"], f["Kn"])
                flush(stdout)
            end
        end
    end
    GC.gc(false)

    bkt_deep = Dict{String,Any}("init_4_to_12" => traj_bkt_init)
    bkt_specs = [
        ("P3_4c_int4", :P3_4c, 4),
        ("P3_4c_int2", :P3_4c, 2),
        ("P3_3c_int4", :P3_3c, 4),
        ("P3_3c_int2", :P3_3c, 2),
        ("P3_5c_int4", :P3_5c, 4),
        ("P3_5c_int2", :P3_5c, 2),
    ]
    for (label, dkey, intv) in bkt_specs
        println("--- Deep BKT: ", label, " (RG12 -> RG64) ---")
        flush(stdout)
        bkt_deep[label] = run_feedback_loop(s12_bkt, 12, 64, 0.5; dir_key=dkey, interval=intv, first_intervention_step=12)
        results["BKT_K0p5_deep"] = bkt_deep
        atomic_toml(out_path, results)
    end

    println("Saved complete Phase 3 TOML to ", out_path)
    flush(stdout)
end

main()

