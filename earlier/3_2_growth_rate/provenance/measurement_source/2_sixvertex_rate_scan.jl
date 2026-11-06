# Q1 of the research plan: does the growth rate of the truncation-seeded perturbation follow
# the prediction 2^(2-K)? Six-vertex model, a = b = 1, O(2) (CU1Irrep) symmetric LoopTNR,
# scanned in the Luttinger parameter K at fixed χ.
#
# With O(2) the only relevant perturbation that is not forbidden is the C-even e = 1
# operator (dimerization, dimension K), relevant for K < 2. K_m - K is second order in its
# coupling, so it grows by 2^(2-K) per step. Conventions: docs/sixvertex/sixvertex_basics.md.
#
# Usage (from Otsuki/):
#   julia --project=tnr tnr/practice/sixvertex_rate_scan.jl compute χ [K ...]   # all K in K_grid if none given
#   julia --project=tnr tnr/practice/sixvertex_rate_scan.jl report χ [χ ...]    # rates, table + figures
#   julia --project=tnr tnr/practice/sixvertex_rate_scan.jl extend χ nto K [K ...]  # RG only, steps nsteps+1..nto
# K values are computed one after another in a single process (compile once); K values whose
# result file already holds all nsteps steps (or records a failure) are skipped, so an
# interrupted scan can be rerun. A failing step (e.g. a singular matrix deep in the massive
# phase) ends that K only; the step and the error are kept in the result's `stopped` field.
#
# Output:
#   data/sixvertex_rate_scan_chiχ_K<K>.jls   c and the lowest dimensions of the needed sectors per step
#   local_tnr_data/Six_vertex/symmetry=CU1Irrep/a=1_b=1/c=<c>/LoopTNR/chi=χ/RGsteps=40_init=true/
#     step_NNN.jls                           the scheme after every step (layout of the shared tensor
#                                            store, README_TNR_DATA.md; not tracked by git)
#   figure/sixvertex/sixvertex_rate_scan_rK.png, figure/sixvertex/sixvertex_rate_scan_dev_chiχ.png
#
# extend: continues the saved run from step nsteps to step nto (RG only; the CFT data are measured
# afterwards with sixvertex_rate_measure.jl). The tensors go to a new folder RGsteps=nto_init=true,
# which also gets copies of steps 0..nsteps; the nsteps folder is left as it is. Used for K close
# to 2, where the deviation grows slowly and does not reach 1% within nsteps steps. The LoopTNR
# source of the TNRKit master is the same as in v0.7.0 (used for steps 1..40); only the dependency
# versions differ. A rerun resumes from the last step in the new folder.
#
# Fit window (fixed before the scan): steps with 1% <= |K_m - K|/K <= 10%, counted from the last
# step at which |K_m - K|/K rises above 1% (earlier crossings belong to the decaying initial
# corrections) up to the last step before it exceeds 10% (beyond that the flow leaves the
# perturbative regime and the ratio drops towards 1). The rate is exp(slope) of a least-squares
# fit of log|K_m - K| against n; at least min_points steps are required.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using TNRKit, TensorKit, Printf, LinearAlgebra, Serialization, Plots

# Same as the pilot and T3; also keeps the runs comparable with each other
BLAS.set_num_threads(1)

const repo = normpath(joinpath(@__DIR__, "..", "..", ".."))
const datadir = joinpath(@__DIR__, "data")
const figdir = joinpath(@__DIR__, "figure", "sixvertex")
const tensorroot = joinpath(repo, "local_tnr_data", "Six_vertex")

const nsteps = 40
const K_grid = vcat(round.(0.60:0.05:1.95; digits = 2), [2.0, 2.25, 2.5, 3.0])
# Sectors and number of levels computed per step (TNRKit's CFTData with the local Nh option):
#   (0, 0): m = 0, C even: identity, the e = 1 even level (K), the three dimension-2 levels
#   (0, 1): m = 0, C odd: J and J̄ (1), the e = 1 odd level (K)
#   (1, 2): m = ±1: lowest 1/(4K) gives K_m, then the descendants
const levels = Dict(CU1Irrep(0, 0) => 6, CU1Irrep(0, 1) => 4, CU1Irrep(1, 2) => 3)
const sector_m1 = string(CU1Irrep(1, 2))
const window = (1e-2, 1e-1)          # range of |K_m - K|/K used in the fit
const min_points = 3

# Vertex weight c for a = b = 1, rounded like the shared c scans (so K = 1.5 gives c = 1 exactly)
vertex_c(K) = round(2 * sin(π / (4K)); digits = 12)
Ktag(K) = @sprintf("%.2f", K)
resultfile(χ, K) = joinpath(datadir, "sixvertex_rate_scan_chi$(χ)_K$(Ktag(K)).jls")
# c as written in the shared catalog (manage.py drops a trailing ".0": c=1, not c=1.0)
cstring(c) = replace(string(c), r"\.0$" => "")
tensordir(χ, K; steps = nsteps) = joinpath(tensorroot, "symmetry=CU1Irrep", "a=1_b=1", "c=$(cstring(vertex_c(K)))",
    "LoopTNR", "chi=$(χ)", "RGsteps=$(steps)_init=true")

# Commit of the repository and whether the local TNRKit or this script differ from it
function code_version()
    git(args...) = readchomp(Cmd(["git", "-C", repo, args...]))
    paths = [joinpath("Otsuki", "tnr", "TNRKit"), joinpath("Otsuki", "tnr", "practice", basename(@__FILE__))]
    return (commit = git("rev-parse", "HEAD"), dirty = !isempty(git("status", "--porcelain", "--", paths...)))
end

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
function compute(χ, K, code)
    c = vertex_c(K)
    scheme = LoopTNR(sixvertex(CU1Irrep; a = 1.0, b = 1.0, c))
    tdir = tensordir(χ, K)
    mkpath(tdir)
    save_tensors(n, history) = serialize(joinpath(tdir, @sprintf("step_%03d.jls", n)),
        (; scheme, normalization_history = copy(history), vertex_c = c, K, chi = χ, rg_step = n,
            symmetry = "CU1Irrep", code))

    # step 0: the normalised initial tensors, as in the shared store
    history = [Float64(finalize!(scheme))]
    save_tensors(0, history)
    steps = NamedTuple{(:n, :c, :dims, :t_rg, :t_cft), Tuple{Int, Float64, Dict{String, Vector{Float64}}, Float64, Float64}}[]
    # Deep in the massive phase (c ≈ 0) the tensors degenerate and the loop optimisation can hit
    # a singular matrix. Such a failure ends this K (recorded in `stopped`), not the whole scan.
    stopped = nothing
    save_result() = serialize(resultfile(χ, K), (; K, vertex_c = c, χ, nsteps,
        levels = Dict(string(k) => v for (k, v) in levels), code, steps, stopped))
    for n in 1:nsteps
        t_rg = t_cft = 0.0
        d = nothing
        try
            t_rg = @elapsed append!(history, run!(scheme, truncrank(χ), maxiter(1); verbosity = 0, finalize_beginning = false))
            save_tensors(n, history)
            t_cft = @elapsed (d = CFTData(scheme; Nh = levels))
        catch err
            err isa InterruptException && rethrow()
            stopped = (n = n, error = sprint(showerror, err))
            save_result()
            @printf "K = %.2f  step %2d  failed, K stopped here: %s\n" K n first(split(stopped.error, '\n'))
            flush(stdout)
            break
        end
        sv = d.scaling_dimensions
        dims = Dict(string(k) => sort(real.(sv[k])) for k in keys(sv.structure))
        push!(steps, (n = n, c = real(d.central_charge), dims = dims, t_rg = t_rg, t_cft = t_cft))
        save_result()
        Δ = get(dims, sector_m1, Float64[])
        Km = isempty(Δ) ? NaN : 1 / (4Δ[1])
        @printf "K = %.2f  step %2d  c = %.5f  K_m = %.5f  (%+.2e)  rg %.1f s, cft %.1f s\n" K n real(d.central_charge) Km (Km - K) / K t_rg t_cft
        flush(stdout)
    end
    return nothing
end

# Complete: all steps computed, or stopped by a failure (rerunning would fail the same way)
function done(χ, K)
    isfile(resultfile(χ, K)) || return false
    res = deserialize(resultfile(χ, K))
    return length(res.steps) == nsteps || (hasproperty(res, :stopped) && res.stopped !== nothing)
end

function compute_all(χ, Ks)
    code = code_version()
    code.dirty && @warn "Local TNRKit or this script has uncommitted changes; the commit alone does not reproduce this run" code.commit
    for K in Ks
        if done(χ, K)
            @printf "K = %.2f: already complete, skipped\n" K
            continue
        end
        compute(χ, K, code)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Extend
# ---------------------------------------------------------------------------
stepfile(dir, n) = joinpath(dir, @sprintf("step_%03d.jls", n))

function extend(χ, K, nto, code)
    src, dst = tensordir(χ, K), tensordir(χ, K; steps = nto)
    mkpath(dst)
    for n in 0:nsteps
        isfile(stepfile(dst, n)) || cp(stepfile(src, n), stepfile(dst, n))
    end
    n0 = maximum(parse(Int, f[6:8]) for f in readdir(dst) if occursin(r"^step_\d{3}\.jls$", f))
    if n0 >= nto
        @printf "K = %.2f: already extended to step %d, skipped\n" K n0
        return nothing
    end
    saved = deserialize(stepfile(dst, n0))
    saved.rg_step == n0 || error("step mismatch in $(dst): file $(n0), saved $(saved.rg_step)")
    scheme = saved.scheme
    history = copy(saved.normalization_history)
    for n in (n0 + 1):nto
        t_rg = 0.0
        try
            t_rg = @elapsed append!(history, run!(scheme, truncrank(χ), maxiter(1); verbosity = 0, finalize_beginning = false))
        catch err
            err isa InterruptException && rethrow()
            @printf "K = %.2f  step %2d  failed, K stopped here: %s\n" K n first(split(sprint(showerror, err), '\n'))
            flush(stdout)
            return nothing
        end
        serialize(stepfile(dst, n), (; scheme, normalization_history = copy(history), vertex_c = vertex_c(K), K,
            chi = χ, rg_step = n, symmetry = "CU1Irrep", code, continued_from = (step = nsteps, dir = src)))
        @printf "K = %.2f  step %2d  rg %.1f s\n" K n t_rg
        flush(stdout)
    end
    return nothing
end

function extend_all(χ, nto, Ks)
    code = code_version()
    code.dirty && @warn "Local TNRKit or this script has uncommitted changes; the commit alone does not reproduce this run" code.commit
    for K in Ks
        extend(χ, K, nto, code)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------
# (K_m - K)/K per step, Dict n => value
function deviation(χ, K)
    res = deserialize(resultfile(χ, K))
    out = Dict{Int, Float64}()
    for s in res.steps
        Δ = get(s.dims, sector_m1, Float64[])
        isempty(Δ) || (out[s.n] = (1 / (4Δ[1]) - K) / K)
    end
    return out
end

# Fit window as described in the header; nothing if the deviation never rises above 1%
function fit_window(dev)
    lo, hi = window
    ns = sort(collect(keys(dev)))
    # A rise needs the previous step below 1%; step 1 (initial corrections) never counts
    rises = [n for n in ns if abs(dev[n]) >= lo && haskey(dev, n - 1) && abs(dev[n - 1]) < lo]
    isempty(rises) && return nothing
    n1 = rises[end]
    n2 = n1
    while haskey(dev, n2 + 1) && lo <= abs(dev[n2 + 1]) <= hi
        n2 += 1
    end
    abs(dev[n1]) <= hi || return nothing
    return (n1, n2)
end

# Least-squares fit of log|x| = α + n log r over steps n1..n2; returns (r, α)
function fit_rate(x, (n1, n2))
    ns = collect(n1:n2)
    α, logr = [ones(length(ns)) ns] \ log.(abs.([x[n] for n in ns]))
    return exp(logr), α
end

# One row per K: window, number of points, rate (NaN when not measured)
function rates(χ)
    rows = NamedTuple[]
    for K in K_grid
        isfile(resultfile(χ, K)) || continue
        dev = deviation(χ, K)
        w = fit_window(dev)
        npts = w === nothing ? 0 : w[2] - w[1] + 1
        r, α = npts >= min_points ? fit_rate(dev, w) : (NaN, NaN)
        push!(rows, (; K, dev, w, npts, r, α))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
function report(χs)
    mkpath(figdir)
    p = plot(; xlabel = "Luttinger parameter K", ylabel = "growth rate r per step", framestyle = :box,
        legend = :topright, title = "O(2) LoopTNR: growth of (K_m - K), fit over 1% ≤ |K_m - K|/K ≤ 10%",
        size = (900, 600), left_margin = 10Plots.mm, bottom_margin = 6Plots.mm)
    Kf = range(0.55, 3.05; length = 200)
    plot!(p, Kf, 2.0 .^ (2 .- Kf); color = :black, label = "prediction 2^(2-K)")
    hline!(p, [1.0]; color = :gray, linestyle = :dash, label = "r = 1")
    for (χ, color) in zip(χs, [:blue, :red, :green])
        rows = rates(χ)
        println("\nχ = $(χ)")
        println("   K     window   points   r        2^(2-K)   r / 2^(2-K)")
        for row in rows
            win = row.w === nothing ? "   -   " : @sprintf("%2d-%2d", row.w...)
            @printf "  %.2f   %7s   %3d    %7.4f   %7.4f   %7.4f\n" row.K win row.npts row.r 2.0^(2 - row.K) row.r / 2.0^(2 - row.K)
        end
        ok = [row for row in rows if !isnan(row.r)]
        scatter!(p, [row.K for row in ok], [row.r for row in ok]; color, markersize = 4,
            markerstrokewidth = 0, label = "χ = $(χ)")
        savefig(deviation_figure(χ, rows), joinpath(figdir, "sixvertex_rate_scan_dev_chi$(χ).png"))
    end
    savefig(p, joinpath(figdir, "sixvertex_rate_scan_rK.png"))
    println("\nfigures saved to ", figdir)
    return nothing
end

# |K_m - K|/K against the step for every K (log scale), fit windows drawn as thick segments
function deviation_figure(χ, rows)
    p = plot(; yscale = :log10, xlabel = "RG step n", ylabel = "|K_m - K| / K", framestyle = :box,
        legend = :outerright, title = "O(2) LoopTNR, χ = $(χ)", size = (1100, 650), ylims = (1e-5, 1),
        left_margin = 10Plots.mm, bottom_margin = 6Plots.mm)
    hline!(p, collect(window); color = :gray, linestyle = :dot, label = "1%, 10%")
    colors = cgrad(:viridis, max(length(rows), 2); categorical = true)
    for (row, color) in zip(rows, colors)
        ns = sort(collect(keys(row.dev)))
        plot!(p, ns, max.(abs.([row.dev[n] for n in ns]), 1e-6); color, linewidth = 1, label = @sprintf("K = %.2f", row.K))
        isnan(row.r) && continue
        nw = collect(row.w[1]:row.w[2])
        plot!(p, nw, exp.(row.α .+ log(row.r) .* nw); color, linewidth = 4, label = "")
    end
    return p
end

# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    mode = ARGS[1]
    if mode == "compute"
        χ = parse(Int, ARGS[2])
        Ks = length(ARGS) > 2 ? parse.(Float64, ARGS[3:end]) : K_grid
        compute_all(χ, Ks)
    elseif mode == "report"
        report(parse.(Int, ARGS[2:end]))
    elseif mode == "extend"
        extend_all(parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse.(Float64, ARGS[4:end]))
    else
        error("mode must be compute, report or extend")
    end
end
