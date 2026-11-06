# Measure CFT data (same sectors and levels as sixvertex_rate_scan.jl) on saved six-vertex
# LoopTNR tensors, without new RG. Two sources, both O(2) (CU1Irrep), a = b = 1:
#   otsuki: this workspace's Q1 scan (χ = 16, 24; 40 steps, χ = 24 at K = 1.55-1.95 extended to 70), labelled by K
#   fushio: Fushio's shared runs (χ = 28, 40; 70 steps), labelled by the weight c
# Both are read from local_tensor_data/Six_vertex/ (the fushio ones fetched with `manage.py sync`).
# Used to put every χ on the same CFT code (the TNRKit master in tnr/TNRKit, which fits the
# modular parameter τ0 of one tensor before reading the dimensions).
#
# Usage (from Otsuki/):
#   julia --project=tnr tnr/practice/sixvertex_rate_measure.jl otsuki χ [K ...]   # default: all of K_grid
#   julia --project=tnr tnr/practice/sixvertex_rate_measure.jl fushio χ [c ...]   # default: 0.5 1 1.4 1.7 1.9
#   julia --project=tnr tnr/practice/sixvertex_rate_measure.jl report             # rates from K_m and P, figures
#
# report: besides K_m (m = 1 level) it reads K_e = Δ(1,0) (the C-even e = 1 level: 2nd level of
# (0, 0) for K < 2, the level closest to K otherwise) and P = 4 Δ(1,0) Δ(0,1) - 1 ≈ (K_e - K_m)/K.
# P vanishes for any radius K, so a drift of K itself drops out of it; the growing dimerization
# shifts it at second order like K_m. Both are fitted with the scan's rule (fit_window, 1%-10%).
# Figures: figure/sixvertex/sixvertex_rate_example.png, sixvertex_rate_controls.png, sixvertex_rate_ratio.png,
#          sixvertex_rate_pkm.png
# Output: data/sixvertex_rate_cft_otsuki_chiχ_K<K>.jls, data/sixvertex_rate_cft_fushio_chiχ_c<c>.jls,
#         same layout as the scan's result files (steps with c and dims), plus the source folder
#         and the modular parameter per step. Steps whose CFT extraction fails are listed in
#         `failed`. Measuring stops once c < massive_c for massive_steps steps (recorded in
#         `stopped`). A rerun resumes from the steps already in the file.

include(joinpath(@__DIR__, "sixvertex_rate_scan.jl"))

const shared_steps = 70
const shared_cs = [0.5, 1.0, 1.4, 1.7, 1.9]
const massive_c = 0.1       # central charge below which the flow counts as massive
const massive_steps = 3     # consecutive massive steps after which a label is stopped

# K for a = b = 1 (inverse of vertex_c)
K_of(c) = π / (4 * asin(c / 2))

label(src, x) = src == "otsuki" ? "K$(Ktag(x))" : "c$(cstring(x))"
measurefile(src, χ, x) = joinpath(datadir, "sixvertex_rate_cft_$(src)_chi$(χ)_$(label(src, x)).jls")
# otsuki runs extended to extended_steps by `sixvertex_rate_scan.jl extend` are read from the
# extended folder; the results already measured on steps 1..nsteps are kept (resume)
const extended_steps = 70
otsuki_dir(χ, K) = isdir(tensordir(χ, K; steps = extended_steps)) ? tensordir(χ, K; steps = extended_steps) : tensordir(χ, K)

function source(src, χ, x)
    src == "otsuki" && return (K = x, c = vertex_c(x), dir = otsuki_dir(χ, x))
    src == "fushio" && return (K = K_of(x), c = x, dir = joinpath(tensorroot, "symmetry=CU1Irrep", "a=1_b=1",
        "c=$(cstring(x))", "LoopTNR", "chi=$(χ)", "RGsteps=$(shared_steps)_init=true"))
    error("source must be otsuki or fushio")
end

function measure(src, χ, x, code)
    (; K, c, dir) = source(src, χ, x)
    files = sort(filter(f -> occursin(r"^step_\d{3}\.jls$", f), readdir(dir)))
    ns = [parse(Int, f[6:8]) for f in files if f != "step_000.jls"]
    out = measurefile(src, χ, x)
    steps = NamedTuple{(:n, :c, :dims, :τ0, :t_cft), Tuple{Int, Float64, Dict{String, Vector{Float64}}, ComplexF64, Float64}}[]
    # A step whose CFT extraction fails (degenerate tensors deep in a massive phase) is recorded
    # in `failed` and skipped; the remaining steps are still measured.
    failed = NamedTuple{(:n, :error), Tuple{Int, String}}[]
    stopped = nothing
    # Resume: keep the steps already in the file and measure only the missing ones
    if isfile(out)
        r = deserialize(out)
        append!(steps, r.steps)
        append!(failed, get(r, :failed, []))
        stopped = get(r, :stopped, nothing)
    end
    if stopped !== nothing || length(steps) + length(failed) == length(ns)
        println(label(src, x), ", χ = $(χ): already measured, skipped")
        return nothing
    end
    done_ns = Set(vcat([s.n for s in steps], [f.n for f in failed]))
    save() = serialize(out, (; K, vertex_c = c, χ, levels = Dict(string(k) => v for (k, v) in levels), code,
        steps, failed, stopped, source = dir, nsaved = length(ns)))
    for n in ns
        n in done_ns && continue
        # Deep in the massive phase the extraction is slow (minutes per step at χ = 40) and the
        # steps are far outside the fit window: stop once c < massive_c for massive_steps steps
        recent = [s.c for s in steps if s.n < n][max(1, end - massive_steps + 1):end]
        if length(recent) == massive_steps && all(<(massive_c), recent)
            stopped = (n = n, error = "massive phase: c < $(massive_c) for $(massive_steps) steps")
            save()
            @printf "%s χ = %d  %s  stopped before step %d (massive phase)\n" src χ label(src, x) n
            flush(stdout)
            break
        end
        saved = deserialize(joinpath(dir, @sprintf("step_%03d.jls", n)))
        saved.rg_step == n || error("step mismatch in $(dir): file $(n), saved $(saved.rg_step)")
        d = nothing
        t_cft = @elapsed try
            d = CFTData(saved.scheme; Nh = levels)
        catch err
            err isa InterruptException && rethrow()
            push!(failed, (n = n, error = sprint(showerror, err)))
        end
        if d === nothing
            save()
            @printf "%s χ = %d  %s  step %2d  CFT failed: %s\n" src χ label(src, x) n first(split(failed[end].error, '\n'))
            flush(stdout)
            continue
        end
        sv = d.scaling_dimensions
        dims = Dict(string(k) => sort(real.(sv[k])) for k in keys(sv.structure))
        push!(steps, (n = n, c = real(d.central_charge), dims = dims, τ0 = ComplexF64(d.modular_parameter), t_cft = t_cft))
        save()
        Δ = get(dims, sector_m1, Float64[])
        Km = isempty(Δ) ? NaN : 1 / (4Δ[1])
        @printf "%s χ = %d  %s (K = %.4f)  step %2d  c = %.5f  K_m = %.5f  (%+.2e)  cft %.1f s\n" src χ label(src, x) K n real(d.central_charge) Km (Km - K) / K t_cft
        flush(stdout)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
const s00 = string(CU1Irrep(0, 0))
const report_runs = [("otsuki", 16, K_grid), ("otsuki", 24, K_grid), ("fushio", 28, shared_cs), ("fushio", 40, shared_cs)]
const χcolors = Dict(16 => :blue, 24 => :red, 28 => :green, 40 => :purple)

# K and, per step, (K_m - K)/K, (K_e - K)/K and P
function series3(src, χ, x)
    r = deserialize(measurefile(src, χ, x))
    K = r.K
    Km, Ke, P = Dict{Int, Float64}(), Dict{Int, Float64}(), Dict{Int, Float64}()
    for s in r.steps
        Δm = get(s.dims, sector_m1, Float64[])
        Δ0 = get(s.dims, s00, Float64[])
        (isempty(Δm) || length(Δ0) < 2) && continue
        Δe = K < 2 ? Δ0[2] : Δ0[2:end][argmin(abs.(Δ0[2:end] .- K))]
        Km[s.n] = 1 / (4Δm[1]) / K - 1
        Ke[s.n] = Δe / K - 1
        P[s.n] = 4Δe * Δm[1] - 1
    end
    return K, Km, Ke, P
end

# (r, window, α) with the scan's rule; r = NaN when fewer than min_points steps
function rate_of(dev)
    w = fit_window(dev)
    (w === nothing || w[2] - w[1] + 1 < min_points) && return (NaN, w, NaN)
    r, α = fit_rate(dev, w)
    return (r, w, α)
end

median_of(v) = isempty(v) ? NaN : sort(v)[cld(length(v), 2)]

function report()
    mkpath(figdir)
    winstr(w) = w === nothing ? "  -  " : @sprintf("%2d-%2d", w...)

    println("floor: median over steps 8-14 of |K_m - K|/K and |P|, labels with K >= 1.5")
    for (src, χ, xs) in report_runs
        fk, fp = Float64[], Float64[]
        for x in xs
            K, Km, _, P = series3(src, χ, x)
            K >= 1.5 || continue
            append!(fk, [abs(Km[n]) for n in 8:14 if haskey(Km, n)])
            append!(fp, [abs(P[n]) for n in 8:14 if haskey(P, n)])
        end
        @printf "  χ = %2d   K_m %.1e   P %.1e\n" χ median_of(fk) median_of(fp)
    end

    rows = NamedTuple[]
    for (src, χ, xs) in report_runs
        println("\nχ = $(χ)")
        println("   K       2^(2-K)  r(K_m)  window   r(P)    window   r(K_m)/pred  r(P)/pred  P/(K_m-K)*K at P window start, median over K_m window")
        for x in xs
            K, Km, _, P = series3(src, χ, x)
            rk, wk, _ = rate_of(Km)
            rp, wp, _ = rate_of(P)
            pred = 2.0^(2 - K)
            ratio = (wp === nothing || !haskey(Km, wp[1])) ? NaN : P[wp[1]] / Km[wp[1]]
            # Same ratio inside the K_m window (1% <= |K_m - K|/K <= 10%, the g^2 regime), where it
            # estimates B_P / B_Km; at the P window start K_m may already be far beyond 10%
            bratio = isnan(rk) ? NaN : median_of([P[n] / Km[n] for n in wk[1]:wk[2] if haskey(P, n)])
            @printf "  %.3f   %6.3f   %6.3f  %s   %6.3f  %s    %6.3f      %6.3f     %+.2f   %+.2f\n" K pred rk winstr(wk) rp winstr(wp) rk / pred rp / pred ratio bratio
            push!(rows, (; χ, K, pred, rk, rp, bratio))
        end
    end

    println("\nearly decay of P per step (fit over steps 5-10) against the umklapp prediction 2^((2-4K)/2)")
    for χ in (16, 24), K in (0.60, 0.65, 0.70, 0.75)
        _, _, _, P = series3("otsuki", χ, K)
        ns = collect(5:10)
        d = exp(([ones(length(ns)) ns] \ log.(abs.([P[n] for n in ns])))[2])
        @printf "  χ = %d  K = %.2f   %.3f   (prediction %.3f)\n" χ K d 2.0^((2 - 4K) / 2)
    end

    savefig(example_figure(), joinpath(figdir, "sixvertex_rate_example.png"))
    savefig(controls_figure(), joinpath(figdir, "sixvertex_rate_controls.png"))
    savefig(ratio_figure(rows), joinpath(figdir, "sixvertex_rate_ratio.png"))
    savefig(bratio_figure(rows), joinpath(figdir, "sixvertex_rate_pkm.png"))
    println("\nfigures saved to ", figdir)
    return nothing
end

# χ = 24, K = 1.10: |K_m - K|/K and |P| per step (filled > 0, open < 0) with the fits
function example_figure(; χ = 24, K = 1.10)
    _, Km, _, P = series3("otsuki", χ, K)
    p = plot(; yscale = :log10, xlabel = "RG step n", ylabel = "relative deviation", framestyle = :box,
        legend = :topleft, ylims = (1e-4, 1), size = (900, 550), left_margin = 8Plots.mm, bottom_margin = 6Plots.mm,
        title = @sprintf("χ = %d, K = %.2f (prediction 2^(2-K) = %.3f per step)", χ, K, 2.0^(2 - K)))
    hline!(p, collect(window); color = :gray, linestyle = :dot, label = "1%, 10%")
    for (dev, name, color) in [(Km, "(K_m - K)/K", :blue), (P, "P", :red)]
        ns = sort(collect(keys(dev)))
        pos = [n for n in ns if dev[n] > 0]
        neg = [n for n in ns if dev[n] < 0]
        scatter!(p, pos, [dev[n] for n in pos]; color, markersize = 4, markerstrokewidth = 0, label = "$(name) > 0")
        scatter!(p, neg, [-dev[n] for n in neg]; markercolor = :white, markerstrokecolor = color,
            markersize = 4, markerstrokewidth = 1.2, label = "$(name) < 0")
        r, w, α = rate_of(dev)
        isnan(r) && continue
        nw = collect(w[1]:w[2])
        plot!(p, nw, exp.(α .+ log(r) .* nw); color, linewidth = 3, label = @sprintf("fit %d-%d: r = %.3f", w..., r))
    end
    return p
end

# No relevant growth: K_m and K_e drift together (a change of the radius) while P stays flat
function controls_figure()
    panels = map([("otsuki", 24, 2.5), ("fushio", 28, 0.5)]) do (src, χ, x)
        K, Km, Ke, P = series3(src, χ, x)
        ns = [n for n in sort(collect(keys(Km))) if n >= 6]
        p = plot(; xlabel = "RG step n", ylabel = "deviation (%)", framestyle = :box, legend = :bottomleft,
            title = @sprintf("χ = %d, K = %.2f (no relevant growth)", χ, K), ylims = (-1.5, 1.5))
        hline!(p, [0]; color = :gray, label = "")
        for (d, lab, col, ls) in [(Km, "(K_m - K)/K", :blue, :solid), (Ke, "(K_e - K)/K", :green, :solid),
                                  (P, "P ≈ (K_e - K_m)/K", :red, :dash)]
            plot!(p, ns, 100 .* [d[n] for n in ns]; color = col, linestyle = ls, marker = :circle, markersize = 2,
                markerstrokewidth = 0, label = lab)
        end
        p
    end
    return plot(panels...; layout = (1, 2), size = (1300, 480), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm)
end

# r / 2^(2-K) against K, from K_m (left) and from P (right), one colour per χ (K < 2 only).
# χ = 16 is left out of the P panel: its P floor (~1%) sits at the fit threshold.
function ratio_figure(rows)
    panels = map([(:rk, "from K_m", (16, 24, 28, 40)), (:rp, "from P (χ ≥ 24)", (24, 28, 40))]) do (f, name, χs)
        p = plot(; xlabel = "Luttinger parameter K", ylabel = "r / 2^(2-K)", framestyle = :box, legend = :bottomleft,
            title = "growth rate / prediction, $(name)", ylims = (0.6, 1.4), xlims = (0.55, 2.05))
        hline!(p, [1.0]; color = :black, label = "prediction")
        hline!(p, [0.95, 1.05]; color = :gray, linestyle = :dot, label = "±5%")
        for χ in χs
            sel = [r for r in rows if r.χ == χ && r.K < 2 && !isnan(getfield(r, f))]
            scatter!(p, [r.K for r in sel], [getfield(r, f) / r.pred for r in sel]; color = χcolors[χ],
                markersize = χ >= 28 ? 6 : 4, markerstrokewidth = 0, label = "χ = $(χ)")
        end
        p
    end
    return plot(panels...; layout = (1, 2), size = (1300, 500), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm)
end

# Left: P / [(K_m - K)/K] in the K_m window against K (≈ B_P / B_Km), χ ≥ 24.
# Right: χ = 24, K = 1.60, where this ratio is small and P reaches 1% only after K_m has left the g^2 regime.
function bratio_figure(rows)
    p = plot(; xlabel = "Luttinger parameter K", ylabel = "P / [(K_m - K)/K]", framestyle = :box, legend = :bottomright,
        title = "P relative to (K_m - K)/K inside the K_m fit window", xlims = (0.55, 2.05))
    hline!(p, [0.0]; color = :gray, label = "")
    for χ in (24, 28, 40)
        sel = [r for r in rows if r.χ == χ && r.K < 2 && !isnan(r.bratio)]
        scatter!(p, [r.K for r in sel], [r.bratio for r in sel]; color = χcolors[χ],
            markersize = χ >= 28 ? 6 : 4, markerstrokewidth = 0, label = "χ = $(χ)")
    end
    return plot(p, example_figure(; χ = 24, K = 1.60); layout = (1, 2), size = (1400, 520),
        left_margin = 8Plots.mm, bottom_margin = 8Plots.mm)
end

if abspath(PROGRAM_FILE) == @__FILE__() && ARGS[1] == "report"
    report()
elseif abspath(PROGRAM_FILE) == @__FILE__
    src = ARGS[1]
    χ = parse(Int, ARGS[2])
    xs = length(ARGS) > 2 ? parse.(Float64, ARGS[3:end]) : (src == "otsuki" ? K_grid : shared_cs)
    code = code_version()
    for x in xs
        measure(src, χ, x, code)
    end
end
