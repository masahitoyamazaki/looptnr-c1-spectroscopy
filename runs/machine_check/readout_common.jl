# Readout helpers for runs/machine_check, copied from Cho/scripts/paper1b_seed_experiments_20261119/lib_common.jl
# (2026-11-24) without the input-perturbation helpers, which this check does not use.
# Shared utilities for Paper 1b mechanism experiments (2026-11-19).
# Strictly isolated under Cho/; mirrors Yamazaki/runs/signed_readout & midrg_comp_20261115 conventions.
using TNRKit, TensorKit, LinearAlgebra, Printf, Random, Dates

BLAS.set_num_threads(1)

const SELECTED_SECTORS = [
    CU1Irrep(0, 0),
    CU1Irrep(0, 1),
    CU1Irrep(1, 2),
    CU1Irrep(2, 2),
    CU1Irrep(3, 2),
]

function sixvertex_cu1(K::Real; a::Real = 1.0, b::Real = 1.0, vertex_c = nothing)
    cv = if vertex_c !== nothing
        Float64(vertex_c)
    elseif abs(K - 0.5) < 1e-12
        2.0
    else
        round(2 * sin(pi / (4 * Float64(K))); digits = 12)
    end
    return sixvertex(CU1Irrep; a = Float64(a), b = Float64(b), c = cv)
end

function make_loop_params(;
    sweeps::Int = 20,
    tol::Float64 = 1e-9,
    one_loop_init::Bool = true,
    nuclear_norm::Bool = false,
    xi_init::Float64 = 1e-5,
    rho::Float64 = 0.8,
    xi_min::Float64 = 1e-7,
)
    default = LoopParameters()
    crit = TNRKit.maxiter(sweeps) & TNRKit.convcrit(tol, default.sweeping.crits[2].f)
    return LoopParameters(
        sweeping = crit,
        one_loop_init = one_loop_init,
        truncentanglement = trunctol(; rtol = 1e-14),
        nuclear_norm = nuclear_norm,
        ρ = rho,
        ξ_init = xi_init,
        ξ_min = xi_min,
    )
end

# ---- Pi-twist contraction (identical to signed_readout/twist_measure.jl & tw_run.jl) ----
function pi_rotation(V)
    z = id(Float64, V)
    for q in sectors(V)
        isinteger(q.j) || error("This CU1-preserving pi-flux construction requires integer-charge blocked legs")
        block(z, q) .*= isodd(Int(q.j)) ? -1 : 1
    end
    @assert norm(z * z - id(Float64, V)) < 1e-12
    return z
end

function multiply_first(A, z)
    @tensor out[-1 -2; -3 -4] := z[-1; a] * A[a -2; -3 -4]
    return out
end

function flux_action(A, B, Az, Bz, x)
    @tensor begin
        y[-1 -2; -3] := Az[c -1; 1 m] * x[1 2; -3] * B[m -2; 2 c]
        out[-1 -2; -3] := Bz[c -1; 1 m] * y[1 2; -3] * A[m -2; 2 c]
    end
    return out
end

function flux_action_vec(A, B, Az, Bz, x)
    @tensor begin
        y[-1 -2] := Az[c -1; 1 m] * x[1 2] * B[m -2; 2 c]
        out[-1 -2] := Bz[c -1; 1 m] * y[1 2] * A[m -2; 2 c]
    end
    return out
end

function leading_2row_eig(A, B, q::CU1Irrep; flux::Bool = false)
    xs = domain(A)[1] ⊗ domain(B)[1]
    Az = flux ? multiply_first(A, pi_rotation(codomain(A)[1])) : A
    Bz = flux ? multiply_first(B, pi_rotation(codomain(B)[1])) : B
    p = zeros(ComplexF64, xs ← Vect[CU1Irrep](q => 1))
    N = length(p.data)
    N == 0 && return ComplexF64(NaN)
    if N <= 512
        M = zeros(ComplexF64, N, N)
        for j in 1:N
            fill!(p.data, 0)
            p.data[j] = 1
            M[:, j] = flux_action(A, B, Az, Bz, p).data
        end
        E = eigvals(M)
        return E[argmax(abs.(E))]
    else
        randn!(MersenneTwister(20261119), p.data)
        p /= norm(p)
        vals, _, info = TNRKit.eigsolve(v -> flux_action(A, B, Az, Bz, v), p, 1, :LM; krylovdim = 40, maxiter = 100, tol = 1e-12, verbosity = 0)
        info.converged >= 1 || @warn "eigsolve did not converge for q=$q N=$N"
        return vals[1]
    end
end

function twist_split(s::LoopTNR)
    lp_even = abs(leading_2row_eig(s.TA, s.TB, CU1Irrep(0, 0); flux = false))
    lf_even = abs(leading_2row_eig(s.TA, s.TB, CU1Irrep(0, 0); flux = true))
    lf_odd  = abs(leading_2row_eig(s.TA, s.TB, CU1Irrep(0, 1); flux = true))
    xe = -log(lf_even / lp_even) / (2pi)
    xo = -log(lf_odd / lp_even) / (2pi)
    dpi = xo - xe
    xbar_pi = (xe + xo) / 2
    gd_pi = -dpi / (2 * sqrt(2) * pi)
    return (delta_pi = dpi, gd_pi = gd_pi, xbar_pi = xbar_pi, xe_pi = xe, xo_pi = xo)
end

# ---- Rotate90 for anisotropy ----
function rotate90_tensors(A, B)
    p = ((3, 1), (4, 2))
    return permute(B, p), permute(A, p)
end

function canonical_area_eig(A, B)
    x = zeros(Float64, domain(A)[1] ⊗ domain(B)[1])
    function f(v)
        @tensor fx[-1 -2] := A[c -1; 1 m] * v[1 2] * B[m -2; 2 c]
        @tensor ffx[-1 -2] := B[c -1; 1 m] * fx[1 2] * A[m -2; 2 c]
        return ffx
    end
    randn!(MersenneTwister(74103), x.data)
    x /= norm(x)
    vals, _, info = TNRKit.eigsolve(f, x, 1, :LR; krylovdim = 40, maxiter = 100, tol = 1e-12, verbosity = 0)
    info.converged >= 1 || @warn "canonical_area eigsolve did not converge"
    return real(vals[1])
end

function aniso_2x2(s::LoopTNR)
    la = canonical_area_eig(s.TA, s.TB)
    Ar, Br = rotate90_tensors(s.TA, s.TB)
    lb = canonical_area_eig(Ar, Br)
    return log(abs(lb / la))
end

# ---- Ordinary readout using TNRKit MPO_action_2gates directly (faster, identical to common.jl measure_loop) ----
function measure_ordinary(s::LoopTNR, K::Real; nh::Int = 5)
    sc = deepcopy(s)
    A0, B0 = sc.TA, sc.TB
    norm0 = canonical_area_eig(A0, B0)
    A = A0 / norm0^0.25
    B = B0 / norm0^0.25
    I = sectortype(A)
    F = field(A)
    xspace = domain(B) ⊗ domain(B)
    qs = [CU1Irrep(0, 0), CU1Irrep(0, 1), CU1Irrep(1, 2)]
    lambda0 = ComplexF64(NaN)
    levels = Dict{String,Vector{Float64}}()
    for (iq, q) in enumerate(qs)
        V = Vect[I](q => 1)
        x = zeros(Float64, xspace ← V)
        dim(x) > 0 || continue
        randn!(MersenneTwister(2140 + iq), x.data)
        x /= norm(x)
        req = min(nh, dim(x))
        vals, _, _ = TNRKit.eigsolve(v -> TNRKit.MPO_action_2gates(A, B, v), x, req, :LM; krylovdim = max(40, 2 * req + 10), maxiter = 100, tol = 1e-12, verbosity = 0)
        order = sortperm(abs.(vals); rev = true)
        if q == CU1Irrep(0, 0)
            lambda0 = ComplexF64(vals[first(order)])
        end
        nonzero = filter(j -> abs(vals[j]) >= 1e-12, order)
        xs_q = [-log(abs(vals[j] / lambda0)) / pi for j in first(nonzero, min(length(nonzero), req))]
        sort!(xs_q)
        levels[string(q)] = xs_q
    end
    c_raw = real(-12 / pi * log(lambda0))
    q00 = get(levels, string(CU1Irrep(0, 0)), Float64[])
    q01 = get(levels, string(CU1Irrep(0, 1)), Float64[])
    q12 = get(levels, string(CU1Irrep(1, 2)), Float64[])
    x01 = isempty(q12) ? NaN : q12[1]
    X = 1.0 - 4.0 * Float64(K) * x01
    eplus = length(q00) >= 2 ? q00[2] : NaN
    j1, j2, eminus = if Float64(K) >= 1.1
        (get(q01, 1, NaN), get(q01, 2, NaN), get(q01, 3, NaN))
    elseif Float64(K) >= 0.8 && length(q01) >= 3
        pairs = ((1, 2, 3), (1, 3, 2), (2, 3, 1))
        a, b, c = pairs[argmin([abs(q01[i] - q01[j]) for (i, j, _) in pairs])]
        (q01[a], q01[b], q01[c])
    else
        (get(q01, 2, NaN), get(q01, 3, NaN), get(q01, 1, NaN))
    end
    xe = (eplus + eminus) / 2
    Kn = (isfinite(xe) && isfinite(x01) && xe > 0 && x01 > 0) ? 0.5 * sqrt(xe / x01) : NaN
    ghat = (isfinite(x01) && isfinite(eminus)) ? 4.0 * (0.5 - (2.0 * x01 + eminus) / 3.0) : NaN
    ratio_12_0 = isempty(q12) ? NaN : exp(-pi * q12[1])
    return (
        c_raw = c_raw,
        x01 = x01,
        X = X,
        eplus = eplus,
        eminus = eminus,
        j1 = j1,
        j2 = j2,
        Kn = Kn,
        ghat = ghat,
        ratio_12_0 = ratio_12_0,
    )
end


# ---- Sine-Gordon kappa_SG(p) and early xi prediction (Eqs. 8-10 of Paper 1b) ----
# Pure-Julia Lanczos approximation (g=7, n=9, accuracy ~1e-15 for z > 0) to avoid extra Project.toml deps
function gamma_fn(z::Float64)
    if z < 0.5
        return pi / (sin(pi * z) * gamma_fn(1.0 - z))
    end
    z -= 1.0
    p = (
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )
    x = p[1]
    for i in 2:length(p)
        x += p[i] / (z + i - 1.0)
    end
    t = z + 7.5
    return sqrt(2.0 * pi) * t^(z + 0.5) * exp(-t) * x
end

function kappa_sg(K::Float64)
    p = K / (2.0 - K)
    return (gamma_fn(p / (p + 1.0)) / (pi * gamma_fn(1.0 / (p + 1.0)))
        * (sqrt(pi) * gamma_fn((p + 1.0) / 2.0) / (2.0 * gamma_fn(p / 2.0)))^(2.0 / (p + 1.0)))
end

function xi_pred_from_gd_pi(K::Float64, n::Int, gd_pi::Float64)
    # gd_pi refers to twisted circumference 2*a_n; L_std / sqrt(2) = 2*a_n = 2*(sqrt(2)^n)
    two_a_n = 2.0 * (sqrt(2.0)^n)
    return (two_a_n / (2.0 * pi)) * (sqrt(2.0) * kappa_sg(K) / abs(gd_pi))^(1.0 / (2.0 - K))
end

# ---- Resume-safe CSV utilities ----
function ensure_csv_header(path::String, cols::Vector{String})
    mkpath(dirname(path))
    if !isfile(path)
        open(path, "w") do io
            println(io, join(cols, ","))
        end
    end
end

function append_csv_row(path::String, cols::Vector{String}, vals::Dict{String,<:Any})
    open(path, "a") do io
        fields = String[]
        for c in cols
            v = get(vals, c, "")
            if v isa AbstractFloat
                push!(fields, isnan(v) ? "NaN" : @sprintf("%.12g", v))
            else
                s = string(v)
                push!(fields, occursin(",", s) ? "\"$s\"" : s)
            end
        end
        println(io, join(fields, ","))
        flush(io)
    end
end

function load_done_ids(path::String; id_col::String = "run_id")
    done = Set{String}()
    isfile(path) || return done
    lines = readlines(path)
    length(lines) <= 1 && return done
    hdr = split(lines[1], ",")
    idx = findfirst(==(id_col), hdr)
    idx === nothing && return done
    for l in lines[2:end]
        isempty(strip(l)) && continue
        parts = split(l, ",")
        length(parts) >= idx && push!(done, String(parts[idx]))
    end
    return done
end
