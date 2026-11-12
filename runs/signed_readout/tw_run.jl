# Signed (pi-twist) readout runs for paper 1, Sec. VI D (2026-11-12, Yamazaki).
# Same LoopTNR parameters, physical tensor, ordinary readout (common.jl) and twisted contraction
# (twist_measure.jl) as Fushio's twist campaign (support/twist_campaign_20261109/worker.jl);
# only the bookkeeping (guards, checkpoint store) is omitted.
#
# ENV: K, CHI, H (a/b alternation mu, default 0), STEPS (max RG steps), STOP (1: stop when |X|>0.1 for n>=6),
#      PERTURB (relative size of a random perturbation of the initial tensors, default 0), SEED, OUT (csv path)
using TNRKit, TensorKit, LinearAlgebra, Printf, Random
const HERE = @__DIR__
include(joinpath(HERE, "common.jl"))
using .SymmetryValidation: physical_tensor, measure_loop
include(joinpath(HERE, "twist_measure.jl"))
const SELECTED = [CU1Irrep(0,0), CU1Irrep(0,1), CU1Irrep(1,2), CU1Irrep(2,2), CU1Irrep(3,2)]

function parameters()
    default = LoopParameters()
    crit = TNRKit.maxiter(20) & TNRKit.convcrit(1e-9, default.sweeping.crits[2].f)
    LoopParameters(sweeping = crit, truncentanglement = trunctol(; rtol = 1e-14))
end

function neutral_top(A, B; flux = false)
    xs = domain(A)[1] ⊗ domain(B)[1]
    Az = flux ? multiply_first(A, pi_rotation(codomain(A)[1])) : A
    Bz = flux ? multiply_first(B, pi_rotation(codomain(B)[1])) : B
    out = Dict{Int,ComplexF64}()
    for (q, parity) in ((CU1Irrep(0,0), 1), (CU1Irrep(0,1), -1))
        p = zeros(ComplexF64, xs ← Vect[CU1Irrep](q => 1)); N = length(p.data)
        N == 0 && continue
        M = zeros(ComplexF64, N, N)
        for j in 1:N
            fill!(p.data, 0); p.data[j] = 1; M[:, j] = flux_action(A, B, Az, Bz, p).data
        end
        E = eigvals(M); out[parity] = E[argmax(abs.(E))]
    end
    out
end

function twist_split(s)
    per = neutral_top(s.TA, s.TB); flux = neutral_top(s.TA, s.TB; flux = true)
    v0 = abs(per[1])
    xe = -log(abs(flux[1]) / v0) / (2pi); xo = -log(abs(flux[-1]) / v0) / (2pi)
    xo - xe
end

function perturbed(T, eps, rng)
    eps == 0 && return T
    U = copy(T)
    for (c, b) in blocks(U)
        b .*= 1 .+ eps .* randn(rng, size(b))
    end
    U
end

function main()
    BLAS.set_num_threads(1)
    K = parse(Float64, ENV["K"]); chi = parse(Int, ENV["CHI"]); h = parse(Float64, get(ENV, "H", "0"))
    steps = parse(Int, get(ENV, "STEPS", "80")); stop = get(ENV, "STOP", "1") == "1"
    eps = parse(Float64, get(ENV, "PERTURB", "0")); rng = MersenneTwister(parse(Int, get(ENV, "SEED", "1")))
    out = ENV["OUT"]
    TA = physical_tensor("CU1", K; a = 1 + h, b = 1 - h); TB = physical_tensor("CU1", K; a = 1 - h, b = 1 + h)
    force2 = get(ENV, "FORCE2", "0") == "1"   # use the two-tensor constructor also at h = 0
    s = h == 0 && eps == 0 && !force2 ? LoopTNR(TA) : LoopTNR(perturbed(TA, eps, rng), perturbed(TB, eps, rng))
    finalize!(s)
    par = parameters()
    open(out, "w") do io
        println(io, "# K=$K chi=$chi h=$h perturb=$eps seed=$(get(ENV,"SEED","1")) julia=$(VERSION) TNRKit=$(pkgversion(TNRKit)) host=$(gethostname())")
        println(io, "n,X,split_pi")
        for n in 0:steps
            if n > 0
                TNRKit.LoggingExtras.withlevel(; verbosity = 0) do
                    TNRKit.step!(s, truncrank(chi), par, 0)
                end
                finalize!(s)
            end
            m = measure_loop(s; nh = 8, selected = SELECTED)
            q = only(filter(r -> r["sector"] == string(CU1Irrep(1,2)) && r["rank"] == 1, m["spectra"]))
            X = 1 - 4K * q["x"]
            sp = (n >= 4 && n % 4 == 0) ? twist_split(s) : NaN
            println(io, n, ",", X, ",", sp); flush(io)
            if stop && n >= 6 && abs(X) > 0.1
                println(io, "# n_e=", n - 1); break
            end
        end
    end
end
main()
