# Width check, stage 2 (2026-11-12, Yamazaki): read saved checkpoints with the [1,8,1] tube of TNRKit 0.7.0,
# restricted to the two sectors needed for the correlation length.
#
# Same construction as TNRKit.CFTData(TA, TB; shape=[1,8,1], trunc) (src/utility/cft.jl): canonical normalization
# area_term(TA,TB)^(1/4), one more coarse-graining by MPO_opt + reduced_MPO truncated to TRUNC, and the transfer
# matrix MPO_action_1x4 on xspace = (A ⊗ B ⊗ A ⊗ B) bond spaces.  Instead of all sectors with Nh = 25 levels,
# only the vacuum sector (0,0) (leading eigenvalue, for the normalization) and the charge-one sector (1,2)
# (lowest NH levels) are solved.  x = -ln|lambda/lambda_0| / (2 pi Im tau), Im tau = 1/8, and
# xi = C / (2 pi x_(1;2)) with C = 8 * sqrt(2)^n, as in width_read.jl.
#
# ENV: DIR (checkpoint folder), STEPS (comma list), TRUNC (default 40), NH (default 3), OUT (csv, appended)
# Output per step: shape, trunc, n, C, x_(1;2), xi, x of second (1;2) level, seconds, krylov info
using TNRKit, TensorKit, KrylovKit, LinearAlgebra, Printf, Serialization

function read181(TA, TB, tr, nh)
    nc = TNRKit.area_term(TA, TB)^(1 / 4)
    dl, ur, ul, dr = TNRKit.MPO_opt(TA / nc, TB / nc, truncrank(tr), trunctol(; rtol = 1.0e-14))
    T = TNRKit.reduced_MPO(dl, ur, ul, dr, truncrank(tr))
    xspace = domain(T)[1] ⊗ domain(T)[1] ⊗ domain(T)[1] ⊗ domain(T)[1]
    f = a -> TNRKit.MPO_action_1x4(T, T, a)
    solve(q, k) = begin
        x = ones(xspace ← Vect[CU1Irrep](q => 1))
        vals, _, info = eigsolve(f, x, k, :LM; krylovdim = max(20, 4k), maxiter = 100, tol = 1.0e-12, verbosity = 0)
        vals, info
    end
    v0, i0 = solve(CU1Irrep(0, 0), 1)
    v1, i1 = solve(CU1Irrep(1, 2), nh)
    imtau = 1 / 8
    x = sort(real.(-log.(abs.(v1) ./ abs(v0[1])) ./ (2pi * imtau)))
    return x, (i0.converged, i1.converged, i1.numiter)
end

function main()
    dir = ENV["DIR"]; tr = parse(Int, get(ENV, "TRUNC", "40")); nh = parse(Int, get(ENV, "NH", "3"))
    open(ENV["OUT"], "a") do io
        for n in parse.(Int, split(ENV["STEPS"], ","))
            s = deserialize(joinpath(dir, @sprintf("step_%03d.jls", n))).scheme
            t0 = time()
            x, info = read181(s.TA, s.TB, tr, nh)
            C = 8 * sqrt(2)^n
            row = ("181", tr, n, C, x[1], C / (2pi * x[1]), length(x) > 1 ? x[2] : NaN, time() - t0, join(info, "/"))
            println(io, join(row, ",")); flush(io)
        end
    end
end
main()
