# Width check of the correlation length for paper 2 (2026-11-12, Yamazaki).
# Reads saved LoopTNR checkpoints (main) with TNRKit 0.7.0's CFTData for several tube shapes.
# CFTData returns x_i = -ln|lambda_i/lambda_0| / (2 pi Im tau); the correlation length in units of the original
# lattice spacing is xi = C / (2 pi x_(1;2)) with C = shape[2] * sqrt(2)^n the physical circumference, which equals
# height / (-ln|lambda_(1;2)/lambda_0|) as in Otsuki/sixvertex/sixvertex_sg_xi.md.
# For wide and [1,8,1], TNRKit coarse-grains (TA, TB) once more, truncated to TRUNC.
#
# ENV: DIR (checkpoint folder), STEPS (comma list), SHAPE (std | wide | 181), TRUNC, OUT (csv, appended)
# Output per step: shape, trunc, n, C, x_(1;2), xi, x of the lowest excited (0,0) level, x of the lowest (0,1), seconds
using TNRKit, TensorKit, LinearAlgebra, Printf, Serialization
const SHAPES = Dict("std" => [sqrt(2), 2sqrt(2), 0], "wide" => [4 / sqrt(10), 2sqrt(10), 2 / sqrt(10)], "181" => [1, 8, 1])

lowest(sv, q; skip = 0) = begin
    k = findfirst(x -> string(x) == string(q), collect(keys(sv.structure)))
    k === nothing && return NaN
    v = sort(real.(collect(sv[collect(keys(sv.structure))[k]])))
    length(v) > skip ? v[1 + skip] : NaN
end

function main()
    dir = ENV["DIR"]; name = ENV["SHAPE"]; shape = SHAPES[name]; tr = parse(Int, get(ENV, "TRUNC", "40"))
    open(ENV["OUT"], "a") do io
        for n in parse.(Int, split(ENV["STEPS"], ","))
            s = deserialize(joinpath(dir, @sprintf("step_%03d.jls", n))).scheme
            t0 = time()
            d = name == "std" ? CFTData(s.TA, s.TB; shape = shape) : CFTData(s.TA, s.TB; shape = shape, trunc = truncrank(tr))
            sv = d.scaling_dimensions
            x12 = lowest(sv, CU1Irrep(1, 2)); C = shape[2] * sqrt(2)^n
            row = (n, C, x12, C / (2pi * x12), lowest(sv, CU1Irrep(0, 0); skip = 1), lowest(sv, CU1Irrep(0, 1)), time() - t0)
            println(io, name, ",", tr, ",", join(row, ",")); flush(io)
        end
    end
end
main()
