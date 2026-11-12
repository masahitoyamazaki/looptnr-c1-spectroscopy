# Compare width_read_181.jl (two sectors) with TNRKit CFTData(shape=[1,8,1]) on a small checkpoint.
using TNRKit, TensorKit, Serialization, Printf
src = read("width_read_181.jl", String)
include_string(Main, src[1:findlast("\nmain()", src)[1]])
s = deserialize(joinpath(ENV["DIR"], @sprintf("step_%03d.jls", parse(Int, ENV["N"])))).scheme
t = time(); d = CFTData(s.TA, s.TB; shape = [1, 8, 1], trunc = truncrank(16)); tfull = time() - t
sv = d.scaling_dimensions
k = [q for q in keys(sv.structure) if string(q) == string(CU1Irrep(1, 2))][1]
full = sort(real.(collect(sv[k])))[1:2]
t = time(); x, info = read181(s.TA, s.TB, 16, 3); tr = time() - t
println("full  x12 = ", full, "  (", round(tfull, digits = 1), " s)")
println("mine  x12 = ", x[1:2], "  (", round(tr, digits = 1), " s)  info=", info)
println("rel diff  = ", abs(full[1] - x[1]) / full[1])
