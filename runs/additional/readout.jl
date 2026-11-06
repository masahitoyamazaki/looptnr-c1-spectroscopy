# B6: re-read saved U(1) checkpoints with more neutral levels (no RG).  Usage:
#   julia --project=<env> readout.jl RUN_DIR NFIRST NLAST NH OUT_CSV
using TNRKit, TensorKit, LinearAlgebra, Printf, Serialization
include(joinpath(@__DIR__, "common.jl"))
using .SymmetryValidation
dir, n1, n2, nh, out = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4]), ARGS[5]
open(out, "w") do io
    println(io, "step,sector,rank,x,phase,lambda_real,lambda_imag,residual,converged,requested")
    for n in n1:n2
        f = joinpath(dir, @sprintf("step_%03d.jls", n)); isfile(f) || continue
        m = measure_loop(deserialize(f).scheme; nh = nh, selected = [U1Irrep(0)])
        for r in m["spectra"]
            @printf(io, "%d,\"%s\",%d,%.16e,%.16e,%.16e,%.16e,%.3e,%d,%d\n", n, r["sector"], r["rank"], r["x"],
                r["phase"], r["lambda_real"], r["lambda_imag"], r["residual"], r["converged"], r["requested"])
        end
        flush(io); println("step $n done"); flush(stdout)
    end
end
