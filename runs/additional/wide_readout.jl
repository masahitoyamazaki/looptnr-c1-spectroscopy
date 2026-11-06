# T1 (2026-11-06): re-read saved O(2) checkpoints with a wider tube (no new RG steps).
# Tube [4/sqrt(10), 2 sqrt(10), 2/sqrt(10)] of TNRKit 0.7.0: one extra coarse-graining of (TA, TB) inside the readout
# (MPO_opt + reduced_MPO, truncated to the run's chi), then the two-gate transfer matrix.  Its circumference is
# sqrt(5) times that of the standard tube [sqrt(2), 2 sqrt(2), 0]; Im tau = 0.2 and Re tau / Im tau = 0.5 follow from
# the shape for an isotropic tensor, as Im tau = 1/2 does for the standard tube.  TNRKit returns Delta + i s.
# Usage:  julia --project=<env> wide_readout.jl RUN_DIR CHI NFIRST NLAST OUT_CSV
using TNRKit, TensorKit, LinearAlgebra, Printf, Serialization

dir, chi, n1, n2, out = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4]), ARGS[5]
const SHAPE = [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]
open(out, "w") do io
    println(io, "step,sector,rank,x,spin,central_charge")
    for n in n1:n2
        f = joinpath(dir, @sprintf("step_%03d.jls", n)); isfile(f) || continue
        s = deserialize(f).scheme
        d = CFTData(s.TA, s.TB; shape = SHAPE, trunc = truncrank(chi))
        sv = d.scaling_dimensions
        for q in sort(collect(keys(sv.structure)); by = string)
            vals = sort(collect(sv[q]); by = real)
            for (rank, v) in enumerate(vals[1:min(8, end)])
                @printf(io, "%d,\"%s\",%d,%.14e,%.6f,%.8f\n", n, string(q), rank, real(v), imag(v), real(d.central_charge))
            end
        end
        flush(io); println("step $n done"); flush(stdout)
    end
end
