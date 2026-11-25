# Paper 1b check (2026-11-24): how much does a round-off difference between machines change xi?
# Same script, TNRKit 0.7.0 and TensorKit 0.16.5 on two machines (Mac, Julia 1.10.4; Linux a6000 machine, Julia 1.11.9).
# Runs the standard LoopTNR (O(2), K = 1, chi = 35) until |X| > 3, recording X(n), x01, and g_d^pi(n) at every step,
# and the sine-Gordon prediction of xi from g_d^pi(n). The machines are compared through the steps at which |X|
# crosses fixed values (compare.py).
# Usage (from this directory): julia --project=~/tnr_1c xi_machine.jl <outfile> [K] [chi] [nmax]
using Printf
include(joinpath(@__DIR__, "readout_common.jl"))

const COLS = ["K", "chi", "n", "X", "x01", "gd_pi", "xi_pred", "seconds"]

function main(path, K, chi, nmax)
    par = make_loop_params()
    s = LoopTNR(sixvertex_cu1(K)); finalize!(s)
    ensure_csv_header(path, COLS)
    t0 = time()
    for n in 1:nmax
        TNRKit.LoggingExtras.withlevel(; verbosity = 0) do
            TNRKit.step!(s, truncrank(chi), par, 0)
        end
        finalize!(s)
        n < 4 && continue
        m = measure_ordinary(s, K); tw = twist_split(s)
        append_csv_row(path, COLS, Dict{String,Any}("K" => K, "chi" => chi, "n" => n, "X" => m.X, "x01" => m.x01,
            "gd_pi" => tw.gd_pi, "xi_pred" => xi_pred_from_gd_pi(K, n, tw.gd_pi), "seconds" => time() - t0))
        @printf("n=%d X=%+.4e gd_pi=%+.4e\n", n, m.X, tw.gd_pi); flush(stdout)
        abs(m.X) > 3 && break
    end
end

main(ARGS[1], length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0, length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 35,
     length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 48)
