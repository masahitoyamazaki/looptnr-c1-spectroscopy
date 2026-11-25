include("worker.jl")
BLAS.set_num_threads(1)
old=TOML.parsefile(joinpath(HERE,"../isotropic_twist_20261109/data/plus_from12.toml"))
checks=Dict{String,Any}[]
for n in (12,24)
    ck=only(filter(r->r["step"]==n,old["checkpoints"]))
    path=joinpath(REPO,ck["path"]);@assert bytes2hex(sha256(read(path)))==ck["sha256"]
    s=deserialize(path).scheme
    t=twist_readout(s,n)
    previous=only(filter(r->r["step"]==n,old["twist_measurements"]))
    delta=abs(t["splitting_x"]-previous["splitting_x"])
    @assert delta<1e-11
    @assert t["numerical_gate"]
    # Compare restricted exact diagonalization with the archived all-sector code.
    full=full_spectrum(s.TA,s.TB;flux=true)
    ev,od=[argmax(mag,filter(r->r["C_parity"]==q,full["levels"])) for q in (1,-1)]
    split=-log(mag(od)/mag(ev))/(2pi)
    @assert abs(split-t["splitting_x"])<1e-11
    push!(checks,Dict("step"=>n,"old_difference"=>delta,"full_sector_difference"=>abs(split-t["splitting_x"]),
        "relative_residual"=>t["leading_relative_residual"],"checkpoint_sha256"=>ck["sha256"]))
end
atomic_toml(joinpath(HERE,"control/validation.toml"),Dict("checks"=>checks,"scope"=>"Neutral restriction agrees with validated all-sector contraction and archived measurements; not physical convergence"))
println(checks)
