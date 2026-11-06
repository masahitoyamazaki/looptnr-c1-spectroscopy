include("measure.jl")
out=Dict{String,Any}[]
for job in readlines(joinpath(HERE,"control/jobs.tsv"))
    source,ns,path,hash=split(job,'\t');n=parse(Int,ns);n in (6,8) || continue
    c=TOML.parsefile(source)["config"];record=deserialize(joinpath(REPO,path))
    Q,_,geo=shifted_Q(record.scheme,n,c["delta"],c["dt"],c["beta_cell"])
    old=TOML.parsefile(joinpath(HERE,"../perturbed_chi32_20261105/native_121/data",c["id"]*".toml"))
    prior=only(m for m in old["measurements"] if m["step"]==n)
    errs=Float64[];mu0=0im
    for q in (CU1Irrep(0,0),CU1Irrep(1,2),CU1Irrep(2,2),CU1Irrep(3,2))
        d=dense_Q_sector(Q,q;nh=1);l=first(d["levels"]);mu=complex(l["mu_real"],l["mu_imag"])
        q==CU1Irrep(0,0) && (mu0=mu)
        x=-log(abs(mu/mu0))/(2pi*imag(geo.tau))
        p=first(only(s for s in prior["sectors"] if s["sector"]==string(q))["levels"])
        push!(errs,abs(x-p["x"]))
    end
    @assert maximum(errs)<1e-10
    push!(out,Dict("id"=>c["id"],"step"=>n,"rotated"=>geo.rotated,"errors"=>errs))
    println(c["id"]," RG",n," maximum old-readout difference ",maximum(errs));flush(stdout)
end
atomic_toml(joinpath(HERE,"control/native_reuse_validation.toml"),Dict("checks"=>out))
