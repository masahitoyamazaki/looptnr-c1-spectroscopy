# CFT-only: saved note11 checkpoints, no RG updates or tensor writes.
include("../continuation_20261103/blocked/source.jl")
const HERE=@__DIR__

# Exact slicing by irrep and multiplicity; retain the entire CU1 irrep.
function sliced_parts(tm)
    V=codomain(tm.TA)[1]; I=sectortype(tm.TA); parts=[]
    for q in sectors(V), i in 1:dim(V,q)
        p=zeros(Float64,V ← Vect[I](q=>1));block(p,q)[i,1]=1
        @tensor a[-1 -2;-3 -4] := conj(p[1;-1])*tm.TA[1 -2;-3 -4]
        @tensor b[-1 -2;-3 -4] := tm.TB[-1 -2;-3 1]*p[1;-4]
        push!(parts,(a,b))
    end
    parts
end
function sliced_action(tm,parts,x)
    total=zero(x)
    for (a,b) in parts
        @tensor y[-1 -2 -3 -4 -5 -6;-7] := a[-1 -2;i -6]*x[i -3 -4 -5;-7]
        @tensor z[-1 -2 -3 -4 -5 -6;-7] := y[-1 -2 i -4 -5 j;-7]*tm.TB[j -3;i -6]
        @tensor y[-1 -2 -3 -4 -5 -6;-7] := z[-1 -2 -3 i -5 j;-7]*tm.TA[j -4;i -6]
        @tensor fx[-1 -2 -3 -4;-5] := y[c -2 -3 -4 i d;-5]*b[d -1;i c]
        axpy!(1.,fx,total)
    end
    total
end

function validation()
    Random.seed!(20261106)
    V=Vect[CU1Irrep]((0,0)=>2,(0,1)=>1,(1,2)=>1)
    a=randn(Float64,V⊗V ← V⊗V);b=randn(Float64,V⊗V ← V⊗V)
    tm=TNRKit.CFTTransferMatrix(a,b,[1.,4.,1.]);parts=sliced_parts(tm)
    errors=Float64[]
    for q in (CU1Irrep(0,0),CU1Irrep(1,2),CU1Irrep(2,2),CU1Irrep(3,2))
        x=randn(Float64,domain(tm) ← Vect[CU1Irrep](q=>1))
        ref=tm(x);got=sliced_action(tm,parts,x)
        push!(errors,norm(got-ref)/norm(ref))
    end
    @assert maximum(errors)<1e-12
    atomic_toml(joinpath(HERE,"control/slicing_validation.toml"),Dict("errors"=>errors,"symmetry"=>"CU1","no_truncation"=>true))
    println("VALIDATED ",errors);flush(stdout)
end

function measure(source,n,path,hash)
    raw=TOML.parsefile(source);cfg=raw["config"]
    dest=joinpath(HERE,"data",cfg["id"],@sprintf("step_%03d.toml",n))
    isfile(dest) && return
    bytes2hex(sha256(read(path)))==hash || error("Checkpoint hash changed")
    saved=deserialize(path);@assert saved.step==n
    geo=blocked_metric(cfg["delta"],cfg["dt"],cfg["beta_cell"],n)
    A=saved.scheme.TA;B=saved.scheme.TB
    if geo.rotated
        p=((3,1),(4,2));A,B=permute(B,p),permute(A,p)
    end
    tm=TNRKit.CFTTransferMatrix(A,B,[1.,4.,1.]);parts=sliced_parts(tm)
    tau=(1+geo.tau0)/4
    out=Dict{String,Any}("config"=>cfg,"h"=>get(cfg,"h",0.),"step"=>n,"shape"=>[1.,4.,1.],"source"=>source,
        "checkpoint_path"=>path,"checkpoint_sha256"=>hash,"status"=>"running","K_vertex"=>geo.K,
        "tau_real"=>real(tau),"tau_imag"=>imag(tau),"rotated"=>geo.rotated,"velocity_dt"=>geo.velocity,
        "physical_ring_L"=>8*2^(n÷2),"Q_time_extent"=>cfg["beta_cell"]*2^(n÷2),
        "measurement_truncation"=>"none; exact CU1 irrep slicing","sectors"=>Dict{String,Any}[],
        "julia_version"=>string(VERSION),"active_project"=>Base.active_project(),"package_root"=>Base.pkgdir(TNRKit))
    if isfile(dest*".progress")
        prior=TOML.parsefile(dest*".progress")
        @assert prior["checkpoint_sha256"]==hash
        out["sectors"]=prior["sectors"]
    end
    out["measurement_source_sha256"]=bytes2hex(sha256(read(@__FILE__)))
    start=time();lambda0=0.0+0im
    for q in (CU1Irrep(0,0),CU1Irrep(1,2),CU1Irrep(2,2),CU1Irrep(3,2))
        done=findfirst(s->s["sector"]==string(q),out["sectors"])
        if !isnothing(done)
            s=out["sectors"][done]
            q==CU1Irrep(0,0) && (lambda0=complex(s["mu_real"],s["mu_imag"]))
            continue
        end
        initial=zeros(Float64,domain(tm) ← Vect[CU1Irrep](q=>1));N=length(initial.data)
        action(x)=sliced_action(tm,parts,x)
        trials=Dict{String,Any}[];best=nothing
        println("START ",cfg["id"]," RG",n," ",q," N=",N);flush(stdout)
        for seed in (20261106,20261107)
            randn!(MersenneTwister(seed),initial.data);normalize!(initial)
            t=time();vals,vs,info=TNRKit.eigsolve(action,initial,1,:LM;krylovdim=12,maxiter=80,tol=1e-11,eager=true,verbosity=0)
            ids=sortperm(abs.(vals);rev=true);j=first(ids);mu=ComplexF64(vals[j]);v=vs[j]
            fx=eltype(v)<:Real ? action(v) : action(real(v))+im*action(imag(v))
            residual=norm(fx-mu*v)/(abs(mu)*norm(v))
            d=Dict("seed"=>seed,"mu_real"=>real(mu),"mu_imag"=>imag(mu),"residual"=>residual,"converged"=>info.converged,"numops"=>info.numops,"seconds"=>time()-t,"krylovdim"=>12,"requested_levels"=>1,"eager"=>true)
            push!(trials,d);best=d
            println("DONE ",q," seed=",seed," sec=",time()-t," residual=",residual);flush(stdout)
        end
        z1=complex(trials[1]["mu_real"],trials[1]["mu_imag"]);z2=complex(trials[2]["mu_real"],trials[2]["mu_imag"])
        agreement=abs(z1-z2)/max(abs(z1),abs(z2));mu=z2
        q==CU1Irrep(0,0) && (lambda0=mu)
        z=mu/lambda0
        push!(out["sectors"],Dict("sector"=>string(q),"dimension"=>N,"trials"=>trials,"start_agreement"=>agreement,
            "mu_real"=>real(mu),"mu_imag"=>imag(mu),"x"=>-log(abs(z))/(2pi*imag(tau)),"phase"=>angle(z),"residual"=>best["residual"]))
        atomic_toml(dest*".progress",out)
    end
    out["seconds"]=time()-start
    out["status"]=all(s["residual"]<1e-8 && s["start_agreement"]<1e-8 for s in out["sectors"]) ? "measured" : "numerical_gate_failure"
    atomic_toml(dest,out);rm(dest*".progress";force=true)
    println("FINISH ",cfg["id"]," RG",n," ",out["status"]," sec=",out["seconds"]);flush(stdout)
end
if abspath(PROGRAM_FILE)==@__FILE__
    BLAS.set_num_threads(1);@assert Threads.nthreads()==1
    @assert realpath(Base.pkgdir(TNRKit))==realpath(joinpath(REPO,"Fushio/vendor/TNRKit.jl"))
    @assert maximum(TOML.parsefile(joinpath(HERE,"control/slicing_validation.toml"))["errors"])<1e-12
    for line in eachline(isempty(ARGS) ? joinpath(HERE,"control/jobs.tsv") : ARGS[1])
        source,n,path,hash=split(line,'\t');measure(source,parse(Int,n),joinpath(REPO,path),hash);GC.gc()
    end
end
