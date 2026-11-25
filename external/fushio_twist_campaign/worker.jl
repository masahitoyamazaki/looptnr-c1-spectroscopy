using TNRKit, TensorKit, LinearAlgebra, Printf, Serialization, SHA, TOML, Random
const HERE=@__DIR__
const REPO=normpath(joinpath(HERE,"../../../../../.."))
include(joinpath(REPO,"Yamazaki/runs/followup_B_20261106/common.jl"))
using .SymmetryValidation: physical_tensor, measure_loop, atomic_toml
include(joinpath(HERE,"../isotropic_twist_20261109/twist_measure.jl"))
const SELECTED=[CU1Irrep(0,0),CU1Irrep(0,1),CU1Irrep(1,2),CU1Irrep(2,2),CU1Irrep(3,2)]
struct RecordedSweeping{C} <: TNRKit.stopcrit
    base::C
    history::Vector{Tuple{Int,Float64}}
end
(c::RecordedSweeping)(n::Int,cost)=(push!(c.history,(n,Float64(real(last(cost)))));c.base(n,cost))
function parameters(trace=nothing)
    default=LoopParameters()
    crit=TNRKit.maxiter(20)&TNRKit.convcrit(1e-9,default.sweeping.crits[2].f)
    LoopParameters(sweeping=trace===nothing ? crit : RecordedSweeping(crit,trace),truncentanglement=trunctol(;rtol=1e-14))
end

# Exactly the previously validated two-row operator, restricted to the two
# neutral C blocks needed for the signed measurement. No RG update is changed.
function neutral_spectrum(A,B;flux=false)
    xs=domain(A)[1]⊗domain(B)[1]
    Az=flux ? multiply_first(A,pi_rotation(codomain(A)[1])) : A
    Bz=flux ? multiply_first(B,pi_rotation(codomain(B)[1])) : B
    rows=Dict{String,Any}[]
    for q in (CU1Irrep(0,0),CU1Irrep(0,1))
        p=zeros(ComplexF64,xs←Vect[CU1Irrep](q=>1));N=length(p.data)
        N==0 && continue
        N<=2048 || error("Neutral dense block exceeds the recorded memory cap")
        M=zeros(ComplexF64,N,N)
        for j in 1:N
            fill!(p.data,0);p.data[j]=1;M[:,j]=flux_action(A,B,Az,Bz,p).data
        end
        E=eigen(M);scale=norm(M)
        for j in sortperm(abs.(E.values);rev=true)[1:min(8,N)]
            v=E.vectors[:,j];mu=E.values[j];err=norm(M*v-mu*v)
            push!(rows,Dict("sector"=>string(q),"C_parity"=>q.s==0 ? 1 : -1,
                "mu_real"=>real(mu),"mu_imag"=>imag(mu),"block_dimension"=>N,
                "matrix_scaled_residual"=>err/max(scale*norm(v),eps()),
                "relative_residual"=>err/max(abs(mu)*norm(v),eps())))
        end
    end
    rows
end
mag(r)=abs(complex(r["mu_real"],r["mu_imag"]))
function twist_readout(s,n)
    n%4==0 || error("Use a fixed lattice orientation")
    per=neutral_spectrum(s.TA,s.TB);flux=neutral_spectrum(s.TA,s.TB;flux=true)
    ev,od=[argmax(mag,filter(r->r["C_parity"]==q,flux)) for q in (1,-1)]
    vac=argmax(mag,filter(r->r["C_parity"]==1,per));v0=mag(vac)
    xe=-log(mag(ev)/v0)/(2pi);xo=-log(mag(od)/v0)/(2pi)
    split=xo-xe
    phase=max(abs(angle(complex(ev["mu_real"],ev["mu_imag"]))),abs(angle(complex(od["mu_real"],od["mu_imag"]))))
    area=SymmetryValidation.canonical_area(s.TA,s.TB)
    area_error=abs(v0/area["eigenvalue"]-1)
    @assert area_error<1e-9
    res=max(ev["relative_residual"],od["relative_residual"],vac["relative_residual"])
    Dict{String,Any}("step"=>n,"tube"=>[2.,2.,0.],"tau_imag"=>1.,
        "normalization"=>"untwisted vacuum; same tensors; fixed Im(tau)=1",
        "untwisted_area_relative_error"=>area_error,"periodic_neutral"=>per,"pi_flux_neutral"=>flux,
        "x_even"=>xe,"x_odd"=>xo,"pair_center"=>(xe+xo)/2,"splitting_x"=>split,
        "g_signed"=>-split/(2sqrt(2)*pi),"max_phase"=>phase,"leading_relative_residual"=>res,
        "numerical_gate"=>phase<1e-5 && res<1e-8 && isfinite(split))
end
function measure!(d,s,n,ck,c)
    seconds=@elapsed m=measure_loop(s;nh=8,selected=SELECTED)
    m["step"]=n;m["seconds"]=seconds;m["checkpoint_sha256"]=ck["sha256"]
    q=only(filter(r->r["sector"]==string(CU1Irrep(1,2))&&r["rank"]==1,m["spectra"]))
    m["paper_X"]=1-4c["K"]*q["x"]
    m["numerical_gate"]=isfinite(q["x"])&&q["x"]>0&&q["residual"]<1e-8&&abs(q["phase"])<1e-5
    push!(d["measurements"],m);d["last_measured_step"]=n
    # A unique measurement namespace on the original checkpoint branch.
    root=dirname(joinpath(REPO,ck["path"]))
    dest=joinpath(root,"CFT/tube=sqrt2_2sqrt2_0/measurement=twist_campaign_20261109",@sprintf("step_%03d.toml",n))
    @assert !ispath(dest)
    atomic_toml(dest,merge(copy(m),Dict("checkpoint_path"=>ck["path"],"config"=>c)))
    if n>=4 && n%4==0
        t=twist_readout(s,n);t["checkpoint_sha256"]=ck["sha256"];push!(d["twist_measurements"],t)
        dest=joinpath(root,"CFT/tube=2_2_0/measurement=twist_campaign_20261109",@sprintf("step_%03d.toml",n))
        @assert !ispath(dest)
        atomic_toml(dest,merge(copy(t),Dict("checkpoint_path"=>ck["path"],"config"=>c)))
    end
    m
end
function main()
    BLAS.set_num_threads(1);@assert Threads.nthreads()==1
    if ARGS==["--profile"]
        println(repr(parameters()));println("PACKAGE_ROOT="*Base.pkgdir(TNRKit));return
    end
    cfg=TOML.parsefile(only(ARGS));c=only(cfg["runs"])
    for (p,h) in cfg["source_sha256"];@assert bytes2hex(sha256(read(p)))==h p;end
    path=joinpath(HERE,"data",c["id"]*".toml");@assert !ispath(path)
    if !get(c,"reuse",false)
        include(joinpath(cfg["coordination_root"],"local_tnr_data/shared_tensor_guard.jl"))
        par=parameters();@assert repr(par)==c["params_repr"]
        Base.invokelatest(shared_tensor_guard;symmetry=CU1Irrep,a=c["a"],b=c["b"],cs=[c["c"]],chis=[c["chi"]],steps=c["steps"],variant=c["variant"],loop_parameters=par)
    end
    atomic_toml(joinpath(HERE,"control/guard_"*c["id"]*".toml"),Dict("epoch"=>time(),"read_only"=>get(c,"reuse",false)))
    d=Dict{String,Any}("config"=>c,"status"=>"running","last_saved_step"=>-1,"last_measured_step"=>-1,
        "rg"=>Dict{String,Any}[],"measurements"=>Dict{String,Any}[],"twist_measurements"=>Dict{String,Any}[],
        "checkpoints"=>Dict{String,Any}[],"errors"=>Dict{String,Any}[],"source_sha256"=>cfg["source_sha256"],
        "physical_convergence"=>false,"julia_version"=>string(VERSION),"TNRKit_version"=>string(pkgversion(TNRKit)),"TensorKit_version"=>string(pkgversion(TensorKit)))
    atomic_toml(path,d)
    try
        if get(c,"reuse",false)
            for ck in c["reuse_checkpoints"]
                n=ck["step"];cp=joinpath(REPO,ck["path"]);@assert bytes2hex(sha256(read(cp)))==ck["sha256"]
                saved=deserialize(cp);s=saved.scheme
                push!(d["checkpoints"],ck);d["last_saved_step"]=n
                measure!(d,s,n,ck,c);atomic_toml(path,d)
                println(c["id"]," reused n=",n);flush(stdout)
            end
            d["status"]="reused_measurements_complete"
            haskey(c,"reused_n_e") && (d["n_e"]=c["reused_n_e"])
        else
            h=c["h"];K=c["K"]
            s=h==0 ? LoopTNR(physical_tensor("CU1",K)) : LoopTNR(physical_tensor("CU1",K;a=1+h,b=1-h),physical_tensor("CU1",K;a=1-h,b=1+h))
            history=[Float64(finalize!(s))]
            for n in 0:c["steps"]
                if n>0
                    trace=Tuple{Int,Float64}[];par=parameters(trace)
                    seconds=@elapsed TNRKit.LoggingExtras.withlevel(;verbosity=0) do
                        TNRKit.step!(s,truncrank(c["chi"]),par,0)
                    end
                    nf=Float64(finalize!(s));@assert isfinite(nf)&&nf>0;push!(history,nf)
                    push!(d["rg"],Dict("step"=>n,"seconds"=>seconds,"normalization"=>nf,"cost_history"=>[Dict("sweep"=>i,"cost"=>v) for (i,v) in unique(trace)]))
                end
                cp=joinpath(REPO,c["checkpoint_path"],@sprintf("step_%03d.jls",n));mkpath(dirname(cp));@assert !ispath(cp)
                serialize(cp*".tmp",(;scheme=s,step=n,rg_step=n,config=c,normalization_history=copy(history)));mv(cp*".tmp",cp)
                ck=Dict("step"=>n,"path"=>relpath(cp,REPO),"sha256"=>bytes2hex(sha256(read(cp))),"bytes"=>filesize(cp))
                push!(d["checkpoints"],ck);d["last_saved_step"]=n;atomic_toml(path,d)
                m=measure!(d,s,n,ck,c);atomic_toml(path,d)
                println(c["id"]," n=",n," X=",m["paper_X"]," gate=",m["numerical_gate"]);flush(stdout)
                if n>=6 && !m["numerical_gate"]
                    d["status"]="numerical_limit";break
                elseif n>=6 && c["stop_at_threshold"] && abs(m["paper_X"])>0.1
                    d["status"]="weak_window_threshold_crossed";d["n_e"]=n-1;d["first_crossing_step"]=n;break
                end
            end
            d["status"]=="running" && (d["status"]="planned_extent_complete")
        end
    catch e
        d["status"]="failed";push!(d["errors"],Dict("error"=>sprint(showerror,e)));showerror(stderr,e,catch_backtrace());println(stderr)
    end
    atomic_toml(path,d)
    d["status"]=="failed" && exit(2)
end
if abspath(PROGRAM_FILE)==(@__FILE__)
    main()
end
