# CFT-only adapter. Scientific kernels are ported from the sources in measurement_environment.json.
using TNRKit, TensorKit, Serialization, LinearAlgebra, TOML, Dates, Printf, SHA, Logging, Random
BLAS.set_num_threads(1)
const JOB=TOML.parsefile(only(ARGS))
const OUT=JOB["output_dir"]
const ROOT=OUT
const WORKER_SHA=bytes2hex(open(sha256,@__FILE__))
const MANIFEST_SHA=bytes2hex(open(sha256,joinpath(@__DIR__,"../environment/Manifest.toml")))
const SHAPES=Dict("small"=>[sqrt(2.),sqrt(2.),0.],"wide"=>[sqrt(2.),2sqrt(2.),0.])
function atomic_toml(path,value)
    isfile(path)&&error("Refusing overwrite: $path")
    mkpath(dirname(path));tmp=path*".tmp"
    open(tmp,"w") do io; TOML.print(io,value); end
    mv(tmp,path)
end
atomic(path,value)=atomic_toml(path,value)
function check_input(row)
    path=row["checkpoint"]
    bytes2hex(open(sha256,path))==row["catalog_sha256"]||error("Checkpoint hash changed: $path")
    d=deserialize(path)
    d.rg_step==parse(Int,row["rg_step"])||error("RG step mismatch")
    d.chi==parse(Int,row["chi"])||error("chi mismatch")
    d.symmetry==row["symmetry"]||error("symmetry mismatch")
    for (name,key) in ((:a,"catalog_a"),(:b,"catalog_b"),(:c,"catalog_c"))
        isapprox(getproperty(d.weights,name),parse(Float64,row[key]);atol=1e-12,rtol=0)||error("weight mismatch: $name")
    end
    d
end

function measure_wide(row)
    sym=row["symmetry"];cv=parse(Float64,row["physical_c"]);n=parse(Int,row["rg_step"])
    label="wide";nh=6;path=row["checkpoint"];output=joinpath(OUT,row["output_record"])
    shape=SHAPES[label]; S=sym=="CU1Irrep" ? CU1Irrep : U1Irrep
    k=pi/(2acos(1-cv^2/2))
    obs=Dict{String,Any}("symmetry"=>sym,"vertex_c"=>cv,"rg_step"=>n,"tube"=>label,
        "shape"=>shape,"chi"=>40,"levels"=>nh,"K_exact"=>k,"status"=>"ok",
        "source"=>relpath(path,ROOT),"julia_version"=>string(VERSION),"utc"=>string(now(UTC)))
    rows=Dict{String,Any}[]; sectorsinfo=Dict{String,Any}[]; warningbuffer=IOBuffer()
    start=time(); println("START $sym c=$cv RG=$n $label");flush(stdout)
    try
        d=check_input(row)
        @assert d.symmetry==sym && d.vertex_c==cv && d.rg_step==n && d.chi==40
        @assert d.weights==(sym=="CU1Irrep" ? (;a=1.,b=1.,c=cv) : (;a=1.,b=cv,c=1.))
        obs["checkpoint_sha256"]=bytes2hex(open(sha256,path))
        with_logger(SimpleLogger(warningbuffer,Logging.Warn)) do
            t=time(); area=TNRKit.area_term(d.scheme.TA,d.scheme.TB)
            isfinite(area)&&area>0 || error("Invalid area normalization $area")
            ta,tb=d.scheme.TA/area^(1/4),d.scheme.TB/area^(1/4)
            tau0,ctau=TNRKit.extract_tau_and_c(ta,tb;fast=true)
            obs["normalization_seconds"]=time()-t
            obs["area_term"]=area; obs["tau0_real"]=real(tau0);obs["tau0_imag"]=imag(tau0)
            obs["central_from_tau"]=real(ctau)
            tm=TNRKit.CFTTransferMatrix(ta,tb,shape)
            tau=TNRKit.modular_parameter(tm,tau0)
            obs["tau_tm_real"]=real(tau);obs["tau_tm_imag"]=imag(tau)
            isfinite(tau)&&imag(tau)>0 || error("Invalid tau $tau")
            denominator=imag(tau)-imag(tau0)*shape[1]*shape[2]/4
            obs["central_denominator"]=denominator
            qs = S==U1Irrep ? [U1Irrep(q) for q in [0,1,-1,2,-2,3,-3]] :
                 [CU1Irrep(0,0),CU1Irrep(1,2),CU1Irrep(2,2),CU1Irrep(3,2),CU1Irrep(0,1)]
            lambda0=0.0+0.0im
            for q in qs
                charge=S==U1Irrep ? Float64(q.charge) : Float64(q.j)
                parity=S==U1Irrep ? -1 : q.s
                initial=ones(Float64,domain(tm) ← Vect[S](q=>1))
                dim(initial)>0 || begin
                    push!(sectorsinfo,Dict("charge"=>charge,"parity"=>parity,"status"=>"absent"));continue
                end
                requested=min(nh,dim(initial));t=time()
                println("  SECTOR $q dim=$(dim(initial))");flush(stdout)
                vals,vecs,info=TNRKit.eigsolve(x->tm(x),initial,requested,:LM;
                    krylovdim=40,maxiter=100,tol=1e-12,verbosity=0)
                order=sortperm(abs.(vals);rev=true)
                isempty(order) && error("No eigenvalues in $q")
                q==first(qs) && (lambda0=ComplexF64(vals[first(order)]))
                abs(lambda0)>0 || error("Vacuum eigenvalue absent/zero")
                info.converged<requested && (obs["status"]="partial_convergence")
                for (rank,i) in enumerate(first(order,min(requested,length(order))))
                    lam=ComplexF64(vals[i]);nx=norm(vecs[i]);fx=eltype(vecs[i]) <: Real ? tm(vecs[i]) : tm(real(vecs[i])) + im*tm(imag(vecs[i]))
                    res=norm(fx-vals[i]*vecs[i])/nx
                    x=-log(abs(lam/lambda0))/(2pi*imag(tau))
                    push!(rows,Dict("charge"=>charge,"parity"=>parity,"rank"=>rank,
                        "lambda_real"=>real(lam),"lambda_imag"=>imag(lam),"x"=>x,
                        "phase"=>angle(lam/lambda0),"residual_absolute"=>res,
                        "residual_relative"=>res/max(abs(lam),floatmin(Float64)),
                        "requested"=>requested,"converged"=>info.converged))
                    if rank==1 && charge!=0
                        obs[charge>0 ? "x$(Int(charge))" : "xm$(Int(-charge))"]=x
                    end
                end
                push!(sectorsinfo,Dict("charge"=>charge,"parity"=>parity,"status"=>"measured",
                    "dimension"=>dim(initial),"seconds"=>time()-t,"requested"=>requested,
                    "converged"=>info.converged,"numops"=>info.numops))
                println("  DONE $q seconds=$(time()-t)");flush(stdout)
            end
            if abs(denominator)>1e-10
                cc=6/pi/denominator*log(lambda0)
                obs["central_charge"]=real(cc);obs["central_imag"]=imag(cc)
            else
                obs["central_charge"]=NaN;obs["central_imag"]=NaN
                obs["central_status"]="normalization_tube_zero_denominator"
            end
            obs["K_raw"]=haskey(obs,"x1") ? 1/(4obs["x1"]) : NaN
        end
    catch e
        (e isa InterruptException || e isa Base.IOError || e isa SystemError) && rethrow()
        obs["status"]="exception";obs["error"]=sprint(showerror,e,catch_backtrace())
    end
    obs["seconds"]=time()-start
    warnings=String(take!(warningbuffer))
    atomic(output,Dict("observables"=>obs,"spectra"=>rows,"sectors"=>sectorsinfo,"warnings"=>warnings))
    println("FINISH $sym c=$cv RG=$n $label $(obs["status"]) seconds=$(obs["seconds"]) x3=$(get(obs,"x3",NaN))");flush(stdout)
end
const SHAPE=SHAPES["small"]
const CS=[.5,1.,1.4,1.7,1.9,1.98]
function extract(scheme,S;nh=6)
    area=TNRKit.area_term(scheme.TA,scheme.TB)
    isfinite(area)&&area>0 || error("Nonpositive/nonfinite area normalization: $area")
    ta,tb=scheme.TA/area^(1/4),scheme.TB/area^(1/4)
    tau0,ctau=TNRKit.extract_tau_and_c(ta,tb;fast=true)
    tm=TNRKit.CFTTransferMatrix(ta,tb,SHAPE)
    tau=TNRKit.modular_parameter(tm,tau0)
    denom=imag(tau)-imag(tau0)*SHAPE[1]*SHAPE[2]/4
    isfinite(tau)&&imag(tau)>0&&abs(denom)>1e-10 || error("Invalid CFT geometry")
    eigen=Dict{S,Any}()
    for q in sectors(fuse(domain(tm)))
        initial=ones(Float64,domain(tm) ← Vect[S](q=>1))
        requested=min(nh,dim(initial))
        vals,vecs,info=TNRKit.eigsolve(x->tm(x),initial,requested,:LM;
            krylovdim=40,maxiter=100,tol=1e-12,verbosity=0)
        order=sortperm(abs.(vals);rev=true)
        residuals=[norm(tm(vecs[i])-vals[i]*vecs[i])/max(norm(vecs[i]),eps()) for i in eachindex(vals)]
        eigen[q]=(;vals=ComplexF64.(vals),residuals,order,requested,converged=info.converged)
        info.converged<requested && @warn "Partial eigensolver convergence" sector=q requested converged=info.converged
    end
    q0=S==U1Irrep ? U1Irrep(0) : CU1Irrep(0,0)
    haskey(eigen,q0)||error("Vacuum sector unavailable")
    e0=eigen[q0];lambda0=e0.vals[first(e0.order)]
    cc=6/pi/denom*log(lambda0)
    rows=Dict{String,Any}[]
    for q in sort(collect(keys(eigen));by=string)
        e=eigen[q]
        for (rank,i) in enumerate(e.order)
            lambda=e.vals[i];x=-log(abs(lambda/lambda0))/(2pi*imag(tau))
            push!(rows,Dict("charge"=>(S==U1Irrep ? Float64(q.charge) : Float64(q.j)),
                "parity"=>(S==U1Irrep ? -1 : q.s),"rank"=>rank,"x"=>x,
                "lambda_real"=>real(lambda),"lambda_imag"=>imag(lambda),
                "lambda_ratio_phase"=>angle(lambda/lambda0),"residual"=>e.residuals[i],
                "selected"=>rank<=e.requested,"requested"=>e.requested,"converged"=>e.converged))
        end
    end
    getx(q,p,r)=begin
        selected=filter(t->t["charge"]==q&&t["parity"]==p&&t["rank"]==r,rows)
        isempty(selected) ? NaN : only(selected)["x"]
    end
    parity=S==U1Irrep ? -1 : 2
    obs=Dict{String,Any}("central_charge"=>real(cc),"central_imag"=>imag(cc),
        "tau_real"=>real(tau0),"tau_imag"=>imag(tau0),
        "tau_tm_real"=>real(tau),"tau_tm_imag"=>imag(tau),
        "area_term"=>area,"central_from_tau"=>real(ctau))
    for q in 1:3
        obs["x$q"]=getx(q,parity,1)
        obs["xm$q"]=S==U1Irrep ? getx(-q,-1,1) : NaN
    end
    obs["neutral_second"]=getx(0,S==U1Irrep ? -1 : 0,2)
    obs["neutral_third_or_odd"]=S==U1Irrep ? getx(0,-1,3) : getx(0,1,1)
    return obs,rows
end
function measure_small(row)
    s=row["symmetry"];c=parse(Float64,row["physical_c"]);n=parse(Int,row["rg_step"])
    phase=row["phase"];unavailable=false
    Random.seed!(1000*findfirst(==(c),CS)+n)
    k=pi/(2acos(1-c*c/2))
    obs=Dict{String,Any}("symmetry"=>s,"vertex_c"=>c,"rg_step"=>n,"phase"=>phase,
        "K_exact"=>k,"x1_exact"=>1/(4k),"x2_exact"=>1/k,"x3_exact"=>9/(4k),
        "status"=>"ok","warning_count"=>0,"cft_seconds"=>0.,"error"=>"")
    for key in ("central_charge","central_imag","x1","x2","x3","xm1","xm2","xm3",
                "neutral_second","neutral_third_or_odd","tau_real","tau_imag",
                "tau_tm_real","tau_tm_imag","area_term","central_from_tau")
        obs[key]=NaN
    end
    rows=Dict{String,Any}[];buffer=IOBuffer();t=time_ns()
    if unavailable
        obs["status"]="rg_unavailable"
        obs["error"]=read(terminal(s,c),String)
    else
        try
            values,rows=with_logger(SimpleLogger(buffer,Logging.Warn)) do
                extract(check_input(row).scheme,s=="U1Irrep" ? U1Irrep : CU1Irrep)
            end
            merge!(obs,values)
            required=s=="U1Irrep" ? ("central_charge","central_imag","tau_real","tau_imag","x1","x2","xm1","xm2") :
                                   ("central_charge","central_imag","tau_real","tau_imag","x1","x2")
            all(key->isfinite(obs[key]),required)||(obs["status"]="nonfinite")
        catch e
            (e isa InterruptException || e isa Base.IOError || e isa SystemError) && rethrow()
            obs["status"]="exception";obs["error"]=sprint(showerror,e,catch_backtrace())
        end
        obs["cft_seconds"]=(time_ns()-t)/1e9
    end
    warning=String(take!(buffer))
    obs["warning_count"]=count(line->occursin("Warning:",line),split(warning,'\n'))
    !isempty(warning)&&obs["status"]=="ok"&&(obs["status"]="warning")
    atomic_toml(joinpath(OUT,row["output_record"]),Dict("observables"=>obs,"spectra"=>rows,"warnings"=>warning))
    println("CFT $s c=$c step=$n phase=$phase cc=$(obs["central_charge"]) status=$(obs["status"])");flush(stdout)
    return obs
end
measure_record(row)=row["profile"]=="legacy_small" ? measure_small(row) : measure_wide(row)

for row in JOB["records"]
    println("START ",row["record_id"]," ",row["profile"]);flush(stdout)
    measure_record(row);GC.gc()
end
