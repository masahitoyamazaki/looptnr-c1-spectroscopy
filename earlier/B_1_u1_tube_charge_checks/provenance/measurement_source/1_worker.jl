# This worker is launched by campaign.py, which holds the process lock and runs
# the unchanged shared reservation guard. It does not create reservations.
import Pkg
using TOML, Serialization, Dates, Printf, Random, LinearAlgebra, Logging
const MODE = isempty(ARGS) ? "help" : ARGS[1]
const OUT = length(ARGS)>1 ? abspath(ARGS[2]) : joinpath(@__DIR__,"results")
const ENVIRONMENT = MODE=="smoke" ? normpath(joinpath(@__DIR__,"../../from_official")) : joinpath(OUT,"source_environment")
Pkg.activate(ENVIRONMENT)
using TNRKit, TensorKit
const CFG = TOML.parsefile(MODE=="smoke" ? joinpath(@__DIR__,"settings.toml") : joinpath(OUT,"settings.toml"))
const CS = Float64.(CFG["physical_cs"])
const SYMS = String.(CFG["symmetries"])
const SHAPE = Float64.(CFG["shape"])
BLAS.set_num_threads(CFG["blas_threads"])
symtype(s) = s=="CU1Irrep" ? CU1Irrep : s=="U1Irrep" ? U1Irrep : error("Unknown symmetry $s")
weights(s,c) = s=="CU1Irrep" ? (;a=1.,b=1.,c) : (;a=1.,b=c,c=1.)
pointdir(s,c) = joinpath(OUT,s,"c_$c")
checkpoint(s,c,n) = joinpath(pointdir(s,c),"checkpoints",@sprintf("step_%03d.jls",n))
measurement(s,c,n) = joinpath(pointdir(s,c),"measurements",@sprintf("step_%03d.toml",n))
terminal(s,c) = joinpath(pointdir(s,c),"terminal.toml")
nowstr() = string(now(UTC))

function atomictoml(path, data; replace=false)
    mkpath(dirname(path))
    !replace && isfile(path) && error("Refusing overwrite: $path")
    open(path*".tmp","w") do io
        TOML.print(io,data)
    end
    mv(path*".tmp",path;force=replace)
end
function save_checkpoint(path,data)
    mkpath(dirname(path))
    isfile(path) && error("Refusing overwrite: $path")
    serialize(path*".tmp",data)
    mv(path*".tmp",path)
end
function saved_steps(s,c)
    dir=joinpath(pointdir(s,c),"checkpoints")
    !isdir(dir) && return Int[]
    steps=sort([parse(Int,match(r"step_(\d+)\.jls",f).captures[1])
                for f in readdir(dir) if occursin(r"^step_\d+\.jls$",f)])
    isempty(steps) || steps==collect(0:maximum(steps)) || error("Noncontiguous checkpoints for $s c=$c")
    return steps
end
function authorize()
    active=TOML.parsefile(joinpath(OUT,"active.toml"))
    get(ENV,"SIXVERTEX_LAUNCH_TOKEN","")==active["token"] || error("Use campaign.py run")
    ccall(:kill,Cint,(Cint,Cint),active["pid"],0)==0 || error("Launcher is not alive")
    @assert isfile(joinpath(OUT,"guards_ready.toml"))
    @assert TOML.parsefile(joinpath(OUT,"guards_ready.toml"))["token"]==active["token"]
    expected=TOML.parsefile(joinpath(OUT,"runtime_expected.toml"))["loop_parameters"]
    replace(repr(LoopParameters()),"TNRKit."=>"")==replace(expected,"TNRKit."=>"") ||
        error("Actual LoopParameters differ from the shared reservation")
end
struct InvalidNormalization <: Exception
    message::String
end
Base.showerror(io::IO,e::InvalidNormalization)=print(io,e.message)

function rg()
    authorize()
    runtime=joinpath(OUT,"runtime.toml")
    if isfile(runtime)
        TOML.parsefile(runtime)["julia_version"]==string(VERSION) || error("Julia version changed since initial run")
    else
        atomictoml(runtime,Dict("julia_version"=>string(VERSION),
            "loop_parameters"=>repr(LoopParameters()),"blas_threads"=>BLAS.get_num_threads(),
            "julia_threads"=>Threads.nthreads(),"created_utc"=>nowstr()))
    end
    for target in (CFG["fixed_step"],CFG["steps"]), s in SYMS, (i,c) in enumerate(CS)
        isfile(terminal(s,c)) && continue # numerical failures are not silently retried
        steps=saved_steps(s,c)
        start=isempty(steps) ? -1 : maximum(steps)
        # A process can stop after the checkpoint rename, before its timing
        # record or terminal marker. The checkpoint remains authoritative.
        recordpath=joinpath(pointdir(s,c),"rg_records",@sprintf("step_%03d.toml",start))
        if start>=0 && !isfile(recordpath)
            tail=deserialize(checkpoint(s,c,start))
            @assert tail.chi==CFG["chi"] && tail.symmetry==s && tail.vertex_c==c
            atomictoml(recordpath,Dict("symmetry"=>s,"physical_c"=>c,"step"=>start,
                "seconds"=>tail.rg_seconds,"normalization"=>last(tail.normalization_history),
                "beta_f"=>tail.beta_f))
        end
        if start==CFG["steps"]
            atomictoml(terminal(s,c),Dict("status"=>"complete","last_saved_step"=>start,
                "time"=>nowstr(),"recovered_after_final_checkpoint"=>true))
            continue
        end
        start>=target && continue
        history=Float64[]
        if start<0
            Random.seed!(CFG["seed"]+i)
            scheme=LoopTNR(sixvertex(symtype(s);weights(s,c)...))
        else
            d=deserialize(checkpoint(s,c,start))
            @assert d.chi==CFG["chi"] && d.symmetry==s && d.vertex_c==c
            @assert d.weights==weights(s,c) && d.rg_step==start
            scheme=d.scheme;append!(history,d.normalization_history)
            copy!(Random.default_rng(),d.rng_state)
            # Match the chi28 campaign's explicit phase-boundary reseeding.
            start==CFG["fixed_step"] && Random.seed!(CFG["seed"]+i)
        end
        step=Ref(start)
        tick=Ref(time_ns())
        function record!(current)
            step[]+=1
            n=Float64(finalize!(current))
            if !(isfinite(n)&&n>0&&isfinite(norm(current.TA))&&isfinite(norm(current.TB)))
                throw(InvalidNormalization("normalization=$n; normTA=$(norm(current.TA)); normTB=$(norm(current.TB)); attempted step=$(step[])"))
            end
            push!(history,n)
            seconds=(time_ns()-tick[])/1e9
            data=(;scheme=current,normalization_history=copy(history),vertex_c=c,
                   chi=CFG["chi"],rg_step=step[],symmetry=s,weights=weights(s,c),
                   config=(;weights(s,c)...,chi=CFG["chi"],steps=CFG["steps"],shape=SHAPE),
                   rng_state=copy(Random.default_rng()),rg_seconds=seconds,
                   beta_f=free_energy(history,1.;scalefactor=2.))
            save_checkpoint(checkpoint(s,c,step[]),data)
            atomictoml(joinpath(pointdir(s,c),"rg_records",@sprintf("step_%03d.toml",step[])),
                Dict("symmetry"=>s,"physical_c"=>c,"step"=>step[],"seconds"=>seconds,
                     "normalization"=>n,"beta_f"=>data.beta_f);replace=true)
            println("RG $s c=$c step=$(step[]) seconds=$seconds");flush(stdout)
            tick[]=time_ns()
            return n
        end
        try
            mkpath(pointdir(s,c))
            open(joinpath(pointdir(s,c),"rg_warnings.log"),"a") do io
                with_logger(SimpleLogger(io,Logging.Warn)) do
                    run!(scheme,truncrank(CFG["chi"]),maxiter(target-max(start,0)),
                         Finalizer(record!,Float64);finalize_beginning=start<0,verbosity=0)
                end
            end
            @assert step[]==target
            if target==CFG["steps"]
                atomictoml(terminal(s,c),Dict("status"=>"complete","last_saved_step"=>step[],"time"=>nowstr()))
            end
        catch e
            (e isa InterruptException || e isa Base.IOError || e isa SystemError) && rethrow()
            laststep=maximum(saved_steps(s,c);init=-1)
            atomictoml(terminal(s,c),Dict("status"=>(e isa InvalidNormalization ? "numerical_failure" : "rg_exception"),
                "last_saved_step"=>laststep,"attempted_step"=>laststep+1,
                "error"=>sprint(showerror,e,catch_backtrace()),"time"=>nowstr()))
            println(stderr,"Stopped $s c=$c after step=$laststep: ",sprint(showerror,e));flush(stderr)
        end
    end
    atomictoml(joinpath(OUT,"RG_FINISHED.toml"),Dict("time"=>nowstr());replace=true)
end

# Same two-leg tube and solver settings as the chi28 campaigns.
function extract(scheme,S;nh=CFG["levels"])
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
function measure(s,c,n,phase;unavailable=false)
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
                extract(deserialize(checkpoint(s,c,n)).scheme,symtype(s))
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
    atomictoml(measurement(s,c,n),Dict("observables"=>obs,"spectra"=>rows,"warnings"=>warning))
    println("CFT $s c=$c step=$n phase=$phase cc=$(obs["central_charge"]) status=$(obs["status"])");flush(stdout)
    return obs
end
function cft()
    authorize()
    done=Dict{Tuple{String,Float64,Int},Any}()
    for s in SYMS,c in CS,n in 1:CFG["steps"]
        path=measurement(s,c,n)
        isfile(path)&&(done[(s,c,n)]=TOML.parsefile(path)["observables"])
    end
    function batch(todo,phase)
        filter!(p->!haskey(done,p),todo)
        while !isempty(todo)
            progress=false
            for (s,c,n) in copy(todo)
                if isfile(checkpoint(s,c,n))
                    done[(s,c,n)]=measure(s,c,n,phase)
                elseif isfile(terminal(s,c))
                    term=TOML.parsefile(terminal(s,c))
                    n<=term["last_saved_step"] && error("Missing checkpoint inside saved range")
                    done[(s,c,n)]=measure(s,c,n,phase;unavailable=true)
                else
                    continue
                end
                filter!(!=((s,c,n)),todo);progress=true
            end
            isfile(joinpath(OUT,"RG_FATAL.toml"))&&error("RG worker failed; see rg.log")
            if !progress
                isfile(joinpath(OUT,"RG_FINISHED.toml"))&&error("Missing checkpoint after RG finished")
                sleep(2)
            end
        end
        atomictoml(joinpath(OUT,"CFT_$(uppercase(phase))_DONE.toml"),Dict("time"=>nowstr());replace=true)
    end
    batch([(s,c,CFG["fixed_step"]) for s in SYMS for c in CS],"fixed15")
    interval=CFG["coarse_interval"]
    coarse=sort(unique(vcat(collect(interval:interval:CFG["steps"]),CFG["fixed_step"],CFG["steps"])))
    batch([(s,c,n) for n in coarse for s in SYMS for c in CS],"coarse")
    adaptive=Tuple{String,Float64,Int}[];intervals=Dict{String,Any}[]
    err(r)=maximum(vcat([abs(r["central_charge"]-1),abs(r["x1"]*4r["K_exact"]-1),abs(r["x2"]*r["K_exact"]-1)],
        r["symmetry"]=="U1Irrep" ? [abs(r["xm1"]*4r["K_exact"]-1),abs(r["xm2"]*r["K_exact"]-1)] : Float64[]))
    for s in SYMS,c in CS,i in 1:length(coarse)-1
        l,r=coarse[i],coarse[i+1];a,b=done[(s,c,l)],done[(s,c,r)]
        e1,e2=err(a),err(b)
        crossing=any(v->min(e1,e2)<v<=max(e1,e2),(0.01,0.05,0.1,0.5))
        bad=a["status"]!="ok"||b["status"]!="ok"||!isfinite(e1)||!isfinite(e2)
        if bad||crossing||abs(e2-e1)>.05
            push!(intervals,Dict("symmetry"=>s,"physical_c"=>c,"left"=>l,"right"=>r,
                "reason"=>(bad ? "flagged" : crossing ? "threshold_crossing" : "error_change_gt_0.05")))
            append!(adaptive,[(s,c,n) for n in l+1:r-1])
        end
    end
    atomictoml(joinpath(OUT,"adaptive_intervals.toml"),Dict("intervals"=>intervals);replace=true)
    batch(unique(adaptive),"adaptive")
    batch([(s,c,n) for n in 1:CFG["steps"] for s in SYMS for c in CS],"fill")
    atomictoml(joinpath(OUT,"CFT_DONE.toml"),Dict("time"=>nowstr());replace=true)
end

function smoke()
    # Read-only regression against previously validated chi28 checkpoints: no RG,
    # no shared reservation, no new tensors, no chi40 campaign output.
    for (s,folder) in (("CU1Irrep","chi28_flow_20261020"),("U1Irrep","chi28_u1_flow_20261020"))
        path=joinpath(@__DIR__,"..",folder,"checkpoints/c_1.0/step_015.jls")
        d=deserialize(path);obs,rows=extract(d.scheme,symtype(s))
        a=TNRKit.area_term(d.scheme.TA,d.scheme.TB)^(1/4)
        ta,tb=d.scheme.TA/a,d.scheme.TB/a
        tau,=TNRKit.extract_tau_and_c(ta,tb;fast=true)
        ref=TNRKit.spec(ta,tb,SHAPE,tau;Nh=CFG["levels"])
        @assert isapprox(obs["central_charge"],real(ref.central_charge);atol=1e-8)
        for q in (s=="U1Irrep" ? [-3,-2,-1,1,2,3] : [1,2,3])
            sector=s=="U1Irrep" ? U1Irrep(q) : CU1Irrep(q,2)
            key=q<0 ? "xm$(abs(q))" : "x$q"
            @assert isapprox(obs[key],minimum(real.(ref.scaling_dimensions[sector]));atol=1e-8)
        end
        # TOML is the restart-safe measurement record format, including NaN.
        io=IOBuffer();TOML.print(io,Dict("observables"=>obs,"spectra"=>rows))
        roundtrip=TOML.parse(String(take!(io)))
        @assert isequal(roundtrip["observables"],obs)
        println("PASS $s: central charge, signed sectors and TOML roundtrip")
    end
    r=copy(Random.default_rng());x=rand()
    io=IOBuffer();serialize(io,(;rng_state=r,weights=weights("U1Irrep",1.4)))
    seekstart(io);saved=deserialize(io);copy!(Random.default_rng(),saved.rng_state)
    @assert rand()==x && saved.weights==(;a=1.,b=1.4,c=1.)
    println("PASS serialized RNG restore; default LoopParameters = ",repr(LoopParameters()))
end
if MODE=="smoke"
    smoke()
elseif MODE in ("rg","cft")
    try
        MODE=="rg" ? rg() : cft()
    catch e
        atomictoml(joinpath(OUT,uppercase(MODE)*"_FATAL.toml"),
                  Dict("time"=>nowstr(),"error"=>sprint(showerror,e,catch_backtrace()));replace=true)
        rethrow()
    end
else
    error("Use campaign.py run, or worker.jl smoke")
end
