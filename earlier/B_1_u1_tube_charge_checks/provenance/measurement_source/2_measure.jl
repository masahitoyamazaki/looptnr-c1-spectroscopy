# CFT-only postprocessing. Never advances RG or writes a tensor checkpoint.
import Pkg
const ROOT = normpath(joinpath(@__DIR__, "../../results/perturbation_chi40_cu1_u1_20261021"))
const SOURCE = normpath(joinpath(ROOT,"../flow_chi40_cu1_u1_20261020/results"))
Pkg.activate(joinpath(SOURCE,"source_environment"))
using TNRKit, TensorKit, Serialization, LinearAlgebra, TOML, Dates, Printf, SHA, Logging
BLAS.set_num_threads(1)
const SHAPES = Dict("small"=>[sqrt(2.),sqrt(2.),0.],
    "wide"=>[sqrt(2.),2sqrt(2.),0.], "twisted"=>[1.,4.,1.],
    "square"=>[2.,2.,0.])
function atomic(path,d)
    mkpath(dirname(path)); isfile(path) && error("Refusing overwrite $path")
    open(path*".tmp","w") do io; TOML.print(io,d); end
    mv(path*".tmp",path)
end
function measure(sym,cv,n,label; nh=6, suffix="")
    output=joinpath(ROOT,"results",sym,"c_$cv",@sprintf("step_%03d_%s%s.toml",n,label,suffix))
    isfile(output) && return
    path=joinpath(SOURCE,sym,"c_$cv","checkpoints",@sprintf("step_%03d.jls",n))
    shape=SHAPES[label]; S=sym=="CU1Irrep" ? CU1Irrep : U1Irrep
    k=pi/(2acos(1-cv^2/2))
    obs=Dict{String,Any}("symmetry"=>sym,"vertex_c"=>cv,"rg_step"=>n,"tube"=>label,
        "shape"=>shape,"chi"=>40,"levels"=>nh,"K_exact"=>k,"status"=>"ok",
        "source"=>relpath(path,ROOT),"julia_version"=>string(VERSION),"utc"=>string(now(UTC)))
    rows=Dict{String,Any}[]; sectorsinfo=Dict{String,Any}[]; warningbuffer=IOBuffer()
    start=time(); println("START $sym c=$cv RG=$n $label");flush(stdout)
    try
        d=deserialize(path)
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
if abspath(PROGRAM_FILE)==@__FILE__
    # TSV columns: symmetry, physical c, RG step, tube, optional level count and suffix.
    for line in eachline(ARGS[1])
        startswith(line,"#") && continue
        p=split(strip(line),'\t');length(p)>=4 || continue
        measure(p[1],parse(Float64,p[2]),parse(Int,p[3]),p[4];
            nh=length(p)>=5 ? parse(Int,p[5]) : 6,suffix=length(p)>=6 ? p[6] : "")
        GC.gc()
    end
end
