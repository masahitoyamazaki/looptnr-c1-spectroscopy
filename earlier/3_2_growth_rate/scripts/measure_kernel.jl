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

function measure_record(row)
    d=check_input(row);started=time();n=parse(Int,row["rg_step"]);K=parse(Float64,row["K"])
    profile=row["profile"]
    obs=Dict{String,Any}("status"=>"ok","rg_step"=>n,"chi"=>40,"symmetry"=>"CU1Irrep",
        "K_target"=>K,"vertex_c"=>parse(Float64,row["physical_c"]),"source"=>row["checkpoint"],
        "checkpoint_sha256"=>row["catalog_sha256"],"match_method"=>row["match_method"],
        "worker_sha256"=>WORKER_SHA,"manifest_sha256"=>MANIFEST_SHA,"julia_version"=>string(VERSION),
        "shape"=>SHAPES["wide"],"measurement_profile"=>profile)
    result=Dict{String,Any}("observables"=>obs)
    levels=Dict(CU1Irrep(0,0)=>6,CU1Irrep(0,1)=>(profile=="o2_log" ? 4 : 6),CU1Irrep(1,2)=>3)
    try
        cft=CFTData(d.scheme;Nh=levels,shape=SHAPES["wide"])
        sv=cft.scaling_dimensions
        dims=Dict(string(q)=>sort(real.(sv[q])) for q in keys(sv.structure))
        result["dimensions"]=dims
        obs["central_charge"]=real(cft.central_charge)
        obs["tau0_real"]=real(cft.modular_parameter);obs["tau0_imag"]=imag(cft.modular_parameter)
        mag=dims[string(CU1Irrep(1,2))]
        isempty(mag)&&error("No charge-1 level")
        obs["x_m1"]=mag[1];obs["K_m"]=1/(4mag[1])
        if profile=="o2_g2"
            even=dims[string(CU1Irrep(0,0))];odd=dims[string(CU1Irrep(0,1))]
            ne,no=K<1 ? (4,5) : (2,3)
            length(even)>=ne&&length(odd)>=no||error("Insufficient levels")
            all(isfinite,vcat(even[1:ne],odd[1:no],mag[1]))||error("Nonfinite levels")
            result["g2"]=Dict(vcat(["even_$i"=>(i<=length(even) ? even[i] : NaN) for i in 1:6],
                                  ["odd_$i"=>(i<=length(odd) ? odd[i] : NaN) for i in 1:6]))
        end
    catch e
        e isa InterruptException&&rethrow()
        obs["status"]="exception";obs["error"]=sprint(showerror,e,catch_backtrace())
    end
    obs["seconds"]=time()-started
    atomic_toml(joinpath(OUT,row["output_record"]),result)
    println("DONE ",row["record_id"]," status=",obs["status"]," seconds=",obs["seconds"]);flush(stdout)
end

for row in JOB["records"]
    println("START ",row["record_id"]," ",row["profile"]);flush(stdout)
    measure_record(row);GC.gc()
end
