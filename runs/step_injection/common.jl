# AI-generated experimental utilities. Physical claims require the recorded benchmarks.
module SymmetryValidation
using TNRKit, TensorKit, LinearAlgebra, TOML, SHA, Serialization, Random
export physical_tensor, native_tensor, atomic_toml, provenance, ROOT, OUT, torus2,
       measure_loop, c_only_transform, exact_small_spectrum, canonical_area
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUT = joinpath(ROOT, "Cho", "output", "symmetry_validation_20261024")
const HADAMARD = [1.0 1.0; 1.0 -1.0]/sqrt(2.0)
function c_only_transform(A)
    H = kron(HADAMARD, HADAMARD)
    reshape(H * reshape(A,4,4) * H',2,2,2,2)
end
function physical_tensor(internal::String, K::Real; a=1.0,b=1.0,field=0.0,vertex_c=nothing,transverse=0.0)
    cv = vertex_c===nothing ? round(2sin(pi/(4K)); digits=12) : Float64(vertex_c)
    transverse==0 || internal in ("dense","C") || error("Transverse perturbation violates U1")
    if internal == "U1"
        T = sixvertex(U1Irrep; a, b=cv, c=b)
        if field != 0
            V=domain(T)[1]; D=zeros(Float64,V,V)
            for q in sectors(V); block(D,q) .= exp(field*Float64(q.charge)); end
            T = T * (id(V) ⊗ D)
        end
        return T
    elseif internal == "CU1"
        field==0 || error("A C-odd arrow field is forbidden by CU1")
        return sixvertex(CU1Irrep; a,b,c=cv)
    elseif internal in ("dense", "C")
        A=convert(Array,sixvertex(Trivial;a,b,c=cv))
        field==0 || error("Dense/C field adapters must be separately calibrated")
        if transverse!=0
            X=[0.0 1;1 0]; E=Matrix{Float64}(LinearAlgebra.I,2,2)
            A=reshape(reshape(A,4,4)*kron(E,exp(transverse*X)),2,2,2,2)
        end
        internal=="dense" && return TensorMap(A,ℂ^2⊗ℂ^2,ℂ^2⊗ℂ^2)
        V=Z2Space(0=>1,1=>1)
        return TensorMap(c_only_transform(A),V⊗V,V⊗V)
    end
    error("Unknown internal symmetry $internal")
end
function native_tensor(T; internal="auto")
    A=permute(T,(1,2,4,3)); V=space(A,1)
    A=reduce(⊗,[isomorphism(V,space(A,i)) for i in 1:4])*A
    if internal in ("dense","C")
        # U1 duality identifies opposite arrow charges. A generic dense/Z2
        # isomorphism does not know this physical charge-conjugation map.
        X=internal=="dense" ? [0.0 1;1 0] : [1.0 0;0 -1]
        D=convert(Array,A)
        @tensor B[i j k l] := D[i j a b]*X[k a]*X[l b]
        A=TensorMap(B,space(A))
    end
    return permute(A,(1,2,4,3))
end
function torus2(T)
    # Same 2x2 periodic cell as LoopTNR.finalize!, without normalization.
    A=convert(Array,T); B=permutedims(A,(1,2,4,3))
    @tensor z = B[1 2 3 4]*B[3 5 1 6]*B[7 4 8 2]*B[8 6 7 5]
    return real(z)
end
function atomic_toml(path,data)
    mkpath(dirname(path))
    open(path*".tmp","w") do io; TOML.print(io,data); end
    mv(path*".tmp",path;force=true)
end
function provenance()
    pkgroot=normpath(joinpath(dirname(pathof(TNRKit)),".."))
    files=vcat([joinpath(ROOT,"Cho","Project.toml"),joinpath(ROOT,"Cho","Manifest.toml")],
        [joinpath(@__DIR__,"common.jl")])
    return Dict{String,Any}("julia"=>string(VERSION),"tnrkit"=>string(pkgversion(TNRKit)),
        "tensorkit"=>string(pkgversion(TensorKit)),"git_commit"=>readchomp(`git -C $ROOT rev-parse HEAD`),
        "source_sha256"=>Dict(relpath(p,ROOT)=>bytes2hex(sha256(read(p))) for p in files),
        "tnrkit_source_sha256"=>Dict(relpath(joinpath(d,f),pkgroot)=>bytes2hex(sha256(read(joinpath(d,f))))
            for (d,_,fs) in walkdir(joinpath(pkgroot,"src")) for f in fs if endswith(f,".jl")),
        "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads())
end
function canonical_area(A,B;exact=false)
    x=zeros(Float64,domain(A)[1]⊗domain(B)[1])
    function f(v)
        @tensor fx[-1 -2] := A[c -1;1 m]*v[1 2]*B[m -2;2 c]
        @tensor ffx[-1 -2] := B[c -1;1 m]*fx[1 2]*A[m -2;2 c]
        return ffx
    end
    randn!(MersenneTwister(74103),x.data);x/=norm(x)
    vals,vecs,info=TNRKit.eigsolve(f,x,1,:LR;krylovdim=40,maxiter=100,tol=1e-12,verbosity=0)
    val=first(vals);v=first(vecs)
    residual=norm(f(v)-val*v)/(abs(val)*norm(v))
    info.converged>=1 && residual<1e-10 && real(val)>0 && abs(imag(val))<1e-10*abs(val) ||
        error("Canonical area eigenpair failed its quality gate")
    report=Dict{String,Any}("eigenvalue"=>real(val),"residual"=>residual,"converged"=>info.converged)
    if exact
        n=length(x.data);n<=512 || error("Exact area diagnostic exceeds dimension cap")
        M=zeros(Float64,n,n)
        for j in 1:n
            fill!(x,0);x.data[j]=1;M[:,j].=f(x).data
        end
        ev=eigvals(M);exactvalue=maximum(real.(ev))
        report["exact_eigenvalue"]=exactvalue
        report["exact_relative_error"]=abs(real(val)-exactvalue)/abs(exactvalue)
    end
    return report
end
function measure_loop(scheme; nh=6, selected=nothing)
    # Operates on copies; no measurement normalization changes the RG trajectory.
    s=deepcopy(scheme); A,B=s.TA,s.TB
    area=canonical_area(A,B);norm0=area["eigenvalue"]
    native_area=TNRKit.area_term(A,B)
    A=A/norm0^.25; B=B/norm0^.25
    I=sectortype(A); F=field(A); xspace=domain(B)⊗domain(B)
    qs=collect(sectors(fuse(xspace)))
    selected===nothing || (qs=filter(q->q in selected,qs))
    one(I) in qs || error("Vacuum sector absent")
    sort!(qs;by=q->q==one(I) ? "0" : "1"*string(q))
    spectra=Dict{String,Any}[]; diagnostics=Dict{String,Any}[]; lambda0=0.0+0im
    for (iq,q) in enumerate(qs)
        V=I==Trivial ? F^1 : Vect[I](q=>1)
        x=zeros(Float64,xspace←V); dim(x)>0 || continue
        # The all-ones vector can be exactly orthogonal to an eigenstate of an
        # unlabelled residual symmetry (observed in the C-only benchmark).
        # Use a reproducible local RNG; never consume the RG/global RNG state.
        randn!(MersenneTwister(2140+iq),x.data);x/=norm(x)
        requested=min(nh,dim(x))
        vals,vecs,info=TNRKit.eigsolve(v->TNRKit.MPO_action_2gates(A,B,v),x,requested,:LM;
            krylovdim=max(40,2requested+10),maxiter=100,tol=1e-12,verbosity=0)
        order=sortperm(abs.(vals);rev=true)
        q==one(I) && (lambda0=ComplexF64(vals[first(order)]))
        abs(lambda0)>0 || error("Invalid vacuum eigenvalue")
        nonzero=filter(j->abs(vals[j])>=1e-12,order)
        push!(diagnostics,Dict("sector"=>string(q),"vector_dimension"=>dim(x),
            "requested"=>requested,"converged"=>info.converged,"returned"=>length(vals),
            "resolved_nonzero"=>length(nonzero),"null_threshold_absolute"=>1e-12))
        for (rank,j) in enumerate(first(nonzero,min(length(nonzero),requested)))
            val=vals[j]; v=vecs[j]
            f=TNRKit.MPO_action_2gates(A,B,v)
            res=norm(f-val*v)/max(norm(v)*abs(val),eps(Float64))
            push!(spectra,Dict("sector"=>string(q),"rank"=>rank,
                "x"=>-log(abs(val/lambda0))/pi,"phase"=>angle(val/lambda0),
                "lambda_real"=>real(val),"lambda_imag"=>imag(val),"residual"=>res,
                "converged"=>info.converged,"requested"=>requested))
        end
    end
    # v0.7 geometry: Im tau=1/2, area=4. No anisotropy correction is hidden here.
    return Dict{String,Any}("central_charge"=>real(-12/pi*log(lambda0)),
        "shape"=>[sqrt(2.0),2sqrt(2.0),0.0],"tau_imag"=>0.5,
        "geometry_convention"=>"TNRKit-0.7-fixed-tau","initial_vector"=>"local-MersenneTwister-2140-plus-sector-index",
        "canonical_area"=>area,"native_all_ones_area"=>native_area,
        "spectra"=>spectra,"sector_diagnostics"=>diagnostics)
end
function exact_small_spectrum(scheme;max_dimension=512)
    s=deepcopy(scheme); A,B=s.TA,s.TB
    scale=canonical_area(A,B)["eigenvalue"]^.25;A=A/scale;B=B/scale
    I=sectortype(A);F=field(A);xs=domain(B)⊗domain(B)
    result=Dict{String,Vector{ComplexF64}}()
    for q in sectors(fuse(xs))
        V=I==Trivial ? F^1 : Vect[I](q=>1)
        v=zeros(Float64,xs←V);n=length(v.data)
        n<=max_dimension || error("Exact diagnostic dimension exceeds $max_dimension")
        M=zeros(Float64,n,n)
        for j in 1:n
            fill!(v,0);v.data[j]=1
            M[:,j].=TNRKit.MPO_action_2gates(A,B,v).data
        end
        vals=ComplexF64.(eigvals(M))
        result[string(q)]=sort(filter(z->abs(z)>=1e-12,vals);by=abs,rev=true)
    end
    return result
end
end
