# Same exact symmetry-twist contraction as the validated readout.
function pi_rotation(V)
    z=id(Float64,V)
    for q in sectors(V)
        isinteger(q.j)||error("This CU1-preserving pi-flux construction requires integer-charge blocked legs")
        block(z,q).*=isodd(Int(q.j)) ? -1 : 1
    end
    @assert norm(z*z-id(Float64,V))<1e-12
    z
end
function multiply_first(A,z)
    @tensor out[-1 -2;-3 -4] := z[-1;a]*A[a -2;-3 -4]
    out
end
function flux_action(A,B,Az,Bz,x)
    @tensor begin
        y[-1 -2;-3] := Az[c -1;1 m]*x[1 2;-3]*B[m -2;2 c]
        out[-1 -2;-3] := Bz[c -1;1 m]*y[1 2;-3]*A[m -2;2 c]
    end
    out
end
function full_spectrum(A,B;flux=false)
    xs=domain(A)[1] ⊗ domain(B)[1]
    Az=flux ? multiply_first(A,pi_rotation(codomain(A)[1])) : A
    Bz=flux ? multiply_first(B,pi_rotation(codomain(B)[1])) : B
    rows=Dict{String,Any}[];maxres=0.;action_error=0.;move_error=0.
    for q in sectors(fuse(xs))
        p=zeros(ComplexF64,xs←Vect[CU1Irrep](q=>1));N=length(p.data);N==0&&continue
        N<=2048||error("Dense block too large")
        M=zeros(ComplexF64,N,N)
        for j in 1:N
            fill!(p.data,0);p.data[j]=1
            M[:,j]=flux_action(A,B,Az,Bz,p).data
        end
        E=eigen(M)
        for j in eachindex(E.values)
            mu=E.values[j];v=E.vectors[:,j]
            res=norm(M*v-mu*v)/max(norm(M)*norm(v),eps())
            maxres=max(maxres,res)
            push!(rows,Dict("charge"=>Float64(q.j),"C_parity"=>q.j==0 ? (q.s==0 ? 1 : -1) : 0,
                "sector"=>string(q),"degeneracy"=>dim(q),"mu_real"=>real(mu),"mu_imag"=>imag(mu),"residual"=>res))
        end
        randn!(MersenneTwister(20261107),p.data);p/=norm(p)

    end
    sort!(rows;by=r->-abs(complex(r["mu_real"],r["mu_imag"])))
    Dict("levels"=>rows,"full_dimension"=>sum(r["degeneracy"] for r in rows),
        "matrix_scaled_maximum_residual"=>maxres,"implementation"=>"explicit two-row contraction for registered TNRKit 0.7.0","shape"=>[2.,2.,0.],"spatial_pi_flux"=>flux)
end
