include("measure.jl")
include("control/spin_reference.jl")
rows=Dict{String,Any}[]
for h in (0.,.01,.02)
    T=upright_dimer_tensor(-.5,.125,h)
    for width in (2,4)
        tm=TNRKit.CFTTransferMatrix(T,T,[1.,Float64(width),1.])
        rv=spin_row(-.5,.125,h,2width)
        refs=Dict{Int,Float64}();actual=Dict{Int,Float64}()
        for m in 0:3
            m>width && continue
            inds=[s+1 for s in 0:(2^(2width)-1) if count_ones(s)==width+m]
            refs[m]=maximum(abs.(eigvals(rv[inds,inds])))
            sectors_m=m==0 ? (CU1Irrep(0,0),CU1Irrep(0,1)) : (CU1Irrep(m,2),)
            vals=Float64[]
            for q in sectors_m
                v=zeros(Float64,domain(tm)←Vect[CU1Irrep](q=>1));N=length(v.data);N==0 && continue
                M=zeros(N,N)
                for j in 1:N
                    fill!(v.data,0);v.data[j]=1;M[:,j].=tm(v).data
                end
                push!(vals,maximum(abs.(eigvals(M))))
            end
            actual[m]=maximum(vals)
        end
        errors=[abs(actual[m]/actual[0]-refs[m]/refs[0]) for m in keys(actual)]
        @assert maximum(errors)<1e-11
        push!(rows,Dict("h"=>h,"tube_width"=>width,"spin_chain_length"=>2width,"time_extent"=>.125,"charge_ratio_errors"=>errors))
        println("geometry h=",h," width=",width," error=",maximum(errors));flush(stdout)
    end
end
atomic_toml(joinpath(HERE,"control/geometry_validation.toml"),Dict("checks"=>rows,"scope"=>"Independent finite-Trotter spin-chain charge-sector eigenvalue ratios, unstacked initial tensors"))
