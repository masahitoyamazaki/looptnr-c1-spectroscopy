include("measure.jl")
function grouped_parts(tm)
    V=codomain(tm.TA)[1];parts=[]
    for q in sectors(V)
        d=dim(V,q);p=zeros(Float64,V ← Vect[CU1Irrep](q=>d));block(p,q).=Matrix{Float64}(I,d,d)
        @tensor a[-1 -2;-3 -4] := conj(p[1;-1])*tm.TA[1 -2;-3 -4]
        @tensor b[-1 -2;-3 -4] := tm.TB[-1 -2;-3 1]*p[1;-4]
        push!(parts,(a,b))
    end
    parts
end
source,n,path,hash=split(first(readlines(joinpath(HERE,"control/jobs.tsv"))),'\t')
s=deserialize(joinpath(REPO,path)).scheme;c=TOML.parsefile(source)["config"]
Q,_,geo=shifted_Q(s,parse(Int,n),c["delta"],c["dt"],c["beta_cell"])
tm=TNRKit.CFTTransferMatrix(real(Q.TA),real(Q.TB),[1.,4.,1.]);p=sliced_parts(tm);g=grouped_parts(tm)
x=randn(Float64,domain(tm)←Vect[CU1Irrep](CU1Irrep(1,2)=>1))
results=Dict{String,Any}()
for (label,parts) in [("single",p),("grouped",g)]
    t=@elapsed y=sliced_action(tm,parts,x)
    t2=@elapsed y2=sliced_action(tm,parts,x)
    results[label]=Dict("seconds"=>t2,"compile_seconds"=>t,"parts"=>length(parts))
    if label=="single";global ref=y2;else;results["relative_error"]=norm(y2-ref)/norm(ref);end
    println(label," ",results[label]);flush(stdout);GC.gc()
end
atomic_toml(joinpath(HERE,"control/benchmark.toml"),results)
