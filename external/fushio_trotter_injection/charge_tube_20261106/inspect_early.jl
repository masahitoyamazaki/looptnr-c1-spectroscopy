include("measure.jl")
for line in eachline(joinpath(HERE,"control/early_jobs0.tsv"))
 source,n,path,hash=split(line,'\t');s=deserialize(joinpath(REPO,path));A=s.scheme.TA;B=s.scheme.TB
 geo=blocked_metric(-.5,.125,4.,parse(Int,n));if geo.rotated;p=((3,1),(4,2));A,B=permute(B,p),permute(A,p);end
 tm=TNRKit.CFTTransferMatrix(A,B,[1.,4.,1.])
 println("RG",n," TA=",space(A)," TMdomain=",domain(tm));flush(stdout)
end
