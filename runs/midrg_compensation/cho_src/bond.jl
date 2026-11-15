# AI-generated experimental adapter. Leg convention follows flux_action.
using TNRKit, TensorKit, LinearAlgebra, Random
include("twist_measure.jl")

function bond_factor(V, mu, mode)
    D = id(Float64, V)
    for q in sectors(V)
        d = mode == :parity ? (isodd(Int(q.j)) ? -1.0 : 1.0) : Float64(q.j)^2
        block(D, q) .*= exp(mu * d)
    end
    return D
end

function multiply_second(A, D)
    @tensor out[-1 -2;-3 -4] := D[-2;a]*A[-1 a;-3 -4]
    return out
end

function intervene(s, mu; mode=:parity, axis=:horizontal)
    A, B = s.TA, s.TB
    if axis in (:horizontal, :both)
        A = multiply_first(A, bond_factor(codomain(A)[1], -mu, mode))
        B = multiply_first(B, bond_factor(codomain(B)[1], mu, mode))
    end
    if axis in (:vertical, :both)
        A = multiply_second(A, bond_factor(codomain(A)[2], mu, mode))
        B = multiply_second(B, bond_factor(codomain(B)[2], -mu, mode))
    end
    return LoopTNR(A, B)
end

function gauge(V, rng, strength)
    G = id(Float64, V)
    for (_, b) in blocks(G)
        b .= exp(strength * randn(rng, size(b)))
    end
    return G
end

function horizontal_gauge(s, rng; strength=0.15)
    A, B = s.TA, s.TB
    GA = gauge(codomain(A)[1], rng, strength)
    GB = gauge(codomain(B)[1], rng, strength)
    IA, IB = inv(GA), inv(GB)
    @tensor Ag[-1 -2;-3 -4] := GA[-1;a]*A[a -2;-3 b]*IB[b;-4]
    @tensor Bg[-1 -2;-3 -4] := GB[-1;a]*B[a -2;-3 b]*IA[b;-4]
    return LoopTNR(Ag, Bg)
end

function tube_sector(s, q; flux=false, nlevels=2)
    A, B = s.TA, s.TB
    Az = flux ? multiply_first(A, pi_rotation(codomain(A)[1])) : A
    Bz = flux ? multiply_first(B, pi_rotation(codomain(B)[1])) : B
    xs = domain(A)[1] ⊗ domain(B)[1]
    p = zeros(ComplexF64, xs ← Vect[CU1Irrep](q => 1))
    N = length(p.data)
    N > 0 || error("Empty tube sector $q")
    M = zeros(ComplexF64, N, N)
    for j in 1:N
        fill!(p.data, 0)
        p.data[j] = 1
        M[:, j] .= flux_action(A, B, Az, Bz, p).data
    end
    E = eigen(M)
    order = sortperm(abs.(E.values); rev=true)
    matrixnorm = norm(M)
    return [(lambda=E.values[j],
             residual=norm(M * E.vectors[:,j] - E.values[j] * E.vectors[:,j]) / matrixnorm)
            for j in first(order, min(N, nlevels))]
end

function signed_tube(s)
    even = tube_sector(s, CU1Irrep(0,0); flux=true)
    odd = tube_sector(s, CU1Irrep(0,1); flux=true)
    delta = -log(abs(odd[1].lambda / even[1].lambda)) / (2pi)
    gap(rows) = length(rows) >= 2 ? -log(abs(rows[2].lambda / rows[1].lambda)) / (2pi) : Inf
    return (; delta, even_gap=gap(even), odd_gap=gap(odd),
            residual=maximum(r.residual for r in vcat(even, odd)))
end
