# Phase 0C / 1 / 2: Projected spectral-gradient directions (3c, 4c, 5c, 6c) and 4-vacuum metric diagnostics
import TNRKit.Zygote

pairdot(x, y) = real(dot(x[1], y[1]) + dot(x[2], y[2]))
pairnorm(x) = sqrt(pairdot(x, x))
pairscale(a, x) = (a * x[1], a * x[2])
pairsub(x, y) = (x[1] - y[1], x[2] - y[2])
pairadd(x, y) = (x[1] + y[1], x[2] + y[2])
paircos(x, y) = pairdot(x, y) / max(pairnorm(x) * pairnorm(y), eps(Float64))

function rotate90_tensors(A, B)
    p = ((3, 1), (4, 2))
    return permute(B, p), permute(A, p)
end

function rotate90(s::LoopTNR)
    Ar, Br = rotate90_tensors(s.TA, s.TB)
    return LoopTNR(Ar, Br)
end

# 2x2 tube [2,2,0] eigenvalue & gradient wrt (A, B), supporting 0-deg and 90-deg orientations
function log_eigen_gradient_2x2(A, B, q; flux=false, rotated=false)
    Ar, Br = rotated ? rotate90_tensors(A, B) : (A, B)
    xs = domain(Ar)[1] ⊗ domain(Br)[1]
    p = zeros(ComplexF64, xs ← Vect[CU1Irrep](q => 1))
    N = length(p.data)
    @assert 0 < N <= 2048
    za = flux ? pi_rotation(codomain(Ar)[1]) : id(Float64, codomain(Ar)[1])
    zb = flux ? pi_rotation(codomain(Br)[1]) : id(Float64, codomain(Br)[1])
    f_inner = (a, b, x) -> flux_action(a, b, multiply_first(a, za), multiply_first(b, zb), x)
    M = zeros(ComplexF64, N, N)
    for j in 1:N
        fill!(p.data, 0); p.data[j] = 1
        M[:, j] = f_inner(Ar, Br, p).data
    end
    E = eigen(M)
    i = argmax(abs.(E.values))
    lam = E.values[i]
    rv = E.vectors[:, i]
    EL = eigen(adjoint(M))
    j = argmin(abs.(EL.values .- conj(lam)))
    lv = EL.vectors[:, j]
    rr = norm(M * rv - lam * rv) / (abs(lam) * norm(rv))
    lr = norm(adjoint(M) * lv - conj(lam) * lv) / (abs(lam) * norm(lv))
    overlap = dot(lv, rv)
    @assert abs(overlap) > 1e-7
    @assert max(rr, lr) < 1e-8 && abs(angle(lam)) < 1e-5
    r = copy(p); l = copy(p)
    r.data .= rv; l.data .= lv
    if !rotated
        grad = Zygote.gradient((a, b) -> real(dot(l, f_inner(a, b, r)) / (lam * overlap)), A, B)
    else
        grad = Zygote.gradient(A, B) do a, b
            ar, br = rotate90_tensors(a, b)
            real(dot(l, f_inner(ar, br, r)) / (lam * overlap))
        end
    end
    mags = sort(abs.(E.values); rev=true)
    gap = length(mags) > 1 ? (mags[1] - mags[2]) / mags[1] : 1.0
    @assert gap > 1e-7
    return grad, Dict("lambda_real" => real(lam), "lambda_imag" => imag(lam),
                      "abs_lambda" => abs(lam), "right_residual" => rr,
                      "left_residual" => lr, "overlap" => abs(overlap), "relative_gap" => gap)
end

# Exact @tensor adjoint of TNRKit.MPO_action_2gates(TA, TB, x) avoiding inner Zygote pullback overhead
function MPO_action_2gates_adj(TA::TensorMap, TB::TensorMap, w::TensorMap)
    ffx_adj = permute(w, ((4, 1, 2, 3), (5,)))
    @tensor fx_adj[-1 -2 -3 -4; 5] := conj(TA[1 2; -4 -1]) * ffx_adj[1 2 3 4; 5] * conj(TA[3 4; -2 -3])
    @tensor x_adj[-1 -2 -3 -4; 5] := conj(TB[1 2; -1 -2]) * fx_adj[1 2 3 4; 5] * conj(TB[3 4; -3 -4])
    return x_adj
end

# Diagonal tube [sqrt(2), 2*sqrt(2), 0] eigenvalue & gradient wrt (A, B) via MPO_action_2gates
function log_eigen_gradient_diag(A, B, q; rank=1, rotated=false)
    Ar, Br = rotated ? rotate90_tensors(A, B) : (A, B)
    xs = domain(Br) ⊗ domain(Br)
    x0 = zeros(Float64, xs ← Vect[CU1Irrep](q => 1))
    N = length(x0.data)
    @assert N > 0
    randn!(MersenneTwister(2140 + Int(round(2 * q.j)) * 10 + q.s), x0.data)
    x0 /= norm(x0)
    requested = min(max(rank + 3, 6), N)
    vals_r, vecs_r, _ = TNRKit.eigsolve(v -> TNRKit.MPO_action_2gates(Ar, Br, v), x0, requested, :LM;
        krylovdim=max(40, 2 * requested + 10), maxiter=100, tol=1e-12, verbosity=0)
    order_r = sortperm(abs.(vals_r); rev=true)
    ir = order_r[rank]
    lam = vals_r[ir]
    rv = vecs_r[ir]

    f_adj = w -> MPO_action_2gates_adj(Ar, Br, w)
    vals_l, vecs_l, _ = TNRKit.eigsolve(f_adj, x0, requested, :LM;
        krylovdim=max(40, 2 * requested + 10), maxiter=100, tol=1e-12, verbosity=0)
    il = argmin(abs.(vals_l .- conj(lam)))
    lv = vecs_l[il]

    rr = norm(TNRKit.MPO_action_2gates(Ar, Br, rv) - lam * rv) / (abs(lam) * norm(rv))
    lr = norm(f_adj(lv) - conj(lam) * lv) / (abs(lam) * norm(lv))
    @assert max(rr, lr) < 1e-8 && abs(imag(lam)) < 1e-8 * abs(lam)
    r_real = copy(x0); l_real = copy(x0)
    r_real.data .= real.(rv.data)
    l_real.data .= real.(lv.data)
    lam_real = real(lam)
    overlap_real = dot(l_real, r_real)
    @assert abs(overlap_real) > 1e-6 "Small overlap $(abs(overlap_real)) in sector $q rank $rank"
    if !rotated
        grad = Zygote.gradient((a, b) -> real(dot(l_real, TNRKit.MPO_action_2gates(a, b, r_real)) / (lam_real * overlap_real)), A, B)
    else
        grad = Zygote.gradient(A, B) do a, b
            ar, br = rotate90_tensors(a, b)
            real(dot(l_real, TNRKit.MPO_action_2gates(ar, br, r_real)) / (lam_real * overlap_real))
        end
    end
    mags = abs.(vals_r[order_r])
    gap = length(mags) > rank ? (mags[rank] - mags[rank+1]) / mags[rank] : 1.0
    return grad, Dict("lambda_real" => real(lam), "lambda_imag" => imag(lam),
                      "abs_lambda" => abs(lam), "right_residual" => rr,
                      "left_residual" => lr, "overlap" => abs(overlap_real), "relative_gap" => gap)
end

# 4-vacuum metric solver: given raw (unnormalized) positive vacuum eigenvalues la, lb (2x2) and ld, ldp (diag)
# Wait: in measure_loop(s), A and B are normalized by la^(1/4), so lambda0_0 = ld / la, and for rotated s_rot,
# A_rot and B_rot are normalized by lb^(1/4), so lambda0_90 = ldp / lb!
# And log(lb / la) comes directly from the unnormalized 2x2 vacuum eigenvalues la, lb.
function fit_4vacuum_metric(log_lb_over_la::Real, log_ld_over_la::Real, log_ldp_over_la::Real)
    F_map(z) = (1 + z) / (2 * (1 - z))
    function residuals(params)
        c, x, y = params
        z = ComplexF64(x, y)
        zb = -1.0 / z
        zd = F_map(z)
        zdp = F_map(zb)
        k = pi * c / 6.0
        return [
            k * (imag(zb) - imag(z)) - log_lb_over_la,
            k * (imag(zd) - imag(z)) - log_ld_over_la,
            k * (imag(zdp) - imag(z)) - log_ldp_over_la
        ]
    end
    p = [1.0, 0.0, 1.0]
    for iter in 1:30
        r = residuals(p)
        norm(r) < 1e-14 && break
        J = zeros(Float64, 3, 3)
        eps_fd = 1e-8
        for col in 1:3
            p_p = copy(p); p_p[col] += eps_fd
            p_m = copy(p); p_m[col] -= eps_fd
            J[:, col] = (residuals(p_p) - residuals(p_m)) / (2 * eps_fd)
        end
        p -= J \ r
    end
    r_final = residuals(p)
    z = ComplexF64(p[2], p[3])
    zd = F_map(z)
    return Dict("c_metric" => p[1], "tau0_real" => p[2], "tau0_imag" => p[3],
                "tau_d_real" => real(zd), "tau_d_imag" => imag(zd),
                "fit_residual" => norm(r_final))
end

function project_direction(target, constraints, ref_pair)
    basis = typeof(target)[]
    for v0 in constraints
        v = v0
        for b in basis
            v = pairsub(v, pairscale(pairdot(b, v), b))
        end
        pairnorm(v) > 1e-12 && push!(basis, pairscale(1.0 / pairnorm(v), v))
    end
    D = target
    for b in basis
        D = pairsub(D, pairscale(pairdot(b, D), b))
    end
    @assert pairnorm(D) > 1e-12
    D = pairscale(pairnorm(ref_pair) / pairnorm(D), D)
    return D, length(basis)
end

function compute_all_directions(s::LoopTNR)
    A, B = s.TA, s.TB
    ge, re = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=true, rotated=false)
    go, ro = log_eigen_gradient_2x2(A, B, CU1Irrep(0,1); flux=true, rotated=false)
    gv, rv = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=false, rotated=false)
    gb, rb = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=false, rotated=true)

    gd_vac, rd_vac = log_eigen_gradient_diag(A, B, CU1Irrep(0,0); rank=1, rotated=false)
    gdp_vac, rdp_vac = log_eigen_gradient_diag(A, B, CU1Irrep(0,0); rank=1, rotated=true)
    g01_raw, r01_raw = log_eigen_gradient_diag(A, B, CU1Irrep(1,2); rank=1, rotated=false)

    # Functional gradients:
    # 1. Twist splitting: delta_pi = (log|lam_even| - log|lam_odd|) / (2*pi)
    g_delta = pairscale(1.0 / (2pi), pairsub(ge, go))
    # 2. Twist pair center: xbar_pi = (2*log|lam_a| - log|lam_even| - log|lam_odd|) / (4*pi)
    g_c = pairscale(1.0 / (4pi), pairsub(pairscale(2.0, gv), pairadd(ge, go)))
    # 3. 2x2 vacuum anisotropy: Phi_aniso = log|lam_b / lam_a|
    g_aniso = pairsub(gb, gv)
    # 4. Diagonal vacuum anisotropy: Phi_aniso_diag = log|lam_d' / lam_d|
    g_aniso_diag = pairsub(gdp_vac, gd_vac)
    # 5. Raw diagonal lowest charged scaling dimension: x01_raw = -log|lam_01 / lam_d| / pi
    g_x01 = pairscale(1.0 / pi, pairsub(gd_vac, g01_raw))
    # 6. Raw central charge: c_raw = -12/pi * log(lam_d / lam_a)
    g_craw = pairscale(-12.0 / pi, pairsub(gd_vac, gv))

    cA = (A, zero(B))
    cB = (zero(A), B)
    ref_pair = (A, B)

    D_3c, r_3c = project_direction(g_delta, [cA, cB, g_c], ref_pair)
    D_4c, r_4c = project_direction(g_delta, [cA, cB, g_c, g_aniso], ref_pair)
    D_5c, r_5c = project_direction(g_delta, [cA, cB, g_c, g_aniso, g_x01], ref_pair)
    D_6c, r_6c = project_direction(g_delta, [cA, cB, g_c, g_aniso, g_aniso_diag, g_x01], ref_pair)

    dirs = Dict(:P3_3c => (D_3c, r_3c), :P3_4c => (D_4c, r_4c),
                :P3_5c => (D_5c, r_5c), :P3_6c => (D_6c, r_6c))
    info = Dict{String,Any}()
    for (name, (D, rnk)) in dirs
        info[string(name)] = Dict(
            "constraint_rank" => rnk,
            "kappa_delta" => pairdot(g_delta, D),
            "deriv_center" => pairdot(g_c, D),
            "deriv_aniso_2x2" => pairdot(g_aniso, D),
            "deriv_aniso_diag" => pairdot(g_aniso_diag, D),
            "deriv_x01" => pairdot(g_x01, D),
            "deriv_craw" => pairdot(g_craw, D),
            "deriv_radial_A" => real(dot(A, D[1])),
            "deriv_radial_B" => real(dot(B, D[2])),
            "cos_with_3c" => paircos(D, D_3c)
        )
    end
    overlaps = Dict(
        "cos_gdelta_ganiso" => paircos(g_delta, g_aniso),
        "cos_gdelta_gx01" => paircos(g_delta, g_x01),
        "cos_gc_ganiso" => paircos(g_c, g_aniso),
        "cos_gc_gx01" => paircos(g_c, g_x01),
        "cos_ganiso_gx01" => paircos(g_aniso, g_x01),
        "cos_ganiso_ganisodiag" => paircos(g_aniso, g_aniso_diag)
    )
    raw_grads = Dict(:g_delta => g_delta, :g_c => g_c, :g_aniso => g_aniso,
                     :g_aniso_diag => g_aniso_diag, :g_x01 => g_x01, :g_craw => g_craw)
    return Dict(:P3_3c => D_3c, :P3_4c => D_4c, :P3_5c => D_5c, :P3_6c => D_6c), info, overlaps, raw_grads
end

function compute_direction_fast(s::LoopTNR, dir_key::Symbol)
    A, B = s.TA, s.TB
    ge, _ = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=true, rotated=false)
    go, _ = log_eigen_gradient_2x2(A, B, CU1Irrep(0,1); flux=true, rotated=false)
    gv, _ = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=false, rotated=false)
    g_delta = pairscale(1.0 / (2pi), pairsub(ge, go))
    g_c = pairscale(1.0 / (4pi), pairsub(pairscale(2.0, gv), pairadd(ge, go)))
    cA = (A, zero(B))
    cB = (zero(A), B)
    ref_pair = (A, B)
    if dir_key == :P3_3c
        D, _ = project_direction(g_delta, [cA, cB, g_c], ref_pair)
        return D, pairdot(g_delta, D)
    end
    gb, _ = log_eigen_gradient_2x2(A, B, CU1Irrep(0,0); flux=false, rotated=true)
    g_aniso = pairsub(gb, gv)
    if dir_key == :P3_4c
        D, _ = project_direction(g_delta, [cA, cB, g_c, g_aniso], ref_pair)
        return D, pairdot(g_delta, D)
    end
    gd_vac, _ = log_eigen_gradient_diag(A, B, CU1Irrep(0,0); rank=1, rotated=false)
    g01_raw, _ = log_eigen_gradient_diag(A, B, CU1Irrep(1,2); rank=1, rotated=false)
    g_x01 = pairscale(1.0 / pi, pairsub(gd_vac, g01_raw))
    D, _ = project_direction(g_delta, [cA, cB, g_c, g_aniso, g_x01], ref_pair)
    GC.gc(false)
    return D, pairdot(g_delta, D)
end

function apply_direction(s::LoopTNR, D, alpha::Real; do_finalize::Bool=false)
    out = deepcopy(s)
    out.TA = s.TA + alpha * D[1]
    out.TB = s.TB + alpha * D[2]
    do_finalize && finalize!(out)
    return out
end

