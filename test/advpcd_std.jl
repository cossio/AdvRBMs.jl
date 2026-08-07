import Random
using Test: @testset, @test
using Random: bitrand
using LinearAlgebra: norm
using Statistics: mean, cov
using Zygote: gradient
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary, Potts, initialize!, standardize,
    sample_v_from_v, inputs_h_from_v
using AdvRBMs: advpcd!, calc_q, calc_Q, ∂wQw

Random.seed!(2)

@testset "advpcd, standardized, unconstrained" begin
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = BinaryRBM(2, 5)
    initialize!(rbm, data)
    std_rbm = standardize(rbm)
    advpcd!(std_rbm, data; iters = 10000, batchsize = 64, steps = 10)

    v_sample = sample_v_from_v(std_rbm, bitrand(2, 10000); steps = 50)

    @test 0.4 < mean(v_sample[1, :]) < 0.6
    @test 0.4 < mean(v_sample[2, :]) < 0.6
    @test 0.4 < mean(v_sample[1, :] .* v_sample[2, :]) < 0.6
end

flatw(rbm) = reshape(rbm.w, :, prod(size(rbm.hidden)))
flatq(q) = reshape(q, :, size(q)[end])

@testset "advpcd, standardized, 1st-order constraint" begin
    # heterogeneous unit statistics, so that scale_v is far from uniform
    p = reshape(range(0.1, 0.9; length = 10), 5, 2)
    data = rand(5, 2, 128) .< p

    rbm = standardize(BinaryRBM(randn(5, 2), randn(3), randn(5, 2, 3)))
    q = randn(5, 2, 1)
    advpcd!(rbm, data; qs = [q], steps = 1, iters = 10, batchsize = 32)
    # the hard constraint acts on the unstandardized weights w ./ scale_v
    @test norm(flatq(q ./ rbm.scale_v)' * flatw(rbm)) < 1.0e-10
    # projecting in the standardized-weight metric would not satisfy this
    @test norm(flatq(q)' * flatw(rbm)) > 1.0e-3

    rbm = standardize(BinaryRBM(randn(5, 2), randn(3), randn(5, 2, 3)))
    q = randn(5, 2, 1)
    ℋ = CartesianIndices((2:3,))
    advpcd!(rbm, data; qs = [q], ℋs = [ℋ], steps = 1, iters = 10, batchsize = 32)
    # only the hidden units in ℋ are constrained
    @test norm(flatq(q ./ rbm.scale_v)' * flatw(rbm)[:, ℋ]) < 1.0e-10
    @test norm(flatq(q ./ rbm.scale_v)' * flatw(rbm)) > 1.0e-3
end

@testset "advpcd, standardized, constrained hidden inputs carry no label info" begin
    N, H, T = 10, 4, 2000
    p = range(0.05, 0.5; length = N)
    labels = bitrand(T)
    data = falses(N, T)
    for i in 1:N, t in 1:T
        # the label shifts each unit's firing probability
        data[i, t] = rand() < clamp(p[i] + 0.1 * (isodd(i) ? 1 : -1) * labels[t], 0, 1)
    end
    q = calc_q(labels, data)

    rbm0 = BinaryRBM(N, H)
    initialize!(rbm0, data)
    rbm = standardize(rbm0)
    advpcd!(rbm, data; qs = [q], steps = 1, iters = 20, batchsize = 64)

    # covariance of hidden inputs with the label vanishes on the training data
    I_h = inputs_h_from_v(rbm, data)
    @test maximum(abs, [cov(I_h[μ, :], labels) for μ in 1:H]) < 1.0e-10

    # the 2nd-order penalty path accepts a data-space Q and keeps the hard constraint
    Q = calc_Q(labels, data)
    advpcd!(rbm, data; qs = [q], Qs = [Q], λQ = 0.01, steps = 1, iters = 5, batchsize = 64)
    I_h = inputs_h_from_v(rbm, data)
    @test maximum(abs, [cov(I_h[μ, :], labels) for μ in 1:H]) < 1.0e-10
end

@testset "λQ penalty gradient incorporates hidden scales" begin
    # check ∂f/∂w = (∂f/∂w̃) ./ scale_h for f = Σₖ ||w̃' Qₖ w̃||²/2, w̃ = w ./ scale_h
    w = randn(6, 3)
    sh = reshape([0.5, 1.0, 2.0], 1, 3)
    Q = randn(6, 6, 2)
    for k in axes(Q, 3)
        Q[:, :, k] = Q[:, :, k] + Q[:, :, k]'
    end
    ∂, = gradient(w) do w
        w̃ = w ./ sh
        sum(norm(w̃' * Q[:, :, k] * w̃)^2 / 2 for k in axes(Q, 3))
    end
    @test ∂ ≈ ∂wQw(w ./ sh, Q) ./ sh
end

@testset "advpcd, standardized, Potts visible: gauge and constraint hold together" begin
    A, N, H, T = 3, 4, 2, 500
    labels = bitrand(T)
    labels[1] = true
    labels[2] = false
    # categorical data with label-dependent, site-dependent color frequencies
    data = falses(A, N, T)
    for i in 1:N, t in 1:T
        r = rand()
        p1 = 0.2 + 0.3 * labels[t] * isodd(i)
        a = r < p1 ? 1 : (r < p1 + 0.3 ? 2 : 3)
        data[a, i, t] = true
    end
    q = calc_q(labels, data)
    # calc_q on one-hot data is zerosum over colors
    @test norm(sum(q; dims = 1)) < 1.0e-12

    rbm0 = RBM(Potts(Float64, (A, N)), Binary(Float64, (H,)), zeros(A, N, H))
    initialize!(rbm0, data)
    rbm = standardize(rbm0)
    advpcd!(rbm, data; qs = [q], steps = 1, iters = 10, batchsize = 32)

    # the hard constraint in the data-space metric ...
    @test norm(flatq(q ./ rbm.scale_v)' * flatw(rbm)) < 1.0e-8
    # ... and the zerosum gauge of the unstandardized weights hold simultaneously
    @test norm(mean(rbm.w ./ rbm.scale_v; dims = 1)) < 1.0e-8
end

@testset "advpcd, standardized, hidden Potts: partially constrained sites" begin
    N, A, S, T = 5, 3, 2, 300 # visible units, hidden colors, hidden sites
    data = bitrand(N, T)
    rbm0 = RBM(Binary(Float64, (N,)), Potts(Float64, (A, S)), 0.01 .* randn(N, A, S))
    initialize!(rbm0, data)
    rbm = standardize(rbm0)
    q = randn(N, 1)
    # constrain only the first hidden color of each site
    ℋ = CartesianIndices((1:1, 1:S))
    advpcd!(rbm, data; qs = [q], ℋs = [ℋ], steps = 1, iters = 10, batchsize = 32)
    q_std = vec(q) ./ rbm.scale_v
    @test norm(q_std' * rbm.w[:, 1, :]) < 1.0e-10
    # the unconstrained colors are not projected
    @test norm(q_std' * rbm.w[:, 2, :]) > 1.0e-6
end
