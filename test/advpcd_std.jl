import Random
using Test: @testset, @test
using Random: bitrand
using LinearAlgebra: norm
using Statistics: mean, cov
using RestrictedBoltzmannMachines: BinaryRBM, initialize!, standardize, sample_v_from_v,
    inputs_h_from_v
using AdvRBMs: advpcd!, calc_q, calc_Q

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
    # heterogeneous unit statistics, so that scale_v is far from uniform and
    # the standardized-weight metric differs from the data-space metric
    p = reshape(range(0.1, 0.9; length = 10), 5, 2)
    data = rand(5, 2, 128) .< p

    rbm = standardize(BinaryRBM(randn(5, 2), randn(3), randn(5, 2, 3)))
    q = randn(5, 2, 1)
    advpcd!(rbm, data; qs = [q], steps = 1, iters = 10, batchsize = 32)
    # the hard constraint acts on the unstandardized weights w ./ scale_v, so
    # that the inputs to the hidden units carry no 1st-order label information
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
