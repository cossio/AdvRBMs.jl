import Random
using Test: @testset, @test
using Random: bitrand
using LinearAlgebra: norm
using Statistics: mean
using RestrictedBoltzmannMachines: BinaryRBM, initialize!, standardize, sample_v_from_v
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
    rbm = standardize(BinaryRBM(randn(5, 2), randn(3), randn(5, 2, 3)))
    q = randn(5, 2, 1)
    data = bitrand(5, 2, 128)
    advpcd!(rbm, data; qs = [q], steps = 1, iters = 10, batchsize = 32)
    # the hard constraint q' * w ≈ 0 acts on the flattened standardized weights
    @test norm(flatq(q)' * flatw(rbm)) < 1.0e-10

    rbm = standardize(BinaryRBM(randn(5, 2), randn(3), randn(5, 2, 3)))
    q = randn(5, 2, 1)
    data = bitrand(5, 2, 128)
    ℋ = CartesianIndices((2:3,))
    advpcd!(rbm, data; qs = [q], ℋs = [ℋ], steps = 1, iters = 10, batchsize = 32)
    # only the hidden units in ℋ are constrained
    @test norm(flatq(q)' * flatw(rbm)[:, ℋ]) < 1.0e-10
    @test norm(flatq(q)' * flatw(rbm)) > 1.0e-3
end

@testset "advpcd, standardized, 2nd-order soft constraint" begin
    data = bitrand(5, 2, 128)
    u = bitrand(128)
    q = calc_q(u, data)
    Q = calc_Q(u, data)
    rbm = standardize(BinaryRBM(randn(5, 2), randn(3), randn(5, 2, 3)))
    advpcd!(rbm, data; qs = [q], Qs = [Q], λQ = 1.0, steps = 1, iters = 10, batchsize = 32)
    @test norm(flatq(q)' * flatw(rbm)) < 1.0e-10
    @test all(isfinite, rbm.w)
end
