import Random
using Optimisers: Adam
using Test: @testset, @test, @inferred
using Random: bitrand
using LinearAlgebra: norm, Diagonal, I, dot
using Statistics: mean, cor
using RestrictedBoltzmannMachines: RBM, BinaryRBM, inputs_h_from_v, initialize!, sample_v_from_v
using AdvRBMs: advpcd!

Random.seed!(2)

@testset "advpcd, unconstrained" begin
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = BinaryRBM(2, 5)
    initialize!(rbm, data)
    advpcd!(rbm, data; iters = 10000, batchsize = 64, steps = 10)

    v_sample = sample_v_from_v(rbm, bitrand(2, 10000); steps=50)

    @test 0.4 < mean(v_sample[1,:]) < 0.6
    @test 0.4 < mean(v_sample[2,:]) < 0.6
    @test 0.4 < mean(v_sample[1,:] .* v_sample[2,:]) < 0.6
end

@testset "advpcd, 1st-order constraint" begin
    rbm = BinaryRBM(randn(5,2), randn(3), randn(5,2,3))
    q = randn(5,2,1)
    data = bitrand(5,2,128)
    advpcd!(rbm, data; qs=[q], steps=1, iters=10, batchsize=32)
    @test norm(inputs_h_from_v(rbm, q)) < 1e-10

    rbm = BinaryRBM(randn(5,2), randn(3), randn(5,2,3))
    q = randn(5,2,1)
    data = bitrand(5,2,128)
    ℋ = CartesianIndices((2:3,))
    advpcd!(rbm, data; qs = [q], ℋs = [ℋ], steps=1, iters=10, batchsize=32)
    @info @test norm(inputs_h_from_v(rbm, q)[ℋ]) < 1e-10
    @info @test norm(inputs_h_from_v(rbm, q)[1,:]) > 1e-5
end
