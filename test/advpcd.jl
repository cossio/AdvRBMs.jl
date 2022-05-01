import MKL
using Test: @testset, @test, @inferred
using Random: bitrand
using LinearAlgebra: norm
using RestrictedBoltzmannMachines: RBM, Binary, BinaryRBM, inputs_v_to_h
using AdvRBMs: advpcd!

@testset "advpcd" begin
    rbm = BinaryRBM(randn(5,2), randn(3), randn(5,2,3))
    q = randn(5,2,1)
    data = bitrand(5,2,128)
    advpcd!(rbm, data; q, steps=1, epochs=1, batchsize=32)
    @test norm(inputs_v_to_h(rbm, q)) < 1e-10

    rbm = BinaryRBM(randn(5,2), randn(3), randn(5,2,3))
    q = randn(5,2,1)
    data = bitrand(5,2,128)
    ℋ = CartesianIndices((2:3,))
    advpcd!(rbm, data; q, ℋ, steps=1, epochs=1, batchsize=32)
    @test norm(inputs_v_to_h(rbm, q)[ℋ]) < 1e-10
    @test norm(inputs_v_to_h(rbm, q)[1,:]) > 1e-5
end
