import Random
using Test: @testset, @test, @inferred
using Statistics: mean
using Random: bitrand
using LinearAlgebra: norm
using AdvRBMs: calc_q, calc_Q, calc_qs, calc_Qs, kernelproj, ∂wQw

# calc_q / calc_Q require labels with both classes present, so seed the RNG to
# keep the bitrand draws below away from the degenerate all-zero/all-one case.
Random.seed!(1)

@testset "calc_q, calc_Q" begin
    u = bitrand(10)
    v = randn(5, 3, 10)

    q = mean(reshape(u, 1, 1, :) .* v; dims = 3) - mean(u) * mean(v; dims = 3)
    @test q ≈ @inferred calc_q(u, v)
    @test calc_q(u, v) ≈ @inferred calc_q(u, v; wts = ones(10))

    u_ = u .- mean(u)
    v_ = v .- mean(v; dims = ndims(v))
    Q = [mean(u_ .* v_[i, :] .* v_[j, :]) for i in CartesianIndices((5, 3)), j in CartesianIndices((5, 3))]
    Q = reshape(Q, size(Q)..., 1)
    @test Q ≈ @inferred calc_Q(u, v)
    @test calc_Q(u, v) ≈ @inferred calc_Q(u, v; wts = ones(10))

    @test calc_q(Float32, u, v)::AbstractArray{Float32} ≈ calc_q(u, v)
    @test calc_Q(Float32, u, v)::AbstractArray{Float32} ≈ calc_Q(u, v)
end

@testset "q, Q for categorical labels" begin
    u = bitrand(10)
    u = BitMatrix([u'; 1 .- u'])
    v = randn(5, 3, 10)
    q = @inferred calc_q(u, v)
    Q = @inferred calc_Q(u, v)
    @test size(q) == (5, 3, 1)
    @test size(Q) == (5, 3, 5, 3, 1)
    @test q[:, :, 1] ≈ calc_q(u[2, :], v)
    @test Q[:, :, :, :, 1] ≈ calc_Q(u[2, :], v)
    Q_flat = reshape(Q, 5 * 3, 5 * 3, 1)
    @test Q_flat[:, :, 1] ≈ Q_flat[:, :, 1]'

    qs = @inferred calc_qs(u, v)
    Qs = @inferred calc_Qs(u, v)

    # entries keep the trailing singleton constraint dimension expected by advpcd!
    @test size(only(qs)) == size(q)
    @test size(only(Qs)) == size(Q)
    @test only(qs) ≈ q
    @test only(Qs) ≈ Q

    # uniform weights reproduce the unweighted statistics
    @test calc_q(u, v; wts = ones(size(u, 2))) ≈ q
end

@testset "calc_qs, calc_Qs entries are usable as advpcd! constraints" begin
    w = randn(5, 3, 7)

    # categorical labels, tensor visible layer; force both classes present
    u = bitrand(20)
    u[1] = true
    u[2] = false
    u = BitMatrix([u'; 1 .- u'])
    v = randn(5, 3, 20)
    for q in calc_qs(u, v)
        wp = kernelproj(w, q)
        @test norm(sum(q .* wp; dims = (1, 2))) < 1.0e-9 * norm(q) * norm(w)
        # the removed component lies in the span of the single flattened q
        removed = reshape(w - wp, 15, 7)
        qf = reshape(q, 15)
        @test norm(removed - qf * (qf' * removed) / (qf' * qf)) < 1.0e-9 * norm(w)
    end
    for Q in calc_Qs(u, v)
        @test size(∂wQw(w, Q)) == size(w)
    end

    # binary labels, vector visible layer; force both classes present
    u = bitrand(20)
    u[1] = true
    u[2] = false
    v = randn(15, 20)
    w = randn(15, 7)
    for q in calc_qs(u, v)
        wp = kernelproj(w, q)
        @test norm(reshape(q, 1, :) * wp) < 1.0e-9 * norm(q) * norm(w)
    end
    for Q in calc_Qs(u, v)
        @test size(∂wQw(w, Q)) == size(w)
    end
end
