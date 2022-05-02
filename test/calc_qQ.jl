using Test: @testset, @test, @inferred
using Statistics: mean
using Random: bitrand
using AdvRBMs: calc_q, calc_Q, calc_qs, calc_Qs

@testset "calc_q, calc_Q" begin
    u = bitrand(10)
    v = randn(5, 3, 10)

    q = mean(reshape(u,1,1,:) .* v; dims=3) - mean(u) * mean(v; dims=3)
    @test q ≈ @inferred calc_q(u, v)
    @test calc_q(u, v) ≈ @inferred calc_q(u, v; wts=ones(10))

    u_ = u .- mean(u)
    v_ = v .- mean(v; dims=ndims(v))
    Q = [mean(u_ .* v_[i,:] .* v_[j,:]) for i in CartesianIndices((5,3)), j in CartesianIndices((5,3))]
    Q = reshape(Q, size(Q)..., 1)
    @test Q ≈ @inferred calc_Q(u, v)
    @test calc_Q(u, v) ≈ @inferred calc_Q(u, v; wts=ones(10))

    @test calc_q(Float32, u, v)::AbstractArray{Float32} ≈ calc_q(u, v)
    @test calc_Q(Float32, u, v)::AbstractArray{Float32} ≈ calc_Q(u, v)
end

@testset "q, Q for categorical labels" begin
    u = bitrand(10)
    u = BitMatrix([u'; 1 .- u'])
    v = randn(5, 3, 10)
    q = @inferred calc_q(u, v)
    Q = @inferred calc_Q(u, v)
    @test q[:,:,1] ≈ calc_q(u[2,:], v)
    @test size(q) == (5,3,1)
    @test size(Q) == (5,3,5,3,2)
    @test selectdim(Q, ndims(Q), 1) ≈ -selectdim(Q, ndims(Q), 2)
    Q_flat = reshape(Q, 5 * 3, 5 * 3, 2)
    @test Q_flat[:,:,1] ≈ Q_flat[:,:,1]'
    @test Q_flat[:,:,2] ≈ Q_flat[:,:,2]'

    qs = @inferred calc_qs(u, v)
    Qs = @inferred calc_Qs(u, v)

    @test only(qs) ≈ q
    @test Qs[1] ≈ Q[:,:,:,:,1]
    @test Qs[2] ≈ Q[:,:,:,:,2]
end
