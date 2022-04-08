using Test: @testset, @test, @inferred
using Statistics: mean
using Random: bitrand
using AdvRBMs: calc_q, calc_Q, calc_qQ

@testset "calc_q, calc_Q, calc_qQ" begin
    u = bitrand(10)
    v = randn(5, 3, 10)

    q = dropdims(mean(reshape(u,1,1,:) .* v; dims=3) - mean(u) * mean(v; dims=3); dims=3)
    @test q ≈ @inferred calc_q(u, v)

    u_ = mean(u)
    Q = [mean((u .- u_) .* v[i,:] .* v[j,:]) for i in CartesianIndices((5,3)), j in CartesianIndices((5,3))]
    @test Q ≈ @inferred calc_Q(u, v)

    q, Q = @inferred calc_qQ(u, v)
    @test q ≈ @inferred calc_q(u, v)
    @test Q ≈ @inferred calc_Q(u, v)
end
