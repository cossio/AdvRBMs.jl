using Test, LinearAlgebra, Statistics
import AdvRBMs
import Zygote

@testset "estimate_q, estimate_Q, estimate_qQ" begin
    u = rand(0:1, 10)
    v = randn(5, 3, 10)

    q = dropdims(mean(reshape(u,1,1,:) .* v; dims=3) - mean(u) * mean(v; dims=3); dims=3)
    @test q ≈ AdvRBMs.estimate_q(u, v)

    u_ = mean(u)
    Q = [mean((u .- u_) .* v[i,:] .* v[j,:]) for i in CartesianIndices((5,3)), j in CartesianIndices((5,3))]
    @test Q ≈ AdvRBMs.estimate_Q(u, v)

    q, Q = AdvRBMs.estimate_qQ(u, v)
    @test q ≈ AdvRBMs.estimate_q(u, v)
    @test Q ≈ AdvRBMs.estimate_Q(u, v)
end
