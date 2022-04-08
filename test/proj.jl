import Zygote
using Test: @test, @testset, @inferred
using LinearAlgebra: norm, dot
using AdvRBMs: ∂qw, ∂wQw, orthogonal_projection, sylvester_projection

@testset "orthogonal_projection" begin
    w, q = randn(10), randn(10)
    wp = @inferred orthogonal_projection(w, q)
    dot(wp, q) ≤ 1e-10min(norm(wp), norm(q))
    @test norm(wp)^2 + dot(w, q)^2 / norm(q)^2 ≈ norm(w)^2 # pythagoras
end

@testset "∂wQw" begin
    w = randn(7, 3)
    Q = randn(7, 7)
    Q = Q + Q' # make symmetric
    ∂, = Zygote.gradient(w) do w
        norm(w' * Q * w)^2 / 2
    end
    @test ∂ ≈ @inferred ∂wQw(w, Q)
end

@testset "∂qw" begin
    w = randn(7, 3)
    q = randn(7)
    ∂, = Zygote.gradient(w) do w
        norm(q' * w)^2 / 2
    end
    @test ∂ ≈ @inferred ∂qw(w, q)
end

@testset "sylvester_projection" begin
    A = randn(5, 4)
    X = randn(5, 4)
    Y = @inferred sylvester_projection(A, X)
    @test A'*Y ≈ -Y'*A
end
