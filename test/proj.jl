import Zygote
using Test: @test, @testset, @inferred
using LinearAlgebra: norm, dot
using AdvRBMs: kernelproj, ∂qw, ∂wQw, sylvester_projection

@testset "kernelproj" begin
    w = randn(28,7)
    q = randn(28,2)
    wp = @inferred kernelproj(w, q)
    @test norm(q' * wp) < 1e-9 * min(norm(w), norm(q))
    @test norm(wp)^2 + norm(q * ((q' * q) \ (q' * w)))^2 ≈ norm(w)^2 # pythagoras

    w = randn(28,28,7)
    q = randn(28,28,2)
    wp = @inferred kernelproj(w, q)
    @test norm(sum(reshape(q, 28, 28, 1, 2) .* wp, dims=(1,2))) < 1e-9 * norm(sum(reshape(q, 28, 28, 1, 2) .* w, dims=(1,2)))

    q = zeros(28,28,2)
    @test @inferred(kernelproj(w, q)) ≈ w
end

@testset "∂qw" begin
    w = randn(7, 5)
    q = randn(7, 2)
    ∂, = Zygote.gradient(w) do w
        norm(q' * w)^2 / 2
    end
    @test ∂ ≈ @inferred ∂qw(w, q)
end

@testset "∂wQw" begin
    w = randn(7, 5)
    Q = randn(7, 7, 2)
    # make symmetric
    for k in 1:2
        Q[:,:,k] = Q[:,:,k] + Q[:,:,k]'
    end
    ∂, = Zygote.gradient(w) do w
        sum(norm(w' * Q[:,:,k] * w)^2 / 2 for k in 1:2)
    end
    @test ∂ ≈ @inferred ∂wQw(w, Q)
end

@testset "sylvester_projection" begin
    A = randn(5, 4)
    X = randn(5, 4)
    Y = @inferred sylvester_projection(A, X)
    @test A'*Y ≈ -Y'*A
end
