using Test: @test, @testset, @inferred
using Zygote: gradient
using LinearAlgebra: norm, dot, pinv
using AdvRBMs: kernelproj, ∂qw, ∂wQw, sylvester_projection, pseudo_inv_of_q

@testset "pseudo_inv_of_q" begin
    q = randn(28, 64, 5)
    @test pseudo_inv_of_q(q) ≈ reshape(pinv(reshape(q, 28 * 64, 5)), 5, 64, 28)
end

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

    w = randn(100, 20)
    q = randn(100, 1)
    wp = @inferred kernelproj(w, q)
    @test wp ≈ w - q * q'w ./ (q'q)
end

@testset "∂qw" begin
    w = randn(7, 5)
    q = randn(7, 2)
    ∂, = gradient(w) do w
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
    ∂, = gradient(w) do w
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
