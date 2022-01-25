using Test, LinearAlgebra
import RBMs12Con, Zygote

@testset "project_plane" begin
    w, q = randn(10), randn(10)
    wp = RBMs12Con.project_plane(w, q)
    dot(wp, q) ≤ 1e-10min(norm(wp), norm(q))
    @test norm(wp)^2 + dot(w, q)^2 / norm(q)^2 ≈ norm(w)^2 # pythagoras
end

@testset "∂wQw" begin
    w = randn(7, 3)
    Q = randn(7, 7)
    Q = Q + Q' # make symmetric
    gs = Zygote.gradient(w, Q) do w, Q
        norm(w' * Q * w)^2 / 2
    end
    @test first(gs) ≈ RBMs12Con.∂wQw(w, Q)
end

@testset "sylvester_projection" begin
    A = randn(5, 4)
    X = randn(5, 4)
    Y = RBMs12Con.sylvester_projection(A, X)
    @test A'*Y ≈ -Y'*A
end
