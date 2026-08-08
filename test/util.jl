using Test: @test, @testset
using AdvRBMs: empty_intersections

@testset "empty_intersections" begin
    @test empty_intersections([1:2, 3:4, 5:6])
    @test !empty_intersections([1:3, 3:4])
    @test empty_intersections([CartesianIndices((1:2,)), CartesianIndices((3:4,))])
    @test !empty_intersections([CartesianIndices((1:2,)), CartesianIndices((2:3,))])
end
