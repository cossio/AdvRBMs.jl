import AdvRBMs
import TestCompatHygiene
using Test: @testset

@testset verbose = true "TestCompatHygiene" begin
    TestCompatHygiene.test_all(AdvRBMs)
end
