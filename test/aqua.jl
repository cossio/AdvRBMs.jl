import Aqua
import AdvRBMs
using Test: @testset

@testset verbose = true "aqua" begin
    Aqua.test_all(AdvRBMs; ambiguities = false)
end
