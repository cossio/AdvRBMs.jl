import AdvRBMs
import ExplicitImports
import FillArrays
import LinearAlgebra
import Optimisers
import Statistics
import StatsBase
using Test: @test, @testset

@testset "ExplicitImports" begin
    ExplicitImports.test_explicit_imports(
        AdvRBMs;
        # AdvRBMs deliberately builds on RestrictedBoltzmannMachines internals,
        # which are not public, so the publicness check covers only the imports
        # from the remaining dependencies.
        all_explicit_imports_are_public = (
            from = (Base, LinearAlgebra, Statistics, StatsBase, FillArrays, Optimisers),
        ),
    )
end
