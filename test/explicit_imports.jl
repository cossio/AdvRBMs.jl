import AdvRBMs
import ExplicitImports
import FillArrays
import LinearAlgebra
import Optimisers
import Statistics
using Test: @test, @testset

@testset "ExplicitImports" begin
    # Julia 1.10 cannot represent `public` bindings. These documented APIs are
    # marked public by their owners on Julia 1.11+, where the ignore is empty.
    public_imports_without_legacy_metadata =
        VERSION < v"1.11" ? (:front, :setup, :update!) : ()
    ExplicitImports.test_explicit_imports(
        AdvRBMs;
        # AdvRBMs deliberately builds on RestrictedBoltzmannMachines internals,
        # which are not public, so the publicness check covers only the imports
        # from the remaining dependencies.
        all_explicit_imports_are_public = (
            from = (Base, LinearAlgebra, Statistics, FillArrays, Optimisers),
            ignore = public_imports_without_legacy_metadata,
        ),
    )
end
