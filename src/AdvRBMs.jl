module AdvRBMs

using Base: front
using LinearAlgebra: dot, Diagonal, pinv
using Statistics: mean
using StatsBase: weights
using FillArrays: Zeros, Falses
using Optimisers: AbstractRule, setup, update!, Adam
using RestrictedBoltzmannMachines: StandardizedRBM, standardize_visible_from_data!,
    standardize_hidden_from_v!, rescale_hidden_activations!
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: CenteredRBM, center_hidden_from_data!, center_from_data!
using RestrictedBoltzmannMachines: sample_v_from_v,
    ∂free_energy, ∂regularize!,
    sample_from_inputs, moments_from_samples, infinite_minibatches,
    zerosum!, rescale_weights!

include("calc_qQ.jl")
include("proj.jl")
include("advpcd.jl")
include("advpcd_std.jl")
include("util.jl")
include("default_q.jl")

end # module
