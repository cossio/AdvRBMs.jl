module AdvRBMs

import RestrictedBoltzmannMachines as RBMs
using Base: front, tail
using LinearAlgebra: sylvester, dot, Diagonal, eigen, pinv
using Statistics: mean
using ValueHistories: MVHistory
using FillArrays: Zeros, Falses
using Optimisers: AbstractRule, setup, update!, Adam
using RestrictedBoltzmannMachines: RBM, AbstractLayer, Binary, Spin, Potts,
    inputs_h_from_v, sample_v_from_v,
    ∂free_energy, ∂regularize!, batchmean, total_meanvar_from_inputs,
    sample_from_inputs, moments_from_samples, infinite_minibatches,
    zerosum!, rescale_hidden!, wmean

include("calc_qQ.jl")
include("proj.jl")
include("advpcd.jl")
include("util.jl")

end # module
