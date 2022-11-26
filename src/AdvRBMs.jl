module AdvRBMs

import RestrictedBoltzmannMachines as RBMs
using Base: front, tail
using LinearAlgebra: sylvester, dot, Diagonal, eigen, pinv
using Statistics: mean
using ValueHistories: MVHistory
using FillArrays: Zeros, Falses
using Flux: Adam
using RestrictedBoltzmannMachines: RBM, AbstractLayer, Binary, Spin, Potts,
    inputs_h_from_v, sample_v_from_v,
    ∂free_energy, ∂regularize!, batchmean, total_meanvar_from_inputs,
    sample_from_inputs, moments_from_samples, minibatches,
    update!, center_gradient, uncenter_step,
    zerosum!, rescale_hidden!, grad2ave, grad2var, wmean,
    ∂logpartition

include("calc_qQ.jl")
include("proj.jl")
include("advpcd.jl")
include("util.jl")

end # module
