module AdvRBMs

import RestrictedBoltzmannMachines as RBMs
using Base: front
using LinearAlgebra: sylvester, dot, Diagonal, eigen
using Statistics: mean
using ValueHistories: MVHistory
using RestrictedBoltzmannMachines: RBM, transfer_sample, inputs_v_to_h, sample_v_from_v,
    ∂free_energy, subtract_gradients, ∂regularize!, batchmean, meanvar_from_inputs,
    _nobs, default_optimizer, fantasy_init, suffstats, minibatches,
    update!, center_gradient, uncenter_step,
    zerosum!, rescale_hidden!, weights, visible, hidden, grad2mean, grad2var, wmean,
    gradmult, ∂logpartition

include("calc_qQ.jl")
include("proj.jl")
include("advpcd.jl")

end # module
