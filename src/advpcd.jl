"""
    advpcd!(rbm, data; qs, Qs, ℋs, ...)

Trains the RBM on data using Persistent Contrastive divergence with constraints.
Each entry `q` of `qs` contains 1st-order constraints, that `q[..., t]' * W` be small, for each `t`.
Each entry `Q` of `Qs` contains 2nd-order constraints, that `W' * Q[..., t] * W` be small, for each `t`.
Each constraint group acts on the hidden units selected by the matching entry of `ℋs`.
"""
function advpcd!(
        rbm::Union{RBM, CenteredRBM},
        data::AbstractArray;
        batchsize::Int = 1,
        iters::Int = 1, # number of parameter updates
        wts::Union{AbstractVector, Nothing} = nothing,
        steps::Int = 1, # fantasy chains MC steps
        optim = Adam(),
        moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer

        # regularization
        l2_fields::Real = 0, # visible fields L2 regularization
        l1_weights::Real = 0, # weights L1 regularization
        l2_weights::Real = 0, # weights L2 regularization
        l2l1_weights::Real = 0, # weights L2/L1 regularization

        # gauge
        zerosum::Bool = true, # zerosum gauge for Potts layers
        rescale::Bool = true, # normalize weights to unit norm

        callback = Returns(nothing), # called for every batch

        vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),
        shuffle::Bool = true,

        # damping to update hidden statistics for centering
        hidden_offset_damping::Real = 1 // 100,

        # constraints are given as a list, where each entry describes the constraints applied
        # to a group of hidden units (the groups must be exclusive)
        # For Potts units, q, Q should themselves be zerosum!
        qs::AbstractVector{<:AbstractArray{<:Real}} = default_qs(rbm), # 1st-order constraints
        Qs::AbstractVector{<:AbstractArray{<:Real}} = default_Qs(rbm, qs), # 2nd-order constraints
        λQ::Real = 0, # 2nd-order adversarial soft constraint, penalty
        # indices of constrained hidden units in each group (the groups must not intersect)
        ℋs::AbstractVector{<:CartesianIndices} = default_ℋs(rbm, qs)
    )
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    isnothing(wts) || @assert size(data)[end] == length(wts)
    @assert 0 ≤ hidden_offset_damping ≤ 1

    @assert 0 ≤ λQ < Inf # hard 2nd-order constraint not supported
    @assert length(qs) == length(Qs) == length(ℋs)
    @assert empty_intersections(ℋs)

    # indices in visible dimensions
    𝒱 = CartesianIndices(size(rbm.visible))

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    # initial centering from data
    if rbm isa CenteredRBM
        center_from_data!(rbm, data)
    end

    wts_mean = isnothing(wts) ? 1 : mean(wts)

    # impose hard 1st-order constraint on initial weights
    qs_inv = map(pseudo_inv_of_q, qs)
    for (q, ℋ, qinv) in zip(qs, ℋs, qs_inv)
        rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q; qinv)
    end

    # define parameters for Optimiser and initialize optimiser state
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # update Markov chains
        vm = sample_v_from_v(rbm, vm; steps)

        # gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # correct weighted minibatch bias
        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ *= batch_weight

        # regularize
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # 2nd order constraint is soft, update gradient accordingly
        if 0 < λQ < Inf
            for (Q, ℋ) in zip(Qs, ℋs)
                ∂.w[𝒱, ℋ] .+= λQ .* ∂wQw(rbm.w[𝒱, ℋ], Q)
            end
        end

        # feed gradient to Optimiser rule and update parameters
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        # centering (damped update of hidden offsets towards <h>_d from minibatch)
        if rbm isa CenteredRBM
            center_hidden_from_data!(rbm, vd; wts = wd, damping = hidden_offset_damping)
        end

        # respect gauge constraints
        zerosum && zerosum!(rbm)
        rescale && rescale_weights!(rbm)

        # 1st-order constraint is hard, project weights
        for (q, ℋ, qinv) in zip(qs, ℋs, qs_inv)
            rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q; qinv)
        end

        callback(; rbm, ∂, optim, iter, vm, vd, wd)
    end
    return rbm
end
