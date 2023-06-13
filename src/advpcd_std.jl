"""
    advpcd!(rbm, data; q, Q, ...)

Trains the RBM on data using Persistent Contrastive divergence with constraints.
Matrix `q` contains the 1st-order constraints, that `q[...,t]' * W` be small, for each `t`.
Matrix `Q` contains the 2nd-order constraints, that `W' * Q[...,t] * W` be small, for each `t`.
"""
function advpcd!(
    rbm::StandardizedRBM,
    data::AbstractArray;

    batchsize::Int = 1,
    shuffle::Bool = true,

    iters::Int = 1, # number of parameter updates

    steps::Int = 1, # fantasy chains MC steps
    vm::AbstractArray = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),

    moments = moments_from_samples(rbm.visible, data), # sufficient statistics for visible layer

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # "pseudocount" for estimating variances of v and h and damping
    damping::Real = 1//100,
    ϵv::Real = 0, ϵh::Real = 0,

    optim::AbstractRule = Adam(),
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
    state = setup(optim, ps),

    # Absorb the scale_h into the hidden unit activation (for continuous hidden units).
    # Results in hidden units with var(h) ~ 1.
    rescale_hidden::Bool = true,

    callback = Returns(nothing),

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers

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
    @assert 0 ≤ damping ≤ 1

    @assert 0 ≤ λQ < Inf # hard 2nd-order constraint not supported
    @assert length(qs) == length(Qs) == length(ℋs)
    @assert empty_intersections(ℋs)

    standardize_visible_from_data!(rbm, data; ϵ = ϵv)

    # indices in visible dimensions
    𝒱 = CartesianIndices(size(rbm.visible))

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale_hidden && rescale_hidden_activations!(rbm)

    # initial centering from data
    if rbm isa CenteredRBM
        center_from_data!(rbm, data)
    end

    # impose hard 1st-order constraint on initial weights
    qs_inv = map(pseudo_inv_of_q, qs)
    for (q, ℋ, qinv) in zip(qs, ℋs, qs_inv)
        rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q; qinv)
    end

    for (iter, (vd,)) in zip(1:iters, infinite_minibatches(data; batchsize, shuffle))
        # update Markov chains
        vm = sample_v_from_v(rbm, vm; steps)

        # update standardization
        standardize_hidden_from_v!(rbm, vd; damping, ϵ=ϵh)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

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

        # respect gauge constraints
        rescale_hidden && rescale_hidden_activations!(rbm)
        zerosum && zerosum!(rbm)

        # 1st-order constraint is hard, project weights
        for (q, ℋ, qinv) in zip(qs, ℋs, qs_inv)
            rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q; qinv)
        end

        callback(; rbm, optim, state, ps, iter, vm, vd, ∂)
    end

    return state, ps
end
