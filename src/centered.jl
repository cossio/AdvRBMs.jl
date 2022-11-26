function advpcd!(
    rbm::CenteredRBM,
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

    callback = Returns(nothing), # called for every batch

    vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),
    shuffle::Bool = true,

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

    @assert 0 ≤ λQ < Inf # hard 2nd-order constraint not supported
    @assert length(qs) == length(Qs) == length(ℋs)
    @assert empty_intersections(ℋs)

    # indices in visible dimensions
    𝒱 = CartesianIndices(size(rbm.visible))

    # gauge constraints
    zerosum && zerosum!(rbm)

    wts_mean = isnothing(wts) ? 1 : mean(wts)

    # impose hard 1st-order constraint on initial weights
    for (q, ℋ) in zip(qs, ℋs)
        rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q)
    end

    # define parameters for Optimiser and initialize optimiser state
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # update Markov chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

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

        # respect gauge constraints
        zerosum && zerosum!(rbm)

        # 1st-order constraint is hard, project weights
        for (q, ℋ) in zip(qs, ℋs)
            rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q)
        end

        callback(; rbm, ∂, optim, iter, vm, vd, wd)
    end
    return rbm
end

function default_qs(rbm::Union{RBM, CenteredRBM})
    q = similar(rbm.w, eltype(rbm.w), size(rbm.visible)..., 1)
    return empty([q])
end

function default_Qs(rbm::Union{RBM, CenteredRBM}, qs::AbstractVector{<:AbstractArray{<:Real}})
    return [Zeros{eltype(q)}(size(rbm.visible)..., size(rbm.visible)..., 1) for q in qs]
end

function default_ℋs(rbm::Union{RBM, CenteredRBM}, qs::AbstractVector{<:AbstractArray{<:Real}})
    return [CartesianIndices(size(rbm.hidden)) for q in qs]
end
