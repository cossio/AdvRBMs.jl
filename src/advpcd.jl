"""
    advpcd!(rbm, data; q, Q, ...)

Trains the RBM on data using Persistent Contrastive divergence with constraints.
Matrix `q` contains the 1st-order constraints, that `q[...,t]' * W` be small, for each `t`.
Matrix `Q` contains the 2nd-order constraints, that `W' * Q[...,t] * W` be small, for each `t`.
"""
function advpcd!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    wts::Union{AbstractVector, Nothing} = nothing,
    steps::Int = 1, # fantasy chains MC steps
    optim = default_optimizer(_nobs(data), batchsize, epochs),
    stats = suffstats(rbm, data; wts), # visible layer sufficient statistics

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers
    rescale::Bool = true, # scale hidden unit activations to var(h) = 1
    center::Bool = true, # center gradients

    # damping for hidden activity statistics tracking
    damp::Real = 1 // 100,
    ϵh = 1e-2, # prevent vanishing var(h)

    callback = nothing, # called for every batch
    mode::Symbol = :pcd, # :pcd, :cd, or :exact

    vm = fantasy_init(rbm; batchsize, mode), # fantasy chains
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
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    @assert ϵh ≥ 0

    @assert 0 ≤ λQ < Inf # hard 2nd-order constraint not supported
    @assert length(qs) == length(Qs) == length(ℋs)
    @assert empty_intersections(ℋs)

    # indices in visible dimensions
    𝒱 = CartesianIndices(size(rbm.visible))

    # we center units using their average activities
    ave_v = batchmean(rbm.visible, data; wts)
    ave_h, var_h = total_meanvar_from_inputs(rbm.hidden, inputs_h_from_v(rbm, data); wts)
    @assert all(var_h .+ ϵh .> 0)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_hidden!(rbm, sqrt.(var_h .+ ϵh))

    wts_mean = isnothing(wts) ? 1 : mean(wts)

    # impose 1st-order constraint on initial weights
    for (q, ℋ) in zip(qs, ℋs)
        rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q) # 1st-order constraint is hard
    end

    for epoch in 1:epochs, (batch_idx, (vd, wd)) in enumerate(minibatches(data, wts; batchsize, shuffle))
        ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
        ∂m = ∂logpartition(rbm; vd, vm, wd, mode, steps)
        ∂ = subtract_gradients(∂d, ∂m)

        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ = gradmult(∂, batch_weight)

        ave_h_batch = grad2ave(rbm.hidden, ∂d.hidden)
        var_h_batch = grad2var(rbm.hidden, ∂d.hidden)
        damp_eff = damp ^ batch_weight
        ave_h .= (1 - damp_eff) * ave_h_batch .+ damp_eff .* ave_h
        var_h .= (1 - damp_eff) * var_h_batch .+ damp_eff .* var_h
        @assert all(var_h .+ ϵh .> 0)

        # regularize
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        if 0 < λQ < Inf
            for (Q, ℋ) in zip(Qs, ℋs)
                ∂.w[𝒱, ℋ] .+= λQ .* ∂wQw(rbm.w[𝒱, ℋ], Q)
            end
        end

        if center
            ∂ = center_gradient(rbm, ∂, ave_v, ave_h)
        end

        for (q, ℋ) in zip(qs, ℋs)
            # 1st-order constraint is hard
            ∂.w[𝒱, ℋ] .= kernelproj(∂.w[𝒱, ℋ], q)
        end

        # compute parameter update step, according to optimizer algorithm
        update!(∂, rbm, optim)

        if center # get step in uncentered parameters
            ∂ = uncenter_step(rbm, ∂, ave_v, ave_h)
        end

        RBMs.update!(rbm, ∂)

        # respect gauge constraints
        zerosum && zerosum!(rbm)
        rescale && rescale_hidden!(rbm, sqrt.(var_h .+ ϵh))

        for (q, ℋ) in zip(qs, ℋs)
            #= Since the adaptive gradients update and
            the centering might move the weights towards q,
            we project the weights to be orthogonal to q after
            each update. =#
            rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q) # 1st-order constraint is hard
        end

        isnothing(callback) || callback(; rbm, ∂, optim, epoch, batch_idx, vd, wd)
    end
    return rbm
end

function default_qs(rbm::RBM)
    q = similar(rbm.w, eltype(rbm.w), size(rbm.visible)..., 1)
    return empty([q])
end

function default_Qs(rbm::RBM, qs::AbstractVector{<:AbstractArray{<:Real}})
    return [Zeros{eltype(q)}(size(rbm.visible)..., size(rbm.visible)..., 1) for q in qs]
end

function default_ℋs(rbm::RBM, qs::AbstractVector{<:AbstractArray{<:Real}})
    return [CartesianIndices(size(rbm.hidden)) for q in qs]
end
