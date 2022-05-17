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

    q::Union{AbstractArray, Nothing} = nothing, # 1st-order constraints (should be zero-sum for Potts layers)
    Q::Union{AbstractArray, Nothing} = nothing, # 2nd-order constraints
    λq::Real = isnothing(q) ? 0 : Inf, # 1st-order adversarial soft constraint, penalty
    λQ::Real = 0, # 2nd-order adversarial soft constraint, penalty

    # indices of constrained hidden units
    ℋ::CartesianIndices = CartesianIndices(size(rbm.hidden))
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    @assert ϵh ≥ 0

    @assert 0 ≤ λq ≤ Inf # set λq = Inf for hard 1st-order constraint
    @assert 0 ≤ λQ < Inf # hard 2nd-order constraint not supported
    @assert isnothing(q) && iszero(λq) || size(q) == (size(rbm.visible)..., size(q)[end])
    @assert isnothing(Q) && iszero(λQ) || size(Q) == (front(size(q))..., front(size(q))..., size(Q)[end])

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

    if λq == Inf # 1st-order constraint is hard
        # impose 1st-order constraint on initial weights
        rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q)
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
            ∂.w[𝒱, ℋ] .+= λQ .* ∂wQw(rbm.w[𝒱, ℋ], Q)
        end
        if 0 < λq < Inf
            ∂.w[𝒱, ℋ] .+= λq .* ∂qw(rbm.w[𝒱, ℋ], q)
        elseif λq == Inf
            # project the gradient to be orthogonal to q
            ∂.w[𝒱, ℋ] .= kernelproj(∂.w[𝒱, ℋ], q)
        end

        if center
            ∂ = center_gradient(rbm, ∂, ave_v, ave_h)
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

        if λq == Inf
            #= Since the adaptive gradients update and
            the centering might move the weights towards q,
            we project the weights to be orthogonal to q after
            each update. =#
            rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q)
        end

        isnothing(callback) || callback(; rbm, ∂, optim, epoch, batch_idx, vd, wd)
    end
    return rbm
end
