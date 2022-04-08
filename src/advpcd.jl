"""
    advpcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence with constraints.
"""
function advpcd!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    wts = nothing,
    steps::Int = 1, # fantasy chains MC steps
    optim = default_optimizer(_nobs(data), batchsize, epochs),
    vm::AbstractArray = fantasy_init(rbm, batchsize), # fantasy chains
    stats = suffstats(rbm, data; wts), # visible layer sufficient statistics

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers
    center::Bool = true, # center gradients

    # scale hidden unit activations to var(h) = 1
    standardize_hidden::Bool = true,

    # damping for hidden activity statistics tracking
    hidden_damp::Real = batchsize / _nobs(data),
    ϵh = 1e-2, # prevent vanishing var(h)

    callback = nothing, # called for every batch

    q::Union{AbstractArray, Nothing} = nothing, # 1st-order adversarial constraint
    Q::Union{AbstractArray, Nothing} = nothing, # 2nd-order adversarial constraint
    λq::Real = isnothing(q) ? 0 : Inf, # 1st-order adversarial soft constraint, penalty
    λQ::Real = 0, # 2nd-order adversarial soft constraint, penalty

    # indices of constrained hidden units
    ℋ::CartesianIndices = CartesianIndices(size(hidden(rbm)))
)
    @assert size(data) == (size(visible(rbm))..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # we center units using their average activities
    ave_v = batchmean(visible(rbm), data; wts)
    ave_h, var_h = meanvar_from_inputs(hidden(rbm), inputs_v_to_h(rbm, data); wts)

    # indices in visible dimensions
    𝒱 = CartesianIndices(size(visible(rbm)))

    @assert 0 ≤ λq ≤ Inf # set λq = Inf for hard 1st-order constraint
    @assert 0 ≤ λQ < Inf # hard 2nd-order constraint not supported
    @assert isnothing(q) && iszero(λq) || size(q) == size(visible(rbm))
    @assert isnothing(Q) && iszero(λQ) || size(Q) == (size(q)..., size(q)...)

    # gauge constraints
    zerosum && zerosum!(rbm)
    standardize_hidden && rescale_hidden!(rbm, inv.(sqrt.(var_h .+ ϵh)))

    if λq == Inf # 1st-order constraint is hard
        # impose 1st-order constraint on initial weights
        orthogonal_projection!(view(weights(rbm), 𝒱, ℋ), q)
    end

    for epoch in 1:epochs, (batch_idx, (vd, wd)) in enumerate(minibatches(data, wts; batchsize))
        vm .= sample_v_from_v(rbm, vm; steps)
        ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = subtract_gradients(∂d, ∂m)

        λh = grad2mean(hidden(rbm), ∂d.hidden)
        νh = grad2var(hidden(rbm), ∂d.hidden)
        ave_h .= (1 - hidden_damp) * λh .+ hidden_damp .* ave_h
        var_h .= (1 - hidden_damp) * νh .+ hidden_damp .* var_h

        if center
            ∂ = center_gradient(rbm, ∂, ave_v, ave_h)
        end

        # regularize
        ∂reg!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        if 0 < λq < Inf
            view(∂.w, 𝒱, ℋ) .+= λq .* ∂qw(view(weights(rbm), 𝒱, ℋ), q)
        end
        if 0 < λQ < Inf
            view(∂.w, 𝒱, ℋ) .+= λQ .* ∂wQw(view(weights(rbm), 𝒱, ℋ), Q)
        end

        if λq == Inf # hard 1st-order constraint
            # project gradient before feeding it to optimizer algorithm
            orthogonal_projection!(view(∂.w, 𝒱, ℋ), q)
        end

        # compute parameter update step, according to optimizer algorithm
        update!(∂, rbm, optim)

        if center # get step in uncentered parameters
            ∂ = uncenter_step(rbm, ∂, ave_v, ave_h)
        end

        RBMs.update!(rbm, ∂)

        if λq == Inf
            orthogonal_projection!(view(weights(rbm), 𝒱, ℋ), q)
        end

        # respect gauge constraints
        zerosum && zerosum!(rbm)
        standardize_hidden && rescale_hidden!(rbm, inv.(sqrt.(var_h .+ ϵh)))

        isnothing(callback) || callback(; rbm, ∂, optim, epoch, batch_idx, vd, wd)
    end
    return rbm
end
