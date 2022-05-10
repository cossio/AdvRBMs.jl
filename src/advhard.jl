"""
    advhard!(rbm, data; q, Q, ...)

Trains the RBM on data using Persistent Contrastive divergence with constraints.
Matrix `q` contains the 1st-order constraints, `q[...,t]' * W = 0`, for each `t`.
Matrix `Q` contains the 2nd-order constraints, `W' * Q[...,t] * W = 0`, for each `t`.
(Only one matrix `Q` supported for now.).
See RestrictedBoltzmannMachines.jl for explanation of the other keyword arguments.
"""
function advhard!(
    rbm::RBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    wts = nothing,
    steps::Int = 1,
    optim = default_optimizer(_nobs(data), batchsize, epochs),
    vm::AbstractArray = fantasy_init(rbm, batchsize),
    stats = suffstats(rbm, data; wts),

    # regularization
    l2_fields::Real = 0,
    l1_weights::Real = 0,
    l2_weights::Real = 0,
    l2l1_weights::Real = 0,

    # gauge
    center::Bool = true,
    standardize_hidden::Bool = true,
    hidden_damp::Real = batchsize / _nobs(data),
    ϵh::Real = 1e-2, # prevent vanishing var(h)

    callback = nothing, # called for every batch

    q::Union{AbstractArray, Nothing} = nothing, # 1st-order constraints
    Q::Union{AbstractArray, Nothing} = nothing, # 2nd-order constraints
    # indices of constrained hidden units
    ℋ::CartesianIndices = CartesianIndices(size(rbm.hidden))
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # we center units using their average activities
    ave_v = batchmean(rbm.visible, data; wts)
    ave_h, var_h = meanvar_from_inputs(rbm.hidden, inputs_v_to_h(rbm, data); wts)

    # indices in visible dimensions
    𝒱 = CartesianIndices(size(rbm.visible))

    @assert isnothing(q) || size(q) == (size(rbm.visible)..., size(q)[end])
    @assert isnothing(Q) || size(Q) == (front(size(q))..., front(size(q))..., size(Q)[end])
    @assert isnothing(Q) || size(Q)[end] == 1 # only one Q matrix supported for now

    # gauge constraints
    if standardize_hidden
        rescale_hidden!(rbm, inv.(sqrt.(var_h .+ ϵh)))
    end

    wts_mean = mean_maybe(wts)

    # impose constraints on initial weights
    rbm.w[𝒱, ℋ] .= project(rbm.w[𝒱, ℋ], q, Q)

    for epoch in 1:epochs, (batch_idx, (vd, wd)) in enumerate(minibatches(data, wts; batchsize))
        vm .= sample_v_from_v(rbm, vm; steps)
        ∂d = ∂free_energy(rbm, vd; wts = wd, stats)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = subtract_gradients(∂d, ∂m)

        batch_weight = mean_maybe(wd) / wts_mean
        ∂ = gradmult(∂, batch_weight)

        damp = hidden_damp ^ batch_weight
        λh = grad2mean(rbm.hidden, ∂d.hidden)
        νh = grad2var(rbm.hidden, ∂d.hidden)
        ave_h .= (1 - damp) * λh .+ damp * ave_h
        var_h .= (1 - damp) * νh .+ damp * var_h

        if center
            ∂ = center_gradient(rbm, ∂, ave_v, ave_h)
        end

        # regularize
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        if !isnothing(q) && isnothing(Q) # 1st-order constraint
            # project gradient before feeding it to optimizer algorithm
            ∂.w[𝒱, ℋ] .= project_1st(∂.w[𝒱, ℋ], q)
        end

        # compute parameter update step, according to optimizer algorithm
        update!(∂, rbm, optim)

        if center # get step in uncentered parameters
            ∂ = uncenter_step(rbm, ∂, ave_v, ave_h)
        end

        RBMs.update!(rbm, ∂)

        if !isnothing(q)
            rbm.w[𝒱, ℋ] .= kernelproj(rbm.w[𝒱, ℋ], q)
        end

        # respect gauge constraints
        if standardize_hidden
            rescale_hidden!(rbm, inv.(sqrt.(var_h .+ ϵh)))
        end

        if !isnothing(callback)
            callback(; rbm, ∂, optim, epoch, batch_idx, vd, wd)
        end
    end
    return rbm
end
