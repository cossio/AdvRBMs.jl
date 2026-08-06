# TODO: Update this file!

"""
    advpcdsoft!(rbm, data; q, Q, ...)

Trains the RBM on data using Persistent Contrastive divergence with (soft) constraints.
Matrix `q` contains the 1st-order constraints, that `q[...,t]' * W` be small, for each `t`.
Matrix `Q` contains the 2nd-order constraints, that `W' * Q[...,t] * W` be small, for each `t`.
"""
function advpcdsoft!(
        rbm::RBM,
        data::AbstractArray;
        batchsize::Int = 1,
        epochs::Int = 1,
        wts = nothing,
        steps::Int = 1, # fantasy chains MC steps
        optim = Adam(),
        vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),
        moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer

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
        hidden_damp::Real = batchsize / size(data)[end],
        ϵh = 1.0e-2, # prevent vanishing var(h)

        callback = nothing, # called for every batch

        qs::AbstractVector = [], # 1st-order constraints
        Qs::AbstractVector = [], # 2nd-order constraints
        λq::Real = 0, # 1st-order adversarial soft constraint, penalty
        λQ::Real = 0, # 2nd-order adversarial soft constraint, penalty

        # indices of constrained hidden units
        ℋ1::AbstractVector{<:CartesianIndices} = [CartesianIndices(size(rbm.hidden))],
        ℋ2::AbstractVector{<:CartesianIndices} = [CartesianIndices(size(rbm.hidden))]
    )
    error("this is outdated, don't call me ... update!!")

    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || size(data)[end] == length(wts)

    # we center units using their average activities
    ave_v = batchmean(rbm.visible, data; wts)
    ave_h, var_h = total_meanvar_from_inputs(rbm.hidden, inputs_h_from_v(rbm, data); wts)

    # indices in visible dimensions
    𝒱 = CartesianIndices(size(rbm.visible))

    @assert 0 ≤ λq < Inf
    @assert 0 ≤ λQ < Inf
    for q in qs
        @assert size(q) == size(rbm.visible)
    end
    for Q in Qs
        @assert size(Q) == (size(q)..., size(q)...)
    end
    @assert length(ℋ1) == length(qs)
    @assert length(ℋ2) == length(Qs)

    # gauge constraints
    zerosum && zerosum!(rbm)
    standardize_hidden && rescale_hidden!(rbm, inv.(sqrt.(var_h .+ ϵh)))

    for epoch in 1:epochs, (batch_idx, (vd, wd)) in enumerate(minibatches(data, wts; batchsize))
        vm .= sample_v_from_v(rbm, vm; steps)
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        λh = grad2ave(rbm.hidden, -∂d.hidden)
        νh = grad2var(rbm.hidden, -∂d.hidden)
        ave_h .= (1 - hidden_damp) * λh .+ hidden_damp .* ave_h
        var_h .= (1 - hidden_damp) * νh .+ hidden_damp .* var_h

        if center
            ∂ = center_gradient(rbm, ∂, ave_v, ave_h)
        end

        # regularize
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        if 0 < λq < Inf
            for (q, ℋ) in zip(qs, ℋ1)
                ∂.w[𝒱, ℋ] .+= λq .* ∂qw(rbm.w[𝒱, ℋ], q)
            end
        end
        if 0 < λQ < Inf
            for (Q, ℋ) in zip(Qs, ℋ2)
                ∂.w[𝒱, ℋ] .+= λQ .* ∂wQw(rbm.w[𝒱, ℋ], Q)
            end
        end

        # compute parameter update step, according to optimizer algorithm
        update!(∂, rbm, optim)

        if center # get step in uncentered parameters
            ∂ = uncenter_step(rbm, ∂, ave_v, ave_h)
        end

        RBMs.update!(rbm, ∂)

        # respect gauge constraints
        zerosum && zerosum!(rbm)
        standardize_hidden && rescale_hidden!(rbm, inv.(sqrt.(var_h .+ ϵh)))

        isnothing(callback) || callback(; rbm, ∂, optim, epoch, batch_idx, vd, wd)
    end
    return rbm
end
