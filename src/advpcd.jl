"""
    advpcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence with constraints.
"""
function advpcd!(rbm::RBMs.RBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    # optimizer algorithm
    optimizer = Flux.ADAM(),
    # stores training log
    history::ValueHistories.MVHistory = ValueHistories.MVHistory(),
    # data point weights
    wts = nothing,
    # Monte Carlo steps to update fantasy particles
    steps::Int = 1,
    # initial state of fantasy chains (default: sample visible layer)
    vm::AbstractArray = transfer_sample(rbm.visible, falses(size(rbm.visible)..., batchsize)),
    # 1st-order constraint
    q::Union{AbstractArray, Nothing} = nothing,
    # 2nd-order constraint
    Q::Union{AbstractArray, Nothing} = nothing,
    # 2nd-order penalty
    λQ::Real = Inf,
    # indices of constrained hidden units
    μc::CartesianIndices = CartesianIndices(size(rbm.hidden))
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    # sufficient statistics for visible layer (e.g., <v>_d)
    stats = RBMs.sufficient_statistics(rbm.visible, data; wts)
    # indices of visible dimensions
    𝒱 = CartesianIndices(size(rbm.visible))

    for epoch in 1:epochs
        batches = RBMs.minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            vm = RBMs.sample_v_from_v(rbm, vm; steps = steps)
            ∂ = RBMs.∂contrastive_divergence(rbm, vd, vm; wd, stats)
            if λQ == Inf
                ∂project!(view(∂.w, 𝒱, μc), q, Q)
            else
                ∂.w .+= λQ .* ∂wQw(view(rbm.w, 𝒱, μc), Q)
            end
            RBMs.update!(optimizer, rbm, ∂)
            push!(history, :∂, RBMs.gradnorms(∂))
        end
        project!(view(rbm.w, 𝒱, μc), q)

        lpl = RBMs.wmean(RBMs.log_pseudolikelihood(rbm, data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"
    end
    return history
end
