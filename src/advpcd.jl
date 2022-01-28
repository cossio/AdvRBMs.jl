"""
    advpcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence with constraints.
"""
function advpcd!(rbm::RBMs.RBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optimizer = Flux.ADAM(), # optimizer algorithm
    history::ValueHistories.MVHistory = ValueHistories.MVHistory(), # stores training log
    wts = nothing, # data point weights
    steps::Int = 1, # Monte Carlo steps to update fantasy particles
    q::Union{AbstractArray, Nothing} = nothing, # 1st-order constraint
    Q::Union{AbstractArray, Nothing} = nothing, # 2nd-order constraint
    λQ::Real = Inf, # 2nd-order penalty
    c1::CartesianIndices = CartesianIndices(size(rbm.hidden)), # hidden units affected by 1st order constraint
    c2::CartesianIndices = CartesianIndices(size(rbm.hidden)), # hidden units affected by 2nd order constraint
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    stats = RBMs.sufficient_statistics(rbm.visible, data; wts)

    # initialize fantasy chains by sampling visible layer
    vm = RBMs.transfer_sample(rbm.visible, falses(size(rbm.visible)..., batchsize))

    C1 = CartesianIndices(axes(CartesianIndices(size(rbm.visible)))..., axes(c1)...)
    C2 = CartesianIndices(axes(CartesianIndices(size(rbm.visible)))..., axes(c2)...)

    for epoch in 1:epochs
        batches = RBMs.minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            vm = RBMs.sample_v_from_v(rbm, vm; steps = steps)
            ∂ = RBMs.∂contrastive_divergence(rbm, vd, vm; wd, stats)
            if λQ == Inf
                ∂project!(view(∂.w, mask_units), q, Q)
            else
                ∂.w .+= λQ .* ∂wQw(view(rbm.w, mask_units), Q)
            end
            RBMs.update!(optimizer, rbm, ∂)
            push!(history, :∂, RBMs.gradnorms(∂))
        end
        project!(view(rbm.w, mask_units), q)

        lpl = RBMs.wmean(log_pseudolikelihood(rbm, data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"
    end
    return history
end
