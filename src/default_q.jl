function default_qs(rbm::Union{RBM, CenteredRBM, StandardizedRBM})
    q = similar(rbm.w, eltype(rbm.w), size(rbm.visible)..., 1)
    return empty([q])
end

function default_Qs(rbm::Union{RBM, CenteredRBM, StandardizedRBM}, qs::AbstractVector{<:AbstractArray{<:Real}})
    return [Zeros{eltype(q)}(size(rbm.visible)..., size(rbm.visible)..., 1) for q in qs]
end

function default_ℋs(rbm::Union{RBM, CenteredRBM, StandardizedRBM}, qs::AbstractVector{<:AbstractArray{<:Real}})
    return [CartesianIndices(size(rbm.hidden)) for q in qs]
end
