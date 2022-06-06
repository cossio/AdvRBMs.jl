const Wts = Union{Nothing,AbstractVector}

function calc_qs(::Type{T}, u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::Wts = nothing) where {T}
    q = calc_q(T, u, v; wts)
    return [collect(selectdim(q, ndims(q), k)) for k in 1:size(q)[end]]
end

function calc_Qs(::Type{T}, u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::Wts = nothing) where {T}
    Q = calc_Q(u, v; wts)
    return [collect(selectdim(Q, ndims(Q), k)) for k in 1:size(Q)[end]]
end

calc_qs(u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::Wts = nothing) = calc_qs(Float64, u, v; wts)
calc_Qs(u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::Wts = nothing) = calc_Qs(Float64, u, v; wts)

# for binary labels
function calc_q(u::AbstractVector{Bool}, v::AbstractArray; wts::Wts = nothing)
    @assert length(u) == size(v, ndims(v)) # same number of examples
    q = calc_q(u, reshape(v, :, length(u)); wts)
    return reshape(q, front(size(v))..., 1)
end

# for binary labels
function calc_q(u::AbstractVector{Bool}, v::AbstractMatrix; wts::Wts = nothing)
    @assert length(u) == size(v, 2) # same number of examples
    @assert 0 < mean(u) < 1 # non-singular
    if isnothing(wts)
        q = (v .- mean(v; dims=2)) * (u .- mean(u)) / length(u)
    else
        @assert length(wts) == length(u)
        v_mean = v * wts / sum(wts)
        u_mean = dot(u, wts) / sum(wts)
        q = (v .- v_mean) * (wts .* (u .- u_mean)) / sum(wts)
    end
    return reshape(q, length(q), 1)
end

# for categorical labels (u is onehot encoded)
function calc_q(u::AbstractMatrix{Bool}, v::AbstractArray; wts::Wts = nothing)
    v_flat = reshape(v, :, size(v)[end])
    q_flat = calc_q(u, v_flat; wts)
    return reshape(q_flat, front(size(v))..., size(q_flat, 2))
end

# for categorical labels (u is onehot encoded)
function calc_q(u::AbstractMatrix{Bool}, v::AbstractMatrix; wts::Wts = nothing)
    @assert size(u, 2) == size(v, 2) # number of samples
    U = u .- wmean(u; dims=2, wts)
    V = v .- wmean(v; dims=2, wts)
    if isnothing(wts)
        q = V * U' / size(v, 2)
    else
        @assert length(wts) == size(v, 2)
        q = V * Diagonal(wts) * U' / sum(wts)
    end
    return q[:, 2:end] # we can drop a row because it is a linear combination of the others
end

# for binary labels
function calc_Q(u::AbstractVector{Bool}, v::AbstractMatrix; wts::Wts = nothing)
    @assert length(u) == size(v, 2)
    U = u .- wmean(u; wts)
    V = v .- wmean(v; wts, dims=2)
    if isnothing(wts)
        Q = V * (U .* V') / length(u)
    else
        Q = V * Diagonal(wts) * (U .* V') / sum(wts)
    end
    return reshape(Q, size(Q)..., 1)
end

# for binary labels
function calc_Q(u::AbstractVector{Bool}, v::AbstractArray; wts::Wts = nothing)
    @assert length(u) == size(v)[end]
    Q = calc_Q(u, reshape(v, :, length(u)); wts)
    return reshape(Q, front(size(v))..., front(size(v))..., 1)
end

# for categorical labels
function calc_Q(u::AbstractMatrix{Bool}, v::AbstractArray; wts::Wts = nothing)
    @assert size(u, 2) == size(v)[end]
    # we can drop a row because it is a linear combination of the others
    Q = zeros(front(size(v))..., front(size(v))..., size(u, 1) - 1)
    for k in 2:size(u, 1)
        selectdim(Q, ndims(Q), k - 1) .= calc_Q(u[k,:], v; wts)
    end
    return Q
end

function calc_q(::Type{T}, u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::Wts = nothing) where {T<:Number}
    q = calc_q(u, v; wts)
    return Array{T}(q)
end

function calc_Q(::Type{T}, u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::Wts = nothing) where {T<:Number}
    Q = calc_Q(u, v; wts)
    return Array{T}(Q)
end

# concatenate across last dimension
batchcat(A::AbstractArray, B::AbstractArray...) = cat(A, B...; dims=ndims(A))

# # for categorical labels
# function calc_Q(u::AbstractMatrix{Bool}, v::AbstractArray; wts::Wts = nothing)
#     @assert length(u) == size(v)[end]
#     Q = calc_Q(u, reshape(v, :, length(u)); wts)
#     return reshape(Q, front(size(v))..., front(size(v))..., 1)
# end
