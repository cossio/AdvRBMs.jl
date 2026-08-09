function calc_qs(::Type{T}, u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(size(u)[end])) where {T}
    q = calc_q(T, u, v; wts)
    # k:k keeps the trailing singleton constraint dimension expected by advpcd!
    return [collect(selectdim(q, ndims(q), k:k)) for k in 1:size(q)[end]]
end

function calc_Qs(::Type{T}, u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(size(u)[end])) where {T}
    Q = calc_Q(T, u, v; wts)
    return [collect(selectdim(Q, ndims(Q), k:k)) for k in 1:size(Q)[end]]
end

calc_qs(u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(size(u)[end])) = calc_qs(Float64, u, v; wts)
calc_Qs(u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(size(u)[end])) = calc_Qs(Float64, u, v; wts)

# for binary labels
function calc_q(u::AbstractVector{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(length(u)))
    @assert length(u) == size(v, ndims(v)) # same number of examples
    q = calc_q(u, reshape(v, :, length(u)); wts)
    return reshape(q, front(size(v))..., 1)
end

# for binary labels
function calc_q(u::AbstractVector{Bool}, v::AbstractMatrix; wts::AbstractVector{<:Real} = Trues(length(u)))
    @assert length(u) == size(v, 2) # same number of examples
    @assert length(wts) == length(u)
    @assert 0 < mean(u) < 1 # non-singular
    U = u .- wmean(u; wts)
    V = v .- wmean(v; wts)
    q = V * (wts .* U) / sum(wts)
    return reshape(q, length(q), 1)
end

# for categorical labels (u is onehot encoded)
function calc_q(u::AbstractMatrix{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(size(u, 2)))
    v_flat = reshape(v, :, size(v)[end])
    q_flat = calc_q(u, v_flat; wts)
    return reshape(q_flat, front(size(v))..., size(q_flat, 2))
end

# for categorical labels (u is onehot encoded)
function calc_q(u::AbstractMatrix{Bool}, v::AbstractMatrix; wts::AbstractVector{<:Real} = Trues(size(u, 2)))
    @assert size(u, 2) == size(v, 2) # number of samples
    @assert length(wts) == size(v, 2)
    U = u .- wmean(u; wts)
    V = v .- wmean(v; wts)
    q = V * Diagonal(wts) * U' / sum(wts)
    return q[:, 2:end] # we can drop a row because it is a linear combination of the others
end

# for binary labels
function calc_Q(u::AbstractVector{Bool}, v::AbstractMatrix; wts::AbstractVector{<:Real} = Trues(length(u)))
    @assert length(u) == size(v, 2)
    @assert length(wts) == length(u)
    U = u .- wmean(u; wts)
    V = v .- wmean(v; wts)
    Q = V * Diagonal(wts) * (U .* V') / sum(wts)
    return reshape(Q, size(Q)..., 1)
end

# for binary labels
function calc_Q(u::AbstractVector{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(length(u)))
    @assert length(u) == size(v)[end]
    Q = calc_Q(u, reshape(v, :, length(u)); wts)
    return reshape(Q, front(size(v))..., front(size(v))..., 1)
end

# for categorical labels
function calc_Q(u::AbstractMatrix{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(size(u, 2)))
    @assert size(u, 2) == size(v)[end]
    # we can drop a row because it is a linear combination of the others
    Q = zeros(front(size(v))..., front(size(v))..., size(u, 1) - 1)
    for k in 2:size(u, 1)
        selectdim(Q, ndims(Q), k - 1) .= calc_Q(u[k, :], v; wts)
    end
    return Q
end

function calc_q(::Type{T}, u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(size(u)[end])) where {T <: Number}
    q = calc_q(u, v; wts)
    return Array{T}(q)
end

function calc_Q(::Type{T}, u::AbstractVecOrMat{Bool}, v::AbstractArray; wts::AbstractVector{<:Real} = Trues(size(u)[end])) where {T <: Number}
    Q = calc_Q(u, v; wts)
    return Array{T}(Q)
end
