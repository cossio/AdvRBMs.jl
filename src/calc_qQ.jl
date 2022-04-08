function calc_q(u::AbstractVector{<:Bool}, v::AbstractArray)
    @assert length(u) == size(v, ndims(v)) # same number of examples
    q = calc_q(u, reshape(v, :, length(u)))
    return reshape(q, Base.front(size(v)))
end

function calc_q(u::AbstractVector{<:Bool}, v::AbstractMatrix)
    @assert length(u) == size(v, 2) # same number of examples
    return (v .- mean(v; dims=2)) * (u .- mean(u)) / length(u)
end

function calc_Q(u::AbstractVector{<:Bool}, v::AbstractMatrix)
    @assert length(u) == size(v, 2)
    u_ = u .- mean(u)
    return v * (u_ .* v') / length(u)
end

function calc_Q(u::AbstractVector{<:Bool}, v::AbstractArray)
    @assert length(u) == size(v)[end]
    Q = calc_Q(u, reshape(v, :, length(u)))
    return reshape(Q, Base.front(size(v))..., Base.front(size(v))...)
end

function calc_qQ(u::AbstractVector{<:Bool}, v::AbstractMatrix)
    @assert length(u) == size(v, 2)
    u_ = u .- mean(u)
    v_ = v .- mean(v; dims=2)
    q = v_ * u_ / length(u)
    Q = v * (u_ .* v') / length(u)
    return (q = q, Q = Q)
end

function calc_qQ(u::AbstractVector{<:Bool}, v::AbstractArray)
    @assert length(u) == size(v)[end]
    q_, Q_ = calc_qQ(u, reshape(v, :, size(v)[end]))
    q = reshape(q_, Base.front(size(v)))
    Q = reshape(Q_, Base.front(size(v))..., Base.front(size(v))...)
    return (q = q, Q = Q)
end