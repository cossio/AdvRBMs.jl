function estimate_q(u::AbstractArray{<:Bool}, v::AbstractArray)
    @assert size(u, ndims(u)) == size(v, ndims(v)) # same number of examples
    uc = u .- Statistics.mean(u; dims=ndims(u))
    vc = v .- Statistics.mean(v; dims=ndims(v))
    u_ = reshape(uc, size(u)[1:(end - 1)]..., ones(Int, ndims(v) - 1)..., size(u)[end])
    v_ = reshape(vc, ones(Int, ndims(u) - 1)..., size(v)[1:(end - 1)]..., size(v)[end])
    @assert ndims(u_) == ndims(v_)
    q = dropdims(Statistics.mean(u_ .* v_; dims=ndims(v_)); dims=ndims(v_))
    return q
end

function estimate_Q(u::AbstractVector, v::AbstractMatrix)
    @assert length(u) == size(v, 2)
    u_ = u .- Statistics.mean(u)
    return v * (u_ .* v') / length(u)
end

function estimate_Q(u::AbstractVector, v::AbstractArray)
    v_ = reshape(v, :, size(v)[end])
    Q_ = estimate_Q(u, v_)
    return reshape(Q_, Base.front(size(v))..., Base.front(size(v))...)
end

function estimate_qQ(u::AbstractVector, v::AbstractMatrix)
    @assert length(u) == size(v, 2)
    u_ = u .- Statistics.mean(u)
    v_ = v .- Statistics.mean(v; dims=2)
    q = v_ * u_ / length(u)
    Q = v * (u_ .* v') / length(u)
    return (q = q, Q = Q)
end

function estimate_qQ(u::AbstractVector, v::AbstractArray)
    @assert length(u) == size(v)[end]
    q_, Q_ = estimate_qQ(u, reshape(v, :, size(v)[end]))
    q = reshape(q_, Base.front(size(v)))
    Q = reshape(Q_, Base.front(size(v))..., Base.front(size(v))...)
    return (q = q, Q = Q)
end
