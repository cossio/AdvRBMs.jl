function estimate_q(u::AbstractVector, v::AbstractMatrix)
    @assert length(u) == size(v, 2)
    u_ = u .- mean(u)
    v_ = v .- mean(v; dims=2)
    return v_ * u_ / length(u)
end

function estimate_q(u::AbstractVector, v::AbstractArray)
    q_ = estimate_q(u, reshape(v, :, size(v)[end]))
    return reshape(q_, Base.front(size(v)))
end

function estimate_Q(u::AbstractVector, v::AbstractMatrix)
    @assert length(u) == size(v, 2)
    u_ = u .- mean(u)
    return v * (u_ .* v') / length(u)
end

function estimate_Q(u::AbstractVector, v::AbstractArray)
    v_ = reshape(v, :, size(v)[end])
    Q_ = estimate_Q(u, v_)
    return reshape(Q_, Base.front(size(v))..., Base.front(size(v))...)
end

function estimate_qQ(u::AbstractVector, v::AbstractMatrix)
    @assert length(u) == size(v, 2)
    u_ = u .- mean(u)
    v_ = v .- mean(v; dims=2)
    q = v_ * u_ / length(u)
    Q = v * (u_ .* v') / length(u)
    return (q = q, Q = Q)
end

function estimate_qQ(u::AbstractVector, v::AbstractMatrix)
    @assert length(u) == size(v, 2)
    u_ = u .- mean(u)
    v_ = v .- mean(v; dims=2)
    q = v_ * u_ / length(u)
    Q = v * (u_ .* v') / length(u)
    return (q = q, Q = Q)
end
