"""
    orthogonal_projection(w, q)

Projects `w` to the (hyper-)plane orthogonal to `q`.
"""
function orthogonal_projection(w::AbstractArray, q::AbstractArray)
    @assert size(w)[1:ndims(q)] == size(q)
    wp = orthogonal_projection(reshape(w, length(q), :), vec(q))
    return reshape(wp, size(w))
end
orthogonal_projection(w::AbstractVecOrMat, q::AbstractVector) = w - q * (q' * w) / (q' * q)

function orthogonal_projection!(w::AbstractArray, q::AbstractArray)
    w .= orthogonal_projection(w, q)
    return w
end

"""
    ∂qw(w, q)

Derivative of `||q' * w||^2 / 2` with respect to `w`.
"""
function ∂qw(w::AbstractArray, q::AbstractArray)
    @assert size(w)[1:ndims(q)] == size(q)
    ∂ = ∂qw(reshape(w, length(q), :), vec(q))
    return reshape(∂, size(w))
end

function ∂qw(w::AbstractMatrix, q::AbstractVector)
    @assert size(w, 1) == length(q)
    return q * (q' * w)
end

"""
    ∂wQw(w, Q)

Derivative of `||w' * Q * w||^2 / 2` with respect to `w`.
"""
function ∂wQw(w::AbstractArray, Q::AbstractArray)
    @assert iseven(ndims(Q))
    𝒱 = ndims(Q) ÷ 2
    @assert size(w)[1:𝒱] == size(Q)[1:𝒱] == size(Q)[(𝒱 + 1):end]
    N = prod(size(Q)[1:𝒱])
    ∂ = ∂wQw(reshape(w, N, :), reshape(Q, N, N))
    return reshape(∂, size(w))
end

function ∂wQw(w::AbstractMatrix, Q::AbstractMatrix)
    @assert size(w, 1) == size(Q, 1) == size(Q, 2)
    @assert issymmetric(Q)
    Qw = Q * w
    return 2Qw * (w' * Qw)
end

#= *********************
Old stuff I'm not using now.
********************* =#

"""
    sylvester_projection(A, X)

Returns the projection of `X` onto the solution space of `A'X + X'A = 0`.
"""
function sylvester_projection(A::AbstractMatrix, X::AbstractMatrix)
    @assert size(A) == size(X)
    AA = A'*A
    AX = A'*X
    L = sylvester(AA, AA, -(AX + AX'))
    return X - A * L
end

"""
    ∂project!(∂w, q, Q)

Projects gradients `∂w` using the given constraints.
"""
∂project!(::AbstractArray, ::Nothing = nothing, ::Nothing = nothing) = nothing

function ∂project!(∂w::AbstractArray, q::AbstractArray, ::Nothing = nothing)
    @assert size(∂w)[1:ndims(q)] == size(q)
    ∂w .= orthogonal_projection(∂w, q)
    return ∂w
end

function ∂project!(∂w::AbstractArray, ::Nothing, Q::AbstractArray)
    @assert size(Q) == (size(∂w)[1:(ndims(Q) ÷ 2)]..., size(∂w)[1:(ndims(Q) ÷ 2)]...)
    Qw = Q * ∂w
    ∂w .= sylvester_projection(Qw, ∂w)
    return ∂w
end

function ∂project!(∂w::AbstractArray, q::AbstractArray, Q::AbstractArray)
    @assert size(Q) == (size(q)..., size(q)...)
    Qw = Q * ∂w
    Qw .= orthogonal_projection(Qw, q)
    ∂w .= sylvester_projection(Qw, ∂w)
    return ∂w
end
