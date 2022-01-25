"""
    project_plane(w, q)

Projects `w` to the plane orthogonal to `q`.
"""
function project_plane(w::AbstractArray, q::AbstractArray)
    @assert size(w)[1:ndims(q)] == size(q)
    proj = project_plane(reshape(w, length(q), :), vec(q))
    return reshape(proj, size(w))
end

project_plane(w::AbstractVecOrMat, q::AbstractVector) = w - q * (q' * w) / (q' * q)

"""
    ∂wQw(w, Q)

Derivative of `norm(w' * Q * w)^2 / 2` with respect to `w`.
"""
function ∂wQw(w::AbstractArray, Q::AbstractArray)
    @assert iseven(ndims(Q))
    Dv = ndims(Q) ÷ 2
    @assert size(w)[1:Dv] == size(Q)[1:Dv] == size(Q)[(Dv + 1):end]
    N = prod(size(Q)[1:Dv])
    Qmat = reshape(Q, N, N)
    wmat = reshape(w, N, :)
    return ∂wQw(wmat, Qmat)
end

function ∂wQw(w::AbstractMatrix, Q::AbstractMatrix)
    @assert size(w, 1) == size(Q, 1) == size(Q, 2)
    @assert issymmetric(Q)
    Qw = Q * w
    return 2Qw * (w' * Qw)
end

"""
    sylvester_projection(A, X)

Projects `X` onto the solution space of `A'X + X'A = 0`.
"""
function sylvester_projection(A::AbstractMatrix, X::AbstractMatrix)
    @assert size(A) == size(X)
    AA = A'*A
    AX = A'*X
    L = sylvester(AA, AA, -(AX + AX'))
    return X - A * L
end
