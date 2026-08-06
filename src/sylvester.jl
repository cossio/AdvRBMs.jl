# This file is not included in the AdvRBMs module. It preserves currently
# unused projection helpers; to use them, include this file after proj.jl
# (it needs `kernelproj` and `using LinearAlgebra: sylvester`).

"""
    sylvester_projection(A, X)

Returns the projection of `X` onto the solution space of `A'X + X'A = 0`.
"""
function sylvester_projection(A::AbstractMatrix, X::AbstractMatrix)
    @assert size(A) == size(X)
    AA = A' * A
    AX = A' * X
    L = sylvester(AA, AA, -(AX + AX'))
    return X - A * L
end

"""
    project∂!(∂w, q, Q)

Projects gradients `∂w` using the given constraints.
"""
project∂!(::AbstractArray, ::Nothing = nothing, ::Nothing = nothing) = nothing

function project∂!(∂w::AbstractArray, q::AbstractArray, ::Nothing = nothing)
    @assert size(∂w)[1:ndims(q)] == size(q)
    return ∂w .= kernelproj(∂w, q)
end

function project∂!(∂w::AbstractArray, ::Nothing, Q::AbstractArray)
    @assert size(Q) == (size(∂w)[1:(ndims(Q) ÷ 2)]..., size(∂w)[1:(ndims(Q) ÷ 2)]...)
    Qw = Q * ∂w
    ∂w .= sylvester_projection(Qw, ∂w)
    return ∂w
end

function project∂!(∂w::AbstractArray, q::AbstractArray, Q::AbstractArray)
    @assert size(Q) == (size(q)..., size(q)...)
    Qw = Q * ∂w
    Qw .= kernelproj(Qw, q)
    ∂w .= sylvester_projection(Qw, ∂w)
    return ∂w
end
