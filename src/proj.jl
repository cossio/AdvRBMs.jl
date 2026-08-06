"""
    kernelproj(w, q)

Projects `w` to the kernel of `q`. That is, the result satisfies
q' * kernelproj(w, q) ≈ 0, up to numerical error.
"""
function kernelproj(w::AbstractArray, q::AbstractArray; qinv::AbstractArray = pseudo_inv_of_q(q))
    K = ndims(q) - 1
    @assert size(w)[1:K] == size(q)[1:K]
    N = prod(size(w, d) for d in 1:K)
    w_proj = kernelproj(reshape(w, N, :), reshape(q, N, :); qinv = reshape(qinv, :, N))
    return reshape(w_proj, size(w))
end

#= The following parenthesization avoids intermediate large matrices. =#
#kernelproj(w::AbstractMatrix, q::AbstractMatrix) = w - q * ((q' * q) \ (q' * w))
#kernelproj(w::AbstractMatrix, q::AbstractMatrix) = w - q' \ (q'w)
#kernelproj(w::AbstractMatrix, q::AbstractMatrix) = w - q * (q \ w) # this is faster usually, but doesn't work with CUDA yet: https://github.com/JuliaGPU/CUDA.jl/issues/104
kernelproj(w::AbstractMatrix, q::AbstractMatrix; qinv::AbstractMatrix = pinv(q)) = w - q * (qinv * w)  # CUDA-friendly

function pseudo_inv_of_q(q::AbstractArray)
    q_flat = reshape(q, :, last(size(q)))
    q_inv = pinv(q_flat)
    return reshape(q_inv, reverse(size(q)))
end

"""
    ∂qw(w, q)

Derivative of `||q' * w||^2 / 2` with respect to `w`.
"""
function ∂qw(w::AbstractArray, q::AbstractArray)
    K = ndims(q) - 1
    @assert size(w)[1:K] == size(q)[1:K]
    N = prod(size(w, d) for d in 1:K)
    ∂ = ∂qw(reshape(w, N, :), reshape(q, N, :))
    return reshape(∂, size(w))
end

∂qw(w::AbstractMatrix, q::AbstractMatrix) = q * (q' * w)

"""
    ∂wQw(w, Q)

Derivative of `∑_k ||w' * Q[:,:,k] * w||^2 / 2` with respect to `w`.
"""
function ∂wQw(w::AbstractArray, Q::AbstractArray)
    @assert isodd(ndims(Q))
    𝒱 = (ndims(Q) - 1) ÷ 2
    @assert size(w)[1:𝒱] == size(Q)[1:𝒱] == size(Q)[(𝒱 + 1):(2𝒱)]
    N = prod(size(w, d) for d in 1:𝒱)
    ∂ = _∂wQw(reshape(w, N, :), reshape(Q, N, N, :))
    return reshape(∂, size(w))
end

function _∂wQw(w::AbstractMatrix, Q::AbstractArray{<:Any, 3})
    @assert size(w, 1) == size(Q, 1) == size(Q, 2)
    return sum(_∂wQw(w, Q[:, :, k]) for k in 1:size(Q, 3))
end

function _∂wQw(w::AbstractMatrix, Q::AbstractMatrix)
    @assert size(w, 1) == size(Q, 1) == size(Q, 2)
    @assert Q ≈ Q'
    Qw = Q * w
    return 2Qw * (w' * Qw) # ∂wQw = 2 * Q * w * w' * Q * w
end
