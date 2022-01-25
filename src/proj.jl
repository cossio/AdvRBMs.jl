"""
    project!(w, q)

Projects weigths `w` using the given constraints.
"""
project!(w::AbstractArray, ::Nothing = nothing) = w
function project!(w::AbstractArray, q::AbstractArray)
    @assert size(w)[1:ndims(q)] == size(q)
    w .= project_plane(w, q)
    return w
end

"""
    ∂project!(∂w, q, Q)

Projects gradients `∂w` using the given constraints.
"""
∂project!(::AbstractArray, ::Nothing = nothing, ::Nothing = nothing) = nothing
function ∂project!(∂w::AbstractArray, q::AbstractArray, ::Nothing = nothing)
    @assert size(∂w)[1:ndims(q)] == size(q)
    ∂w .= project_plane(∂w, q)
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
    Qw .= project_plane(Qw, q)
    ∂w .= sylvester_projection(Qw, ∂w)
    return ∂w
end
