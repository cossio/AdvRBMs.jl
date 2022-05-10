"""
    project_1st_only(w, q)

Projects `w` to the kernel of `q`. That is, the result satisfies
q' * project_1st_only(w, q) ≈ 0, up to numerical error.
"""
function project_1st_only(w::AbstractArray, q::AbstractArray)
    @assert size(w)[1:(ndims(q) - 1)] == front(size(q))
    q_mat = reshape(q, :, size(q)[end])
    w_mat = reshape(w, size(q_mat, 1), :)
    w_proj = project_1st_only(w_mat, q_mat)
    return reshape(w_proj, size(w))
end

#= The following parenthesization avoids intermediate large matrices. =#
project_1st_only(w::AbstractMatrix, q::AbstractMatrix) = w - q * ((q' * q) \ (q' * w))

project(w::AbstractArray, q::AbstractArray, Q::Nothing) = project_1st_only(w, q)

∂project(∂::AbstractArray, q::AbstractArray, Q::Nothing) = project_1st_only(w, q)

project!(w::AbstractArray, q, Q) = w .= project(w, q, Q)

function project(w::AbstractArray, q::AbstractArray, Q::AbstractArray)

end

function parametric_2nd()
end
