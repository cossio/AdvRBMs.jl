"""
    empty_intersections(X)

For a given list of collections `X`, returns true if the intersections between pairs of
collections are empty. Otherwise returns false.
"""
function empty_intersections(X)
    for i in eachindex(X), j in eachindex(X)
        i < j || continue
        if !isempty(intersect(X[i], X[j]))
            return false
        end
    end
    return true
end
