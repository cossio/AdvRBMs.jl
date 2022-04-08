import MKL
using SafeTestsets: @safetestset

@time @safetestset "proj" begin include("proj.jl") end
@time @safetestset "calc_qQ" begin include("calc_qQ.jl") end
@time @safetestset "advpcd" begin include("advpcd.jl") end
