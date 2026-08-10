# Tests GPU compatibility without a physical GPU, using JLArrays (the reference
# GPUArrays.jl backend). With allowscalar(false), any code path falling back to
# scalar indexing errors out, just like CuArray on CI. The setting is session-global,
# which is intentional: all JLArrays tests live in this file, and any future test
# using JLArrays should run under allowscalar(false) too.
import Random
import LinearAlgebra
using Test: @test, @testset, @test_broken
using Random: bitrand
using LinearAlgebra: norm, pinv
using Adapt: adapt
using JLArrays: JLArray, JLArrays
using RestrictedBoltzmannMachines: BinaryRBM, center, standardize, free_energy, inputs_h_from_v
using AdvRBMs: advpcd!, calc_q, calc_Q, kernelproj, pseudo_inv_of_q, ∂qw, ∂wQw

JLArrays.allowscalar(false)

# CUDA.jl provides the RNG-less `Random.rand!(::AnyCuArray)`, but JLArrays does not, and
# the Random stdlib's `rand!(A)` skips the two-arg method that GPUArrays implements
# GPU-friendly, falling into a scalar-indexing loop. Emulate the CUDA.jl overload here.
Random.rand!(A::JLArray) = Random.rand!(Random.default_rng(), A)

# CUDA.jl supports `pinv(::CuMatrix)` through its CUSOLVER svd, but JLArrays has no SVD,
# so `pseudo_inv_of_q` (which advpcd! calls on the constraints) cannot run natively.
# Emulate the CUDA.jl behavior by computing on the host and moving the (small) result
# back to the device.
LinearAlgebra.pinv(A::JLArray{<:Any, 2}; kwargs...) = JLArray(pinv(Array(A); kwargs...))

Random.seed!(3)

const N = (3, 2) # visible site grid
const M = 3 # hidden units
const B = 64 # number of samples

flatw(w::AbstractArray) = reshape(w, :, M)
flatq(q::AbstractArray) = reshape(q, :, size(q)[end])

@testset "kernelproj and pseudo_inv_of_q" begin
    w = randn(N..., M)
    for t in (1, 2)
        q = randn(N..., t)
        jl_w = JLArray(w)
        jl_q = JLArray(q)

        jl_qinv = pseudo_inv_of_q(jl_q)
        @test jl_qinv isa JLArray
        @test Array(jl_qinv) ≈ pseudo_inv_of_q(q)

        jl_wp = kernelproj(jl_w, jl_q)
        @test jl_wp isa JLArray
        @test Array(jl_wp) ≈ kernelproj(w, q)
        # the projection satisfies the constraint, computed on device
        @test norm(flatq(jl_q)' * flatw(jl_wp)) < 1.0e-9 * norm(w)
        # explicit qinv, e.g. adapted from a host-side computation
        @test Array(kernelproj(jl_w, jl_q; qinv = JLArray(pseudo_inv_of_q(q)))) ≈ kernelproj(w, q)
    end

    # eltype is preserved on device
    w32 = randn(Float32, N..., M)
    q32 = randn(Float32, N..., 1)
    jl_wp32 = kernelproj(JLArray(w32), JLArray(q32))
    @test jl_wp32 isa JLArray
    @test eltype(jl_wp32) == Float32
end

@testset "∂qw and ∂wQw" begin
    w = randn(N..., M)
    q = randn(N..., 2)
    Q = randn(N..., N..., 2)
    Q = (Q + permutedims(Q, (3, 4, 1, 2, 5))) / 2 # symmetrize

    jl_∂q = ∂qw(JLArray(w), JLArray(q))
    @test jl_∂q isa JLArray
    @test Array(jl_∂q) ≈ ∂qw(w, q)

    jl_∂Q = ∂wQw(JLArray(w), JLArray(Q))
    @test jl_∂Q isa JLArray
    @test Array(jl_∂Q) ≈ ∂wQw(w, Q)
end

@testset "calc_q and calc_Q" begin
    v = randn(N..., B)
    u = bitrand(B)
    uc = falses(3, B) # categorical (one-hot) labels
    for i in 1:B
        uc[rand(1:3), i] = true
    end
    wts = rand(B)

    jl_v = JLArray(v)
    jl_u = JLArray(u)
    jl_uc = JLArray(collect(uc))
    jl_wts = JLArray(wts)

    # binary labels, default (lazy uniform) weights
    jl_q = calc_q(jl_u, jl_v)
    @test jl_q isa JLArray
    @test Array(jl_q) ≈ calc_q(u, v)

    # explicit device weights
    @test Array(calc_q(jl_u, jl_v; wts = jl_wts)) ≈ calc_q(u, v; wts)
    @test Array(calc_Q(jl_u, jl_v; wts = jl_wts)) ≈ calc_Q(u, v; wts)
    @test Array(calc_q(jl_uc, jl_v; wts = jl_wts)) ≈ calc_q(uc, v; wts)

    # with the default lazy `Trues` weights these paths build `Diagonal(wts)`, which
    # mixes a host matrix into device matmuls and falls back to scalar indexing
    @test_broken calc_Q(jl_u, jl_v) isa JLArray
    @test_broken calc_q(jl_uc, jl_v) isa JLArray
    # categorical calc_Q accumulates into a host `zeros` array
    @test_broken calc_Q(jl_uc, jl_v; wts = jl_wts) isa JLArray
end

@testset "advpcd! on device" begin
    data = float(bitrand(N..., B))
    u = bitrand(B)
    q = calc_q(u, data)
    Q = calc_Q(u, data)
    jl_data = JLArray(data)
    jl_q = JLArray(q)
    jl_Q = JLArray(Q)
    ℋ = CartesianIndices((1:2,))

    # unconstrained runs exercise the default_qs/default_Qs/default_ℋs code paths
    for make_rbm in (identity, center, standardize)
        jl_rbm = adapt(JLArray, make_rbm(BinaryRBM(zeros(N...), zeros(M), randn(N..., M) / 3)))
        advpcd!(jl_rbm, jl_data; iters = 4, batchsize = 16)
        @test jl_rbm.w isa JLArray
        @test all(isfinite, Array(jl_rbm.w))
    end

    # RBM with 1st-order hard constraint on a subgroup and 2nd-order soft constraint
    jl_rbm = adapt(JLArray, BinaryRBM(randn(N...), randn(M), randn(N..., M) / 3))
    advpcd!(
        jl_rbm, jl_data;
        qs = [jl_q], Qs = [jl_Q], ℋs = [ℋ], λQ = 0.1,
        iters = 10, batchsize = 16, steps = 2,
    )
    w = Array(jl_rbm.w)
    @test all(isfinite, w)
    @test norm(flatq(q)' * flatw(w)[:, ℋ]) < 1.0e-10
    @test norm(flatq(q)' * flatw(w)[:, 3]) > 1.0e-5 # unconstrained unit
    # constraint residual can also be computed on device
    @test norm(inputs_h_from_v(jl_rbm, jl_q)[ℋ]) < 1.0e-10
    @test all(isfinite, Array(free_energy(jl_rbm, jl_data)))

    # weighted data
    jl_wts = JLArray(vcat(fill(1.0, B ÷ 2), fill(2.0, B ÷ 2)))
    advpcd!(
        jl_rbm, jl_data;
        wts = jl_wts, qs = [jl_q], ℋs = [ℋ],
        iters = 4, batchsize = 16,
    )
    @test all(isfinite, Array(jl_rbm.w))
    @test norm(flatq(q)' * flatw(Array(jl_rbm.w))[:, ℋ]) < 1.0e-10

    # CenteredRBM
    jl_rbm = adapt(JLArray, center(BinaryRBM(randn(N...), randn(M), randn(N..., M) / 3)))
    advpcd!(
        jl_rbm, jl_data;
        qs = [jl_q], Qs = [jl_Q], ℋs = [ℋ], λQ = 0.1,
        iters = 10, batchsize = 16, steps = 2,
    )
    @test jl_rbm.offset_v isa JLArray
    @test jl_rbm.offset_h isa JLArray
    @test all(isfinite, Array(jl_rbm.w))
    @test norm(flatq(q)' * flatw(Array(jl_rbm.w))[:, ℋ]) < 1.0e-10
end

@testset "advpcd! standardized on device" begin
    # heterogeneous unit statistics, so that scale_v is far from uniform
    p = reshape(range(0.1, 0.9; length = prod(N)), N...)
    data = float(rand(N..., B) .< p)
    u = bitrand(B)
    q = calc_q(u, data)
    Q = calc_Q(u, data)
    jl_data = JLArray(data)
    jl_q = JLArray(q)
    jl_Q = JLArray(Q)
    ℋ = CartesianIndices((1:2,))

    jl_rbm = adapt(JLArray, standardize(BinaryRBM(randn(N...), randn(M), randn(N..., M) / 3)))
    advpcd!(
        jl_rbm, jl_data;
        qs = [jl_q], Qs = [jl_Q], ℋs = [ℋ], λQ = 0.1,
        iters = 10, batchsize = 16, steps = 2,
    )
    @test jl_rbm.scale_v isa JLArray
    @test jl_rbm.scale_h isa JLArray
    w = Array(jl_rbm.w)
    @test all(isfinite, w)
    # the hard constraint acts on the unstandardized weights w ./ scale_v
    q_scaled = Array(jl_q ./ jl_rbm.scale_v)
    @test norm(flatq(q_scaled)' * flatw(w)[:, ℋ]) < 1.0e-10
    @test all(isfinite, Array(free_energy(jl_rbm, jl_data)))
end
