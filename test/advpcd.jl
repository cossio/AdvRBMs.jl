import Random
import Flux
using Test: @testset, @test, @inferred
using Random: bitrand
using LinearAlgebra: norm, Diagonal, I, dot
using Statistics: mean, cor
using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: RBM, Binary, Gaussian, BinaryRBM, inputs_h_from_v, wmean
using RestrictedBoltzmannMachines: mean_h_from_v, var_h_from_v, batchmean, batchvar
using RestrictedBoltzmannMachines: free_energy, extensive_sample, initialize!, mean_from_inputs
using AdvRBMs: advpcd!

Random.seed!(2)

function train_nepochs(;
    nsamples::Int, # number observations in the data
    nupdates::Int, # desired number of parameter updates
    batchsize::Int # size of each mini-batch
)
    return ceil(Int, nupdates * batchsize / nsamples)
end

@testset "advpcd" begin
    rbm = BinaryRBM(randn(5,2), randn(3), randn(5,2,3))
    q = randn(5,2,1)
    data = bitrand(5,2,128)
    advpcd!(rbm, data; qs=[q], steps=1, epochs=1, batchsize=32)
    @test norm(inputs_h_from_v(rbm, q)) < 1e-10

    rbm = BinaryRBM(randn(5,2), randn(3), randn(5,2,3))
    q = randn(5,2,1)
    data = bitrand(5,2,128)
    ℋ = CartesianIndices((2:3,))
    advpcd!(rbm, data; qs = [q], ℋs = [ℋ], steps=1, epochs=1, batchsize=32)
    @info @test norm(inputs_h_from_v(rbm, q)[ℋ]) < 1e-10
    @info @test norm(inputs_h_from_v(rbm, q)[1,:]) > 1e-5
end

@testset "advpcd -- teacher/student, Binary, with weights, exact, no constraint" begin
    N = 5
    batchsize = 2^N
    nupdates = 10000
    teacher = RBM(Binary((N,)), Binary((1,)), zeros(N,1))
    teacher.w[:,1] .= range(-2, 2, length=N)
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Binary((N,)), Binary((1,)), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(range(-10, 10, length=N))
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=2)
    advpcd!(student, data; wts, epochs, batchsize, mode=:exact, optim=Flux.AdaBelief())
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.9999
    @test free_energy(teacher, data) .- mean(free_energy(teacher, data)) ≈ free_energy(student, data) .- mean(free_energy(student, data)) rtol=1e-6

    wts_student = softmax(-free_energy(student, data))

    v_student = batchmean(student.visible, data; wts = wts_student)
    v_teacher = batchmean(student.visible, data; wts)
    @info @test norm(v_student - v_teacher) < 1e-10

    h_student = batchmean(student.hidden, mean_h_from_v(student, data); wts=wts_student)
    h_teacher = batchmean(student.hidden, mean_h_from_v(student, data); wts)
    @info @test norm(h_student - h_teacher) < 1e-10

    vh_student = data * Diagonal(wts_student) * mean_h_from_v(student, data)' / sum(wts_student)
    vh_teacher = data * Diagonal(wts) * mean_h_from_v(student, data)' / sum(wts)
    @info @test norm(vh_student - vh_teacher) < 1e-10
end

@testset "advpcd -- teacher/student, Binary, with weights, exact, with constraint" begin
    Random.seed!(1)
    N = 7
    M = 20
    batchsize = 2^N
    nupdates = 50000
    teacher = RBM(Binary((N,)), Binary((M,)), randn(N,M))
    teacher.w .= [sin(i^2 * exp(j)) for i = 1:N, j = 1:M]
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    q = [1; zeros(N - 1);;]
    @test norm(q) ≈ 1
    Prj = I - vec(q) * vec(q)'
    student = RBM(Binary((N,)), Binary((1,)), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(range(-10, 10, length=N)) / 100
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=2)
    advpcd!(student, data; wts, qs=[q], epochs, batchsize, shuffle=false, center=true, mode=:exact, optim=Flux.AdaBelief())

    @info @test abs(dot(q, student.w)) < 1e-10 * norm(student.w) * norm(q)
    @info @test student.w ≈ Prj * student.w
    @info @test norm(student.w) > 1e-3 # check w is non-trivial

    wts_student = softmax(-free_energy(student, data))

    v_student = batchmean(student.visible, data; wts = wts_student)
    v_teacher = batchmean(student.visible, data; wts)
    @info @test v_student ≈ v_teacher rtol=1e-5

    h_student = batchmean(student.hidden, mean_h_from_v(student, data); wts=wts_student)
    h_teacher = batchmean(student.hidden, mean_h_from_v(student, data); wts)
    @info @test h_student ≈ h_teacher rtol=1e-5

    vh_student = data * Diagonal(wts_student) * mean_h_from_v(student, data)' / sum(wts_student)
    vh_teacher = data * Diagonal(wts) * mean_h_from_v(student, data)' / sum(wts)
    @info @test Prj * vh_student ≈ Prj * vh_teacher rtol=1e-5
    @info @test norm(q' * (vh_student - vh_teacher)) ≈ norm(vh_student - vh_teacher) rtol=1e-5
    @test norm(Prj * (vh_student - vh_teacher)) < 0.01norm(q' * (vh_student - vh_teacher))
end

@testset "advpcd -- teacher/student, Binary, with weights, exact, constraint and one free" begin
    Random.seed!(67)
    N = 7
    M = 5
    batchsize = 2^N
    nupdates = 50000
    teacher = RBM(Binary((N,)), Binary((M,)), zeros(N,M))
    teacher.w .= [sin(i^2 * exp(j)) for i = 1:N, j = 1:M]
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    q = [1; zeros(N - 1);;]
    @test norm(q) ≈ 1
    Prj = I - vec(q) * vec(q)'
    student = RBM(Binary((N,)), Binary((2,)), zeros(N,2))
    initialize!(student, data; wts)
    student.w .= [cos(i^2 * exp(i + j)) for i = 1:N, j = 1:2] / 10
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=2)
    ℋ = CartesianIndices((2:2,))
    advpcd!(student, data; wts, qs=[q], ℋs=[ℋ], epochs, batchsize, center=true, shuffle=false, mode=:exact, optim=Flux.AdaBelief())

    @info @test abs(dot(q, student.w[:,ℋ])) < 1e-10 * norm(student.w[:,ℋ]) * norm(q)
    @info @test student.w[:,ℋ] ≈ Prj * student.w[:,ℋ]
    @info @test norm(student.w[:,ℋ]) > 1e-3 # check w is non-trivial

    wts_student = softmax(-free_energy(student, data))

    v_student = batchmean(student.visible, data; wts = wts_student)
    v_teacher = batchmean(student.visible, data; wts)
    @info @test v_student ≈ v_teacher rtol=1e-4

    h_student = batchmean(student.hidden, mean_h_from_v(student, data); wts=wts_student)
    h_teacher = batchmean(student.hidden, mean_h_from_v(student, data); wts)
    @info @test h_student ≈ h_teacher rtol=1e-4

    vh_student = data * Diagonal(wts_student) * mean_h_from_v(student, data)' / sum(wts_student)
    vh_teacher = data * Diagonal(wts) * mean_h_from_v(student, data)' / sum(wts)
    @info @test vh_student[:,1] ≈ vh_teacher[:,1] rtol=1e-5
    @info @test Prj * vh_student[:,ℋ] ≈ Prj * vh_teacher[:,ℋ] rtol=1e-5
    @info @test norm(q' * (vh_student - vh_teacher)[:,ℋ]) ≈ norm((vh_student - vh_teacher)[:,ℋ]) rtol=1e-3
    @test norm(Prj * (vh_student - vh_teacher)[:,ℋ]) < 0.1norm(q' * (vh_student - vh_teacher)[:,ℋ])
end

# test hidden rescaling
@testset "pcd (no constraint) -- teacher/student, Gaussian, with weights, exact" begin
    N = 5
    batchsize = 2^N
    nupdates = 10000
    teacher = RBM(Binary((N,)), Gaussian((1,)), zeros(N,1))
    teacher.w[:,1] .= range(-1, 1, length=N)
    data = extensive_sample(teacher.visible)
    wts = softmax(-free_energy(teacher, data))
    @test sum(wts) ≈ 1
    nsamples = size(data)[end]
    epochs = train_nepochs(; nsamples, batchsize, nupdates)
    student = RBM(Binary((N,)), Gaussian(1), zeros(N,1))
    initialize!(student, data; wts)
    student.w .= cos.(1:N)
    @test mean_from_inputs(student.visible) ≈ wmean(data; wts, dims=2)
    advpcd!(student, data; wts, epochs, batchsize, ϵh=1e-2, shuffle=false, mode=:exact, optim=Flux.AdaBelief())
    @info @test cor(free_energy(teacher, data), free_energy(student, data)) > 0.99
    wts_student = softmax(-free_energy(student, data))
    ν_int = batchmean(student.hidden, var_h_from_v(student, data); wts = wts_student)
    ν_ext = batchvar(student.hidden, mean_h_from_v(student, data); wts = wts_student)
    @test only(ν_int + ν_ext) ≈ 1 - 1e-2 # not exactly 1 because of ϵh
end
