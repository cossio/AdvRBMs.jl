import LinearAlgebra
import Statistics
import Random
import Logging
import Profile
import StatsBase
import Flux
import ProgressMeter
import CairoMakie
import Makie
import MLDatasets
import RestrictedBoltzmannMachines as RBMs
import AdvRBMs
using RestrictedBoltzmannMachines: RBM, Binary, BinaryRBM

"""
    imggrid(A)

Given a four dimensional tensor `A` of size `(width, height, ncols, nrows)`
containing `width x height` images in a grid of `nrows x ncols`, this returns
a matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap
to display all images.
"""
function imggrid(A::AbstractArray{<:Any,4})
    width, height, ncols, nrows = size(A)
    return reshape(permutedims(A, (1,3,2,4)), width * ncols, height * nrows)
end

train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
Float = Float32
train_x = Array{Float}(train_x[:, :, (train_y .== 0) .| (train_y .== 1)] .> 0.5)
tests_x = Array{Float}(tests_x[:, :, (tests_y .== 0) .| (tests_y .== 1)] .> 0.5)
train_y = BitVector(train_y[(train_y .== 0) .| (train_y .== 1)])
tests_y = BitVector(tests_y[(tests_y .== 0) .| (tests_y .== 1)])
train_nsamples = length(train_y)
tests_nsamples = length(tests_y)
(train_nsamples, tests_nsamples)

nrows, ncols = 10, 30
fig = Makie.Figure(resolution=(30ncols, 30nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
digits = reshape(train_x[:, :, rand(1:size(train_x,3), nrows * ncols)], 28, 28, ncols, nrows)
Makie.image!(ax, imggrid(digits), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig

rbm = BinaryRBM(Float, (28,28), 256)
RBMs.initialize!(rbm, train_x)
q = AdvRBMs.calc_q(train_y, train_x)
Makie.image(q)

inputs_h_from_v(rbm, q)

Profile.init(n = 10^7, delay = 0.01)

AdvRBMs.advpcd!(rbm, train_x; q, steps=1, epochs=1, batchsize=128)


@profview RBMs.pcd!(rbm, train_x; steps=1, epochs=10, batchsize=128)
