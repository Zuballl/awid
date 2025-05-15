using Test
using MyProject.MyAD

# Dummy layer struct for test
mutable struct DummyDense
    weights::Array{Float32,2}
    bias::Array{Float32,1}
    weights_grad::Array{Float32,2}
    bias_grad::Array{Float32,1}
end

function DummyDense(in_features, out_features)
    weights = ones(Float32, out_features, in_features)
    bias = zeros(Float32, out_features)
    weights_grad = zeros(Float32, out_features, in_features)
    bias_grad = zeros(Float32, out_features)
    DummyDense(weights, bias, weights_grad, bias_grad)
end

@testset "MyAD Dense forward/backward" begin
    layer = DummyDense(3, 2)
    x = rand(Float32, 4, 3) # batch_size=4, in_features=3
    xvar = CNNVariable(x)
    out = dense(xvar, layer.weights, layer.bias, layer)
    y = out.output
    @test size(y) == (2, 4)
    # Backward: gradient = ones
    grad_out = ones(Float32, 2, 4)
    backward(out, grad_out)
    @test all(layer.weights_grad .!= 0)
    @test all(layer.bias_grad .!= 0)
end

mutable struct DummyConv
    filters::Array{Float32,4}
    bias::Array{Float32,1}
    filters_grad::Array{Float32,4}
    bias_grad::Array{Float32,1}
    stride::Tuple{Int,Int}
    padding::Tuple{Int,Int}
end

function DummyConv()
    filters = ones(Float32, 2, 2, 1, 1)
    bias = zeros(Float32, 1)
    filters_grad = zeros(Float32, 2, 2, 1, 1)
    bias_grad = zeros(Float32, 1)
    stride = (1,1)
    padding = (0,0)
    DummyConv(filters, bias, filters_grad, bias_grad, stride, padding)
end

@testset "MyAD Conv2D forward/backward" begin
    layer = DummyConv()
    x = rand(Float32, 1, 1, 4, 4) # batch_size=1, channels=1, 4x4
    xvar = CNNVariable(x)
    out = conv2d(xvar, layer.filters, layer.bias, layer.stride, layer.padding, layer)
    y = out.output
    @test size(y) == (1, 1, 3, 3)
    grad_out = ones(Float32, 1, 1, 3, 3)
    backward(out, grad_out)
    @test all(layer.filters_grad .!= 0)
    @test all(layer.bias_grad .!= 0)
end 