export Dense, ReLU, Softmax, forward, backward

using ..AD: MLPVariable

mutable struct Dense
    weights::Matrix{Float64}
    bias::Vector{Float64}
    input::Union{MLPVariable,Nothing}
    output::Union{MLPVariable,Nothing}
end

function Dense(input_size::Int, output_size::Int)
    scale = sqrt(2.0 / input_size)
    weights = randn(output_size, input_size) * scale
    bias = zeros(output_size)
    return Dense(weights, bias, nothing, nothing)
end

function forward(layer::Dense, input::MLPVariable)
    layer.input = input
    output = layer.weights * input.output .+ layer.bias
    layer.output = MLPVariable(output)
    return layer.output
end

function backward(layer::Dense, grad::Matrix{Float64})
    if layer.input === nothing
        error("No input stored in layer")
    end
    
    grad_weights = grad * layer.input.output'
    
    grad_bias = sum(grad, dims=2)
    
    grad_input = layer.weights' * grad
    
    layer.weights .-= 0.01 * grad_weights
    layer.bias .-= 0.01 * grad_bias
    
    return grad_input
end

mutable struct ReLU
    input::Union{MLPVariable,Nothing}
    output::Union{MLPVariable,Nothing}
end

ReLU() = ReLU(nothing, nothing)

function forward(layer::ReLU, input::MLPVariable)
    layer.input = input
    output = max.(0, input.output)
    layer.output = MLPVariable(output)
    return layer.output
end

function backward(layer::ReLU, grad::Matrix{Float64})
    if layer.input === nothing
        error("No input stored in layer")
    end
    
    grad_input = grad .* (layer.input.output .> 0)
    return grad_input
end

mutable struct Softmax
    input::Union{MLPVariable,Nothing}
    output::Union{MLPVariable,Nothing}
end

Softmax() = Softmax(nothing, nothing)

function forward(layer::Softmax, input::MLPVariable)
    layer.input = input
    exp_x = exp.(input.output .- maximum(input.output, dims=1))
    output = exp_x ./ sum(exp_x, dims=1)
    layer.output = MLPVariable(output)
    return layer.output
end

function backward(layer::Softmax, grad::Matrix{Float64})
    if layer.input === nothing || layer.output === nothing
        error("No input or output stored in layer")
    end
    
    grad_input = layer.output.output .* (grad .- sum(grad .* layer.output.output, dims=1))
    return grad_input
end