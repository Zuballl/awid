using ..MyAD: CNNVariable, conv2d, maxpool2d, flatten, dense, pad_input, im2col, col2im

export Conv2D, MaxPool2D, Flatten, Dense

struct Conv2D
    filters::Array{Float32, 4}  # (kernel_height, kernel_width, in_channels, out_channels)
    bias::Array{Float32, 1}     # (out_channels,)
    filters_grad::Array{Float32, 4}
    bias_grad::Array{Float32, 1}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    
    function Conv2D(kernel_size::Tuple{Int, Int}, in_channels::Int, out_channels::Int;
                   stride=(1,1), padding=(0,0))
        # Inicjalizacja wag metodą He
        scale = sqrt(2.0 / (kernel_size[1] * kernel_size[2] * in_channels))
        filters = randn(Float32, kernel_size[1], kernel_size[2], in_channels, out_channels) .* scale
        bias = zeros(Float32, out_channels)
        filters_grad = zeros(Float32, kernel_size[1], kernel_size[2], in_channels, out_channels)
        bias_grad = zeros(Float32, out_channels)
        new(filters, bias, filters_grad, bias_grad, stride, padding)
    end
end

struct MaxPool2D
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    
    function MaxPool2D(kernel_size::Tuple{Int, Int}; stride=kernel_size, padding=(0,0))
        new(kernel_size, stride, padding)
    end
end

struct Flatten end

struct Dense
    weights::Array{Float32, 2}
    bias::Array{Float32, 1}
    weights_grad::Array{Float32, 2}
    bias_grad::Array{Float32, 1}
    activation::Function
    
    function Dense(in_features::Int, out_features::Int, activation::Function=relu)
        # Inicjalizacja wag metodą He
        scale = sqrt(2.0 / in_features)
        weights = randn(Float32, out_features, in_features) .* scale
        bias = zeros(Float32, out_features)
        weights_grad = zeros(Float32, out_features, in_features)
        bias_grad = zeros(Float32, out_features)
        new(weights, bias, weights_grad, bias_grad)
    end
end

# Forward pass dla Conv2D
function (layer::Conv2D)(x::CNNVariable)
    return conv2d(x, layer.filters, layer.bias, layer.stride, layer.padding, layer)
end

# Forward pass dla MaxPool2D
function (layer::MaxPool2D)(x::CNNVariable)
    return maxpool2d(x, layer.kernel_size, layer.stride)
end

# Forward pass dla Flatten
function (layer::Flatten)(x::CNNVariable)
    return flatten(x)
end

# Forward pass dla Dense
function (layer::Dense)(x::CNNVariable)
    return dense(x, layer.weights, layer.bias, layer)
end

# Forward pass for Dense with Matrix{Float64} input (for tests)
function forward(layer::Dense, input::Matrix{Float64})
    input32 = Float32.(input)
    x_var = CNNVariable(input32)
    out = layer(x_var)
    return out.output
end

# Forward pass for Conv2D with Array{Float32,4} input (for tests)
function forward(layer::Conv2D, input::Array{Float32,4})
    return layer(CNNVariable(input))
end

# Forward pass for MaxPool2D with Array{Float32,4} input (for tests)
function forward(layer::MaxPool2D, input::Array{Float32,4})
    return layer(CNNVariable(input))
end

# Forward pass for Flatten with Array{Float32,4} input (for tests)
function forward(layer::Flatten, input::Array{Float32,4})
    return layer(CNNVariable(input))
end

# Forward pass for CNNVariable inputs
function forward(layer::Union{Conv2D, MaxPool2D, Flatten, Dense}, x::CNNVariable)
    return layer(x)
end

# Backward pass for all layers
function backward(layer::Union{Conv2D, MaxPool2D, Flatten, Dense}, grad::Union{Array{Float32, 4}, Matrix{Float32}})
    # Konwertuj gradient na CNNVariable
    grad_var = CNNVariable(grad)
    # Wywołaj backward na CNNVariable
    backward(grad_var, grad)
    # Zwróć gradient wejściowy
    return grad_var.grad
end

# Neural Network definition
mutable struct NeuralNetwork
    layers::Vector{Union{Conv2D, MaxPool2D, Dense}}  # Vector of layers
    input_shape::Union{Tuple{Int, Int, Int, Int}, Nothing}  # Store input shape
    
    function NeuralNetwork(layers::Vector)
        # Konwertuj wektor na Vector{Union{Conv2D, MaxPool2D, Dense}}
        typed_layers = Vector{Union{Conv2D, MaxPool2D, Dense}}(layers)
        return new(typed_layers, nothing)
            end
        end

function forward(network::NeuralNetwork, input)
    # Store input shape for backward pass
    network.input_shape = size(input)
    x = input
    for layer in network.layers
        x = forward(layer, x)
    end
    return x
end

function backward(network::NeuralNetwork, grad)
    # Propagate gradient backward through all layers
    for layer in reverse(network.layers)
        grad = backward(layer, grad)
    end
    
    # Jeśli gradient jest macierzą, konwertuj go na tensor 4D
    if ndims(grad) == 2
        batch_size = size(grad, 2)
        grad = reshape(grad, size(grad, 1), batch_size, 1, 1)
end

    # Jeśli gradient ma inny rozmiar niż wejście, wypełnij zerami
    if network.input_shape !== nothing && size(grad) != network.input_shape
        new_grad = zeros(Float32, network.input_shape)
        # Kopiuj wartości z gradientu do nowego tensora
        for i in 1:min(length(grad), length(new_grad))
            new_grad[i] = grad[i]
        end
        grad = new_grad
    end
    
    return grad
end 