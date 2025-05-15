module CNNModels

import ..CNNLayers
import ..AD

export CNN, forward, parameters, relu, softmax

# Funkcje aktywacji
function relu(x::AD.CNNVariable)
    return AD.relu(x)
end

function softmax(x::Array{Float32, 2})
    exp_x = exp.(x .- maximum(x, dims=1))
    return exp_x ./ sum(exp_x, dims=1)
end

struct CNN
    layers::Vector{Any}
    
    function CNN()
        layers = [
            CNNLayers.Conv2D((3, 3), 1, 32),  # Pierwsza warstwa konwolucyjna
            x -> relu(x),                  # ReLU
            CNNLayers.MaxPool2D((2, 2)),
            CNNLayers.Conv2D((3, 3), 32, 64), # Druga warstwa konwolucyjna
            x -> relu(x),                  # ReLU
            CNNLayers.MaxPool2D((2, 2)),
            CNNLayers.Flatten(),
            CNNLayers.Dense(1600, 128),       # Warstwa gęsta
            x -> relu(x),                  # ReLU
            CNNLayers.Dense(128, 10),         # Warstwa wyjściowa
            x -> AD.CNNVariable(softmax(x.output), nothing, nothing, nothing)
        ]
        new(layers)
    end
end

function forward(model::CNN, x::Array{Float32, 4})
    x_var = AD.CNNVariable(x)
    for layer in model.layers
        x_var = layer(x_var)
    end
    return x_var
end

function parameters(model::CNN)
    params = []
    for layer in model.layers
        if isa(layer, CNNLayers.Conv2D)
            push!(params, layer.filters, layer.bias)
        elseif isa(layer, CNNLayers.Dense)
            push!(params, layer.weights, layer.bias)
        end
    end
    return params
end

end # module CNNModels 