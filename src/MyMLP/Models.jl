export Chain, relu, softmax, predict, crossentropy

using ..AD: MLPVariable

mutable struct Chain
    layers::Vector{Any}
end

Chain(layers...) = Chain([layers...])

function (model::Chain)(x::MLPVariable)
    current = x
    for layer in model.layers
        current = forward(layer, current)
    end
    return current
end

function backward(model::Chain, grad::Matrix{Float64})
    current_grad = grad
    for layer in reverse(model.layers)
        current_grad = backward(layer, current_grad)
    end
    return current_grad
end

function update!(model::Chain, learning_rate::Float64)
    for layer in model.layers
        if isa(layer, Dense)
            update!(layer, learning_rate)
        end
    end
end

function relu(x::MLPVariable)
    return max.(x.output, 0)
end

function softmax(x::MLPVariable)
    exp_x = exp.(x.output .- maximum(x.output, dims=1))
    return exp_x ./ sum(exp_x, dims=1)
end

function predict(model::Chain, X::Matrix{Float64})
    scores = model(MLPVariable(X))
    return [argmax(scores.output[:, i]) for i in 1:size(X, 2)]
end

function crossentropy(ŷ::MLPVariable, y::Int)
    log_probs = log.(ŷ.output .+ 1e-10)
    return -log_probs[y]
end 