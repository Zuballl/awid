# Add vectorized versions of activation functions
function relu(x::AbstractVector)
    return relu.(x)
end

function sigmoid(x::AbstractVector)
    return sigmoid.(x)
end

function tanh(x::AbstractVector)
    return tanh.(x)
end

# Add methods for single values
function relu(x::Float32)
    return max(0f0, x)
end

function sigmoid(x::Float32)
    return 1f0 / (1f0 + exp(-x))
end

function tanh(x::Float32)
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end 