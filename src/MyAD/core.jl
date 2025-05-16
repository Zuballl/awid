# Abstract types for the computation graph
abstract type GraphNode end
abstract type Variable <: GraphNode end
abstract type Operator <: GraphNode end

# Base Variable type
struct Constant <: Variable
    output::Any
    gradient::Any
    Constant(x) = new(x, nothing)
end

# CNN-specific types
mutable struct CNNVariable <: Variable
    output::Array{Float32, N} where N
    grad::Union{Array{Float32, N} where N, Nothing}
    grad_fn::Union{Function, Nothing}
    grad_inputs::Union{Vector{CNNVariable}, Nothing}
end

# Constructors for CNNVariable
CNNVariable(output::Array{Float32, N} where N) = CNNVariable(output, nothing, nothing, nothing)

# Allow constructing CNNVariable from a 2D matrix (e.g., after flattening)
function CNNVariable(x::AbstractMatrix)
    CNNVariable(reshape(Float32.(x), size(x,1), size(x,2), 1, 1))
end

# Add size method for CNNVariable
Base.size(x::CNNVariable) = size(x.output)
Base.size(x::CNNVariable, dim) = size(x.output, dim)

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