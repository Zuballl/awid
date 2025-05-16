module MyAD

export CNNVariable, backward, forward
export Dual, relu, sigmoid, my_tanh, gradient
export sin, cos, tanh

include("models.jl")
include("operations.jl")

end # module 