module MyAD

export CNNVariable, forward, backward, grad
export conv2d, maxpool2d, relu, flatten

include("tensor_ops.jl")
include("models.jl")

end # module 