module MyAD

export CNNVariable, backward, relu, sigmoid, tanh, conv2d, maxpool2d, flatten, dense
export pad_input, im2col, col2im

include("core.jl")
include("tensor_ops.jl")  # Najpierw funkcje pomocnicze
include("models.jl")      # Potem modele, które ich używają

end # module 