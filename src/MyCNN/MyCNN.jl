module MyCNN

using ..MyAD: CNNVariable, relu, sigmoid, tanh, tensor_ops, engine, operators
import ..MyAD: backward

export Conv2D, MaxPool2D, Dense, NeuralNetwork, CNN
export forward, backward
export mse_loss, cross_entropy_loss, train_step!
export Embedding

include("Layers.jl")
include("Losses.jl")
include("Models.jl")
include("Utils.jl")

end # module 