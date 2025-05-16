module MyCNN

using ..MyAD: CNNVariable, relu, sigmoid, tanh
import ..MyAD: backward

export Conv2D, MaxPool2D, Dense, NeuralNetwork, forward, backward
export mse_loss, cross_entropy_loss, train_step!

include("Layers.jl")
include("Losses.jl")
include("Training.jl")

end # module 