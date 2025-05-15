module MyAD

using Statistics
using Random
using LinearAlgebra

# Core AD types and operations
include("core.jl")

# Make sure CNNVariable is available
include("MyAD/models.jl")

# Neural network model definition
include("MyCNN/Models.jl")

# Neural network components
include("MyCNN/Layers.jl")

# Training utilities
include("MyCNN/Training.jl")

# Data loading and preprocessing
include("MyCNN/Data.jl")

export train_model, predict, load_imdb_data

end # module 