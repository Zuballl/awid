module MyProject

# First load the core modules
include("MyAD/MyAD.jl")
include("MyCNN/MyCNN.jl")

# Export all modules and their contents
export MyAD
export MyCNN

# Re-export commonly used types and functions
export CNNVariable, CNN
export forward, backward
export train_step!
export mse_loss, cross_entropy_loss

end # module