module MyAD

# Export types
export GraphNode, Variable, Operator
export CNNVariable, Constant

# Export operators
export ScalarOp, BroadcastedOp
export forward, backward
export update!

# Export activation functions
export relu, sigmoid, tanh
export gradient

# Export tensor operations
export pad_input, im2col, col2im

# Load core functionality
include("core.jl")

# Load tensor operations
include("tensor_ops.jl")

# Load the rest of the functionality
include("engine.jl")
include("operators.jl")
include("operations.jl")
include("models.jl")

end # module 