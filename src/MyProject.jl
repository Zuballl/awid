module MyProject

export AD, CNN, Conv2D, MaxPool2D, Flatten, Dense, forward, parameters, train, evaluate
export load_data, create_data_loader, split_data
export save_model, load_model, plot_training_history, plot_confusion_matrix
export evaluate_model, visualize_predictions

# ============ AD =============
module AD
    export GraphNode, Operator, Constant, Variable, CNNVariable
    export dense, conv2d, backward, flatten
    export ScalarOp, BroadcastedOp, forward!, backward!, gradient, jacobian
    export forward, back, update!, topological_sort
    include("MyAD/models.jl")
    include("MyAD/operators.jl")
    include("MyAD/engine.jl")
end

# ============ CNN =============
include("MyCNN/Layers.jl")
include("MyCNN/Models.jl")
include("MyCNN/Training.jl")
include("MyCNN/Utils.jl")
include("MyCNN/Data.jl")

end