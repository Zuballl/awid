module MyProject

export AD, MyCNN

# ============ AD =============
module AD
  export GraphNode, Operator, Constant, Variable, CNNVariable
  export ScalarOp, BroadcastedOp, forward!, backward!, gradient, jacobian
  export forward, back, update!, topological_sort

  include("MyAD/models.jl")
  include("MyAD/operators.jl")
  include("MyAD/engine.jl")
end

# ============ MyCNN =============
module MyCNN
  import ..AD
  
  # Eksportujemy wszystkie funkcje i typy
  export CNN, Conv2D, MaxPool2D, Flatten, Dense
  export forward, parameters, train, evaluate
  export load_data, create_data_loader, split_data
  export save_model, load_model, plot_training_history, plot_confusion_matrix
  export evaluate_model, visualize_predictions

  # Najpierw definiujemy moduły
  module CNNLayers
    import ..AD
    include("MyCNN/Layers.jl")
  end

  module CNNData
    using JSON
    using Random
    using Statistics
    using LinearAlgebra
    include("MyCNN/Data.jl")
  end

  # Teraz definiujemy Models, który zależy od Layers
  module CNNModels
    import ..CNNLayers
    import ..AD
    include("MyCNN/Models.jl")
  end

  # Teraz definiujemy Training, który zależy od Models
  module CNNTraining
    import ..CNNModels
    import ..AD
    import ..CNNLayers
    include("MyCNN/Training.jl")
  end

  # Na końcu definiujemy Utils, który zależy od wszystkich pozostałych
  module CNNUtils
    import ..CNNModels
    import ..CNNTraining
    import ..CNNData
    using Plots
    using Statistics
    using JSON
    include("MyCNN/Utils.jl")
  end

  # Eksportujemy zawartość modułów
  using .CNNLayers: Conv2D, MaxPool2D, Flatten, Dense
  using .CNNModels: CNN, forward, parameters
  using .CNNTraining: train, evaluate, train_epoch, cross_entropy_loss
  using .CNNData: load_data, create_data_loader, split_data
  using .CNNUtils: save_model, load_model, plot_training_history, plot_confusion_matrix, evaluate_model, visualize_predictions
end

end # module MyProject