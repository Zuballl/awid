module Utils

import ..Models
import ..Training
import ..Data
using Plots
using Statistics
using JSON

export save_model, load_model, plot_training_history, plot_confusion_matrix
export evaluate_model, visualize_predictions

# Funkcja do zapisywania modelu
function save_model(model::Models.CNN, path::String)
    model_state = Dict(
        "layers" => model.layers,
        "parameters" => Models.parameters(model)
    )
    
    open(path, "w") do io
        JSON.print(io, model_state)
    end
end

# Funkcja do wczytywania modelu
function load_model(path::String)
    model_state = JSON.parsefile(path)
    
    # Stwórz nowy model
    model = Models.CNN()
    
    # Przywróć parametry
    for (layer, param) in zip(model.layers, model_state["parameters"])
        if isa(layer, Layers.Conv2D)
            layer.filters .= param["filters"]
            layer.bias .= param["bias"]
        elseif isa(layer, Layers.Dense)
            layer.weights .= param["weights"]
            layer.bias .= param["bias"]
        end
    end
    
    return model
end

# Funkcja do rysowania historii treningu
function plot_training_history(train_losses::Vector{Float32}, train_accs::Vector{Float32},
                             val_losses::Vector{Float32}, val_accs::Vector{Float32})
    epochs = 1:length(train_losses)
    
    # Wykres straty
    p1 = plot(epochs, train_losses, label="Train Loss", 
             xlabel="Epoch", ylabel="Loss", title="Training History")
    plot!(epochs, val_losses, label="Validation Loss")
    
    # Wykres dokładności
    p2 = plot(epochs, train_accs, label="Train Accuracy",
             xlabel="Epoch", ylabel="Accuracy")
    plot!(epochs, val_accs, label="Validation Accuracy")
    
    # Połącz wykresy
    plot(p1, p2, layout=(2,1), size=(800,600))
end

# Funkcja do rysowania macierzy pomyłek
function plot_confusion_matrix(y_true::Vector{Int}, y_pred::Vector{Int}, n_classes::Int)
    # Oblicz macierz pomyłek
    cm = zeros(Int, n_classes, n_classes)
    for i in 1:length(y_true)
        cm[y_true[i], y_pred[i]] += 1
    end
    
    # Normalizuj dla lepszej wizualizacji
    cm_norm = cm ./ sum(cm, dims=2)
    
    # Stwórz heatmap
    heatmap(cm_norm, 
            xlabel="Predicted", ylabel="True",
            title="Confusion Matrix",
            color=:viridis,
            aspect_ratio=:equal)
end

# Funkcja do oceny modelu
function evaluate_model(model::Models.CNN, X::Array{Float32, 4}, y::Vector{Int})
    # Predykcje
    y_pred = argmax(Models.forward(model, X), dims=1)
    
    # Dokładność
    accuracy = mean(y_pred .== y)
    
    # Macierz pomyłek
    confusion_matrix = zeros(Int, 10, 10)  # Zakładamy 10 klas
    for i in 1:length(y)
        confusion_matrix[y[i], y_pred[i]] += 1
    end
    
    # Precyzja i recall dla każdej klasy
    precision = zeros(Float32, 10)
    recall = zeros(Float32, 10)
    
    for i in 1:10
        true_positives = confusion_matrix[i, i]
        false_positives = sum(confusion_matrix[:, i]) - true_positives
        false_negatives = sum(confusion_matrix[i, :]) - true_positives
        
        precision[i] = true_positives / (true_positives + false_positives)
        recall[i] = true_positives / (true_positives + false_negatives)
    end
    
    # F1-score
    f1_score = 2 .* (precision .* recall) ./ (precision .+ recall)
    
    return Dict(
        "accuracy" => accuracy,
        "confusion_matrix" => confusion_matrix,
        "precision" => precision,
        "recall" => recall,
        "f1_score" => f1_score
    )
end

# Funkcja do wizualizacji wyników
function visualize_predictions(model::Models.CNN, X::Array{Float32, 4}, y::Vector{Int}; n_samples=5)
    # Losowe próbki
    indices = randperm(length(y))[1:n_samples]
    X_samples = X[indices, :, :, :]
    y_true = y[indices]
    
    # Predykcje
    y_pred = argmax(Models.forward(model, X_samples), dims=1)
    
    # Wyświetl wyniki
    for i in 1:n_samples
        println("Sample $i:")
        println("  True label: $(y_true[i])")
        println("  Predicted: $(y_pred[i])")
        println("  Correct: $(y_true[i] == y_pred[i])")
    end
end

end # module 