using ..MyAD: CNNVariable
using Statistics: mean
using ..MyCNN: NeuralNetwork, Conv2D, Dense

export train_epoch, train, evaluate, cross_entropy_loss, train_step!

# Funkcja straty
function cross_entropy_loss(pred::Array{Float32, 2}, y::Vector{Int})
    batch_size = size(pred, 2)
    loss = 0.0
    
    for i in 1:batch_size
        loss -= log(pred[y[i], i] + 1e-10)
    end
    
    return loss / batch_size
end

# Gradient funkcji straty
function cross_entropy_gradient(pred::Array{Float32, 2}, y::Vector{Int})
    batch_size = size(pred, 2)
    grad = zeros(Float32, size(pred))
    
    for i in 1:batch_size
        grad[y[i], i] = -1.0 / (pred[y[i], i] + 1e-10)
    end
    
    return grad
end

function zero_grad!(model)
    for layer in model.layers
        if isa(layer, Conv2D)
            fill!(layer.filters, 0)
            fill!(layer.bias, 0)
        elseif isa(layer, Dense)
            fill!(layer.weights, 0)
            fill!(layer.bias, 0)
        end
    end
end

function get_grads(model)
    grads = []
    for layer in model.layers
        if isa(layer, Conv2D)
            push!(grads, layer.filters_grad, layer.bias_grad)
        elseif isa(layer, Dense)
            push!(grads, layer.weights_grad, layer.bias_grad)
        end
    end
    return grads
end

function train_epoch(model::NeuralNetwork, train_loader, learning_rate::Float32, loss_fn)
    total_loss = 0.0
    correct = 0
    total = 0
    
    for (x, y) in train_loader
        # Forward pass przez własny AD
        out = forward(model, x)
        loss = loss_fn(out.output, y)
        # Wyzeruj gradienty
        zero_grad!(model)
        # Backward przez własny AD
        backward(out, 1.0f0)
        # Zbierz gradienty
        grads = get_grads(model)
        # Update parametrów (prosty SGD)
        # (Możesz tu podpiąć Flux.Optimise.update! jeśli chcesz Adam)
        # ...
        # Statystyki
        total_loss += loss
        correct += sum(argmax(out.output, dims=1) .== y)
        total += length(y)
    end
    
    return total_loss / length(train_loader), correct / total
end

function train(model::NeuralNetwork, train_loader, val_loader, optimizer, loss_fn;
              epochs=10, early_stopping_patience=3, learning_rate=0.001)
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = nothing
    
    for epoch in 1:epochs
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, learning_rate, loss_fn)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        
        println("Epoch $epoch:")
        println("  Train Loss: $train_loss, Train Acc: $train_acc")
        println("  Val Loss: $val_loss, Val Acc: $val_acc")
        
        # Early stopping
        if val_acc > best_val_acc
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = deepcopy(parameters(model))
        else
            patience_counter += 1
            if patience_counter >= early_stopping_patience
                println("Early stopping triggered!")
                # Przywróć najlepszy stan modelu
                for (param, best_param) in zip(parameters(model), best_model_state)
                    param .= best_param
                end
                break
            end
        end
    end
end

function evaluate(model::NeuralNetwork, data_loader, loss_fn)
    total_loss = 0.0
    correct = 0
    total = 0
    
    for (x, y) in data_loader
        pred = forward(model, x)
        loss = loss_fn(pred, y)
        
        total_loss += loss
        correct += sum(argmax(pred, dims=1) .== y)
        total += length(y)
    end
    
    return total_loss / length(data_loader), correct / total
end

"""
    train_step!(model::NeuralNetwork, x::Array{Float32, 4}, y::Vector{Int}; learning_rate=0.01)

Perform one training step on the model.
"""
function train_step!(model::NeuralNetwork, x::Array{Float32, 4}, y::Vector{Int}; learning_rate=0.01)
    # Forward pass
    output = forward(model, x)
    
    # Convert labels to one-hot encoding
    num_classes = size(output, 1)
    y_one_hot = zeros(Float32, num_classes, length(y))
    for (i, label) in enumerate(y)
        y_one_hot[label, i] = 1.0f0
    end
    
    # Reshape y_one_hot to 4D tensor
    y_one_hot_4d = reshape(y_one_hot, num_classes, length(y), 1, 1)
    
    # Compute loss
    loss = cross_entropy_loss(output, y_one_hot_4d)
    
    # Backward pass
    grad = ones(Float32, size(output))
    backward(model, grad)
    
    # Update weights
    for layer in model.layers
        if isa(layer, Conv2D)
            layer.filters .-= learning_rate .* layer.filters_grad
            layer.bias .-= learning_rate .* layer.bias_grad
            layer.filters_grad .= 0
            layer.bias_grad .= 0
        elseif isa(layer, Dense)
            layer.weights .-= learning_rate .* layer.weights_grad
            layer.bias .-= learning_rate .* layer.bias_grad
            layer.weights_grad .= 0
            layer.bias_grad .= 0
        end
    end
    
    return loss
end 