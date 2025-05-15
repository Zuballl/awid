using ..AD: MLPVariable

export train!

function train!(model::Chain, X::Matrix{Float64}, Y::Matrix{Float64}, opt::Adam; epochs=50, batchsize=16)
    N = size(X, 2)  
    n_batches = ceil(Int, N / batchsize)

    for epoch in 1:epochs
        idx = shuffle(1:N)
        X_shuffled = X[:, idx]
        Y_shuffled = Y[:, idx]

        epoch_loss = 0.0
        for i in 1:n_batches
            start_idx = (i-1)*batchsize + 1
            end_idx = min(i*batchsize, N)
            batch_idx = start_idx:end_idx

            x_batch = X_shuffled[:, batch_idx]
            y_batch = Y_shuffled[:, batch_idx]

            x_var = MLPVariable(x_batch)
            ŷ = model(x_var)

            loss = -sum(y_batch .* log.(ŷ.output .+ 1e-10)) / size(x_batch, 2)
            epoch_loss += loss

            grad = ŷ.output .- y_batch
            backward(model, grad)
            
            params = MLPVariable[]
            for layer in model.layers
                if typeof(layer) == Dense
                    push!(params, MLPVariable(layer.weights))
                    bias_matrix = reshape(layer.bias, length(layer.bias), 1)
                    push!(params, MLPVariable(bias_matrix))
                end
            end
            update!(opt, params)
        end
        avg_loss = epoch_loss / n_batches
        @info "Training progress" epoch=epoch avg_loss=avg_loss
    end
end