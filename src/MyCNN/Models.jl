using ..MyAD: relu, CNNVariable

export CNN, forward, parameters, softmax

# Funkcje aktywacji
function softmax(x::Array{Float32, 2})
    exp_x = exp.(x .- maximum(x, dims=1))
    return exp_x ./ sum(exp_x, dims=1)
end

struct Embedding
    weights::Array{Float32, 2}  # (embedding_dim, vocab_size)
end

# Forward pass dla Embedding
function (layer::Embedding)(x::Array{Int, 2})
    # x: (batch, seq_len)
    # Zwraca: (batch, seq_len, embedding_dim)
    batch, seq_len = size(x)
    embedding_dim, vocab_size = size(layer.weights)
    out = Array{Float32, 3}(undef, batch, seq_len, embedding_dim)
    for i in 1:batch
        for j in 1:seq_len
            idx = x[i, j]
            out[i, j, :] = layer.weights[:, idx]
        end
    end
    return out
end

struct CNN
    embedding::Embedding
    conv_layers::Vector{Conv2D}
    fc::Dense
    max_length::Int

    function CNN(; embedding::Embedding, n_filters::Int, filter_sizes::Vector{Int}, n_classes::Int, seq_len::Int)
        embedding_dim, vocab_size = size(embedding.weights)
        conv_layers = Conv2D[]
        for filter_size in filter_sizes
            push!(conv_layers, Conv2D((filter_size, embedding_dim), 1, n_filters))
        end
        # Compute output size after conv
        kernel_size = filter_sizes[1]
        stride = 1
        out_height = (seq_len - kernel_size) ÷ stride + 1
        n_features = n_filters * out_height
        fc = Dense(n_features, n_classes)
        new(embedding, conv_layers, fc, seq_len)
    end
end

function forward(model::CNN, x::Array{Int, 2})
    embedded = model.embedding(x)  # (batch, seq_len, embedding_dim)
    batch, seq_len, embedding_dim = size(embedded)
    x_cnn = reshape(embedded, batch, 1, seq_len, embedding_dim)
    x_var = CNNVariable(x_cnn)

    conv_outputs = []
    for conv in model.conv_layers
        conv_out = conv(x_var)
        conv_out = relu(conv_out)
        push!(conv_outputs, conv_out)
    end

    # Po Conv2D: (batch, n_filters, out_height, 1)
    # Spłaszcz do (batch, n_filters * out_height)
    conv_out = conv_outputs[1].output  # (batch, n_filters, out_height, 1)
    batch, n_filters, out_height, _ = size(conv_out)
    flat = reshape(conv_out, batch, n_filters * out_height)  # (batch, features)
    flat_t = permutedims(flat, (2,1))  # (features, batch)
    flat_var = CNNVariable(flat_t)

    output = model.fc(flat_var)
    return CNNVariable(softmax(output.output), nothing, nothing, nothing)
end

function parameters(model::CNN)
    params = []
    push!(params, model.embedding.weights)
    for conv in model.conv_layers
        push!(params, conv.filters, conv.bias)
    end
    push!(params, model.fc.weights, model.fc.bias)
    return params
end