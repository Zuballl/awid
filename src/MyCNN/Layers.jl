using ..MyAD: CNNVariable
import ..AD

export Conv2D, MaxPool2D, Flatten, Dense

struct Conv2D
    filters::Array{Float32, 4}  # (kernel_height, kernel_width, in_channels, out_channels)
    bias::Array{Float32, 1}     # (out_channels,)
    filters_grad::Array{Float32, 4}
    bias_grad::Array{Float32, 1}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    
    function Conv2D(kernel_size::Tuple{Int, Int}, in_channels::Int, out_channels::Int;
                   stride=(1,1), padding=(0,0))
        # Inicjalizacja wag metodą He
        scale = sqrt(2.0 / (kernel_size[1] * kernel_size[2] * in_channels))
        filters = randn(Float32, kernel_size[1], kernel_size[2], in_channels, out_channels) .* scale
        bias = zeros(Float32, out_channels)
        filters_grad = zeros(Float32, kernel_size[1], kernel_size[2], in_channels, out_channels)
        bias_grad = zeros(Float32, out_channels)
        new(filters, bias, filters_grad, bias_grad, stride, padding)
    end
end

struct MaxPool2D
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    
    function MaxPool2D(kernel_size::Tuple{Int, Int}; stride=kernel_size, padding=(0,0))
        new(kernel_size, stride, padding)
    end
end

struct Flatten end

struct Dense
    weights::Array{Float32, 2}
    bias::Array{Float32, 1}
    weights_grad::Array{Float32, 2}
    bias_grad::Array{Float32, 1}
    activation::Function
    
    function Dense(in_features::Int, out_features::Int, activation::Function=relu)
        # Inicjalizacja wag metodą He
        scale = sqrt(2.0 / in_features)
        weights = randn(Float32, out_features, in_features) .* scale
        bias = zeros(Float32, out_features)
        weights_grad = zeros(Float32, out_features, in_features)
        bias_grad = zeros(Float32, out_features)
        new(weights, bias, weights_grad, bias_grad)
    end
end

# Funkcje pomocnicze dla Conv2D
function pad_input(x::Array{Float32, 4}, padding::Tuple{Int, Int})
    batch_size, channels, height, width = size(x)
    padded_height = height + 2 * padding[1]
    padded_width = width + 2 * padding[2]
    
    padded = zeros(Float32, batch_size, channels, padded_height, padded_width)
    padded[:, :, padding[1]+1:padding[1]+height, padding[2]+1:padding[2]+width] = x
    return padded
end

function im2col(x::Array{Float32, 4}, kernel_size::Tuple{Int, Int}, stride::Tuple{Int, Int})
    batch_size, channels, height, width = size(x)
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    
    out_height = div(height - kernel_h, stride_h) + 1
    out_width = div(width - kernel_w, stride_w) + 1
    
    if out_height <= 0 || out_width <= 0
        @warn "im2col: output shape is invalid (out_height=$out_height, out_width=$out_width). Returning empty array."
        return zeros(Float32, kernel_h * kernel_w * channels, 0)
    end
    
    col = zeros(Float32, kernel_h * kernel_w * channels, out_height * out_width * batch_size)
    
    for b in 1:batch_size
        for h in 1:out_height
            for w in 1:out_width
                h_start = (h-1) * stride_h + 1
                w_start = (w-1) * stride_w + 1
                
                patch = x[b, :, h_start:h_start+kernel_h-1, w_start:w_start+kernel_w-1]
                col_idx = (b-1) * out_height * out_width + (h-1) * out_width + w
                col[:, col_idx] = vec(patch)
            end
        end
    end
    
    return col
end

function col2im(col::Array{Float32, 2}, x_shape::Tuple{Int, Int, Int, Int}, 
                kernel_size::Tuple{Int, Int}, stride::Tuple{Int, Int})
    batch_size, channels, height, width = x_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    
    out_height = div(height - kernel_h, stride_h) + 1
    out_width = div(width - kernel_w, stride_w) + 1
    
    if out_height <= 0 || out_width <= 0
        @warn "col2im: output shape is invalid (out_height=$out_height, out_width=$out_width). Returning zeros."
        return zeros(Float32, x_shape)
    end
    
    x = zeros(Float32, x_shape)
    
    for b in 1:batch_size
        for h in 1:out_height
            for w in 1:out_width
                h_start = (h-1) * stride_h + 1
                w_start = (w-1) * stride_w + 1
                
                col_idx = (b-1) * out_height * out_width + (h-1) * out_width + w
                patch = reshape(col[:, col_idx], channels, kernel_h, kernel_w)
                x[b, :, h_start:h_start+kernel_h-1, w_start:w_start+kernel_w-1] = patch
            end
        end
    end
    
    return x
end

# Forward pass dla Conv2D
function (layer::Conv2D)(x::CNNVariable)
    return MyAD.conv2d(x, layer.filters, layer.bias, layer.stride, layer.padding, layer)
end

# Forward pass dla MaxPool2D
function (layer::MaxPool2D)(x::CNNVariable)
    return MyAD.maxpool2d(x, layer.kernel_size, layer.stride)
end

# Forward pass dla Flatten
function (layer::Flatten)(x::CNNVariable)
    return MyAD.flatten(x)
end

# Forward pass dla Dense
function (layer::Dense)(x::CNNVariable)
    return MyAD.dense(x, layer.weights, layer.bias, layer)
end

# Forward pass for Dense with Matrix{Float64} input (for tests)
function forward(layer::Dense, input::Matrix{Float64})
    input32 = Float32.(input)
    x_var = CNNVariable(input32)
    out = layer(x_var)
    return out.output
end 