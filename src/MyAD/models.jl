# Remove type definitions and keep only implementations
function forward(x::CNNVariable)
    return x
end

function backward(x::CNNVariable, grad::Array{Float32, 4})
    if x.grad === nothing
        x.grad = grad
    else
        x.grad .+= grad
    end
    
    if x.grad_fn !== nothing && x.grad_inputs !== nothing
        input_grads = x.grad_fn(x.grad)
        for (input, input_grad) in zip(x.grad_inputs, input_grads)
            backward(input, input_grad)
        end
    end
end

# Add backward method for Matrix{Float32} gradient
function backward(x::CNNVariable, grad::Matrix{Float32})
    # Reshape gradient to 4D tensor
    batch_size = size(grad, 2)
    grad_4d = reshape(grad, size(grad, 1), batch_size, 1, 1)
    # Call backward with 4D gradient
    backward(x, grad_4d)
end

# Operacje na tensorach 4D
function conv2d(x::CNNVariable, filters::Array{Float32, 4}, bias::Array{Float32, 1}, stride::Tuple{Int, Int}, padding::Tuple{Int, Int}, layer)
    @info "conv2d input" size_x=size(x.output) size_filters=size(filters)
    padded_x = pad_input(x.output, padding)
    col_x = im2col(padded_x, (size(filters, 1), size(filters, 2)), stride)
    filters_reshaped = reshape(filters, :, size(filters, 4))
    out = filters_reshaped' * col_x
    out = out .+ bias
    batch_size = size(x.output, 1)
    out_channels = size(filters, 4)
    out_height = div(size(padded_x, 3) - size(filters, 1), stride[1]) + 1
    out_width = div(size(padded_x, 4) - size(filters, 2), stride[2]) + 1
    output = reshape(out, batch_size, out_channels, out_height, out_width)
    @info "conv2d output reshaped" size_output=size(output)
    grad_fn = function(grad::Array{Float32, 4})
        filters_grad = nothing
        bias_grad = nothing
        input_grad = nothing
        grad_perm = permutedims(grad, (2, 1, 3, 4))
        grad_reshaped = reshape(grad_perm, size(filters, 4), :) # (out_channels, batch_size*out_height*out_width)
        filters_grad = grad_reshaped * col_x'
        filters_grad = reshape(filters_grad, size(filters))
        bias_grad = sum(grad, dims=(1,3,4))[:]
        input_grad = filters_reshaped * grad_reshaped
        input_grad = col2im(input_grad, size(x.output), (size(filters, 1), size(filters, 2)), stride)
        layer.filters_grad .= filters_grad
        layer.bias_grad .= bias_grad
        return [input_grad]
    end
    result = CNNVariable(output, nothing, grad_fn, [x])
    return result
end

function maxpool2d(x::CNNVariable, kernel_size::Tuple{Int, Int}, stride::Tuple{Int, Int})
    batch_size, channels, height, width = size(x.output)
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    
    out_height = div(height - kernel_h, stride_h) + 1
    out_width = div(width - kernel_w, stride_w) + 1
    
    output = zeros(Float32, batch_size, channels, out_height, out_width)
    mask = zeros(Int, batch_size, channels, out_height, out_width, 2)  # store h_idx, w_idx
    
    for b in 1:batch_size
        for c in 1:channels
            for h in 1:out_height
                for w in 1:out_width
                    h_start = (h-1) * stride_h + 1
                    w_start = (w-1) * stride_w + 1
                    
                    patch = x.output[b, c, h_start:h_start+kernel_h-1, w_start:w_start+kernel_w-1]
                    output[b, c, h, w], idx = findmax(patch)
                    h_idx, w_idx = Tuple(idx)
                    mask[b, c, h, w, 1] = h_idx
                    mask[b, c, h, w, 2] = w_idx
                end
            end
        end
    end
    
    grad_fn = function(grad::Array{Float32, 4})
        input_grad = zeros(Float32, size(x.output))
        
        for b in 1:batch_size
            for c in 1:channels
                for h in 1:out_height
                    for w in 1:out_width
                        h_start = (h-1) * stride_h + 1
                        w_start = (w-1) * stride_w + 1
                        h_idx = mask[b, c, h, w, 1]
                        w_idx = mask[b, c, h, w, 2]
                        input_grad[b, c, h_start+h_idx-1, w_start+w_idx-1] = grad[b, c, h, w]
                    end
                end
            end
        end
        
        return [input_grad]
    end
    
    result = CNNVariable(output, nothing, grad_fn, [x])
    return result
end

function relu(x::CNNVariable)
    output = max.(0, x.output)
    
    grad_fn = function(grad::Array{Float32, 4})
        input_grad = grad .* (x.output .> 0)
        return [input_grad]
    end
    
    result = CNNVariable(output, nothing, grad_fn, [x])
    return result
end

function flatten(x::CNNVariable)
    batch_size = size(x.output, 1)
    return reshape(x.output, batch_size, :)
end

function grad(x::CNNVariable)
    x.grad
end

function dense(x::AbstractArray{Float32,2}, weights::Array{Float32, 2}, bias::Array{Float32, 1}, layer)
    # x: (in_features, batch)
    output = weights * x .+ bias  # (out_features, batch)
    grad_fn = function(grad::Array{Float32, 2})
        # grad: (out_features, batch)
        weights_grad = grad * x'
        bias_grad = sum(grad, dims=2)
        input_grad = weights' * grad
        layer.weights_grad .= weights_grad
        layer.bias_grad .= bias_grad
        return [input_grad]
    end
    result = CNNVariable(output, nothing, grad_fn, [])
    return result
end

# Add this method to support dense(CNNVariable, ...)
function dense(x::CNNVariable, weights::Matrix{Float32}, bias::Vector{Float32}, layer)
    return dense(x.output, weights, bias, layer)
end

# Add this method to support dense(Array{Float32, 4}, ...)
function dense(x::Array{Float32, 4}, weights::Matrix{Float32}, bias::Vector{Float32}, layer)
    # Reshape input to 2D matrix
    batch_size = size(x, 1)
    x_reshaped = reshape(x, batch_size, :)
    return dense(x_reshaped, weights, bias, layer)
end