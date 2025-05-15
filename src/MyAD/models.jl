export GraphNode, Operator, Constant, Variable, CNNVariable

abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output::T
end

mutable struct Variable <: GraphNode
    output::Any
    gradient::Any
    name::String
    Variable(val; name="?") = new(val, nothing, name)
end

Base.length(v::Variable) = length(v.output)

mutable struct CNNVariable
    output::Array{Float32, 4}  # (batch_size, channels, height, width)
    grad::Union{Array{Float32, 4}, Nothing}
    grad_fn::Union{Function, Nothing}
    grad_inputs::Union{Vector{CNNVariable}, Nothing}
end

CNNVariable(output::Array{Float32, 4}) = CNNVariable(output, nothing, nothing, nothing)

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

# Operacje na tensorach 4D
function conv2d(x::CNNVariable, filters::Array{Float32, 4}, bias::Array{Float32, 1}, stride::Tuple{Int, Int}, padding::Tuple{Int, Int}, layer)
    padded_x = pad_input(x.output, padding)
    col_x = im2col(padded_x, size(filters, 1:2), stride)
    filters_reshaped = reshape(filters, :, size(filters, 4))
    out = filters_reshaped' * col_x
    out = out .+ bias
    batch_size = size(x.output, 1)
    out_channels = size(filters, 4)
    out_height = div(size(padded_x, 3) - size(filters, 1), stride[1]) + 1
    out_width = div(size(padded_x, 4) - size(filters, 2), stride[2]) + 1
    output = reshape(out, out_channels, batch_size, out_height, out_width)
    grad_fn = function(grad::Array{Float32, 4})
        grad_reshaped = reshape(grad, size(filters, 4), :)
        filters_grad = grad_reshaped * col_x'
        filters_grad = reshape(filters_grad, size(filters))
        bias_grad = sum(grad, dims=(1,3,4))[1,:]
        input_grad = filters_reshaped * grad_reshaped
        input_grad = col2im(input_grad, size(x.output), size(filters, 1:2), stride)
        # Zapisz gradienty do pól warstwy
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
    mask = zeros(Int, batch_size, channels, out_height, out_width)
    
    for b in 1:batch_size
        for c in 1:channels
            for h in 1:out_height
                for w in 1:out_width
                    h_start = (h-1) * stride_h + 1
                    w_start = (w-1) * stride_w + 1
                    
                    patch = x.output[b, c, h_start:h_start+kernel_h-1, w_start:w_start+kernel_w-1]
                    output[b, c, h, w], idx = findmax(patch)
                    mask[b, c, h, w] = idx
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
                        
                        idx = mask[b, c, h, w]
                        h_idx = div(idx-1, kernel_w) + 1
                        w_idx = mod(idx-1, kernel_w) + 1
                        
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
    output = reshape(x.output, batch_size, :)
    
    grad_fn = function(grad::Array{Float32, 2})
        input_grad = reshape(grad, size(x.output))
        return [input_grad]
    end
    
    result = CNNVariable(output, nothing, grad_fn, [x])
    return result
end

function grad(x::CNNVariable)
    x.grad
end

function dense(x::CNNVariable, weights::Array{Float32, 2}, bias::Array{Float32, 1}, layer)
    # x.output: (batch_size, in_features)
    output = weights * x.output' .+ bias
    output = output  # (out_features, batch_size)
    grad_fn = function(grad::Array{Float32, 2})
        # grad: (out_features, batch_size)
        weights_grad = grad * x.output
        bias_grad = sum(grad, dims=2)
        input_grad = weights' * grad
        # Zapisz gradienty do pól warstwy
        layer.weights_grad .= weights_grad
        layer.bias_grad .= bias_grad
        return [input_grad']
    end
    result = CNNVariable(output, nothing, grad_fn, [x])
    return result
end