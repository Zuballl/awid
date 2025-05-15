module TensorOps

export pad_input, im2col, col2im

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

end # module 