using ..MyAD: CNNVariable
using Statistics: mean

export mse_loss, cross_entropy_loss

"""
    mse_loss(pred::CNNVariable, target::Array{Float32, 4})

Mean Squared Error loss function.
"""
function mse_loss(pred::CNNVariable, target::Array{Float32, 4})
    return mean((pred.output .- target).^2)
end

"""
    cross_entropy_loss(pred::CNNVariable, target::Array{Float32, 4})

Cross Entropy loss function.
"""
function cross_entropy_loss(pred::CNNVariable, target::Array{Float32, 4})
    # Softmax
    exp_pred = exp.(pred.output)
    softmax_pred = exp_pred ./ sum(exp_pred, dims=1)
    
    # Cross entropy
    return -mean(sum(target .* log.(softmax_pred .+ 1e-10), dims=1))
end 