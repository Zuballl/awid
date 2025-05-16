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

Cross Entropy loss function with debug prints for NaN diagnosis.
"""
function cross_entropy_loss(pred::CNNVariable, target::Array{Float32, 4})
    # Softmax
    exp_pred = exp.(pred.output)
    softmax_pred = exp_pred ./ sum(exp_pred, dims=1)
    # Debug
    if any(isnan.(softmax_pred))
        println("NaN in softmax_pred!")
        display(softmax_pred)
    end
    if any(target .> 1) || any(target .< 0)
        println("Target out of range!")
        display(target)
    end
    # Cross entropy
    ce = -mean(sum(target .* log.(softmax_pred .+ 1e-10), dims=1))
    if isnan(ce)
        println("NaN in cross_entropy_loss! softmax_pred:"); display(softmax_pred)
        println("target:"); display(target)
    end
    return ce
end 