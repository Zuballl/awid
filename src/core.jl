# Dual number type for automatic differentiation
struct Dual{T}
    val::T
    grad::T
end

# Type conversion
Base.convert(::Type{T}, x::Dual{T}) where T = x.val
Base.convert(::Type{T}, x::Dual{S}) where {T,S} = convert(T, x.val)

# Basic arithmetic operations for Dual numbers
Base.:+(a::Dual, b::Dual) = Dual(a.val + b.val, a.grad + b.grad)
Base.:-(a::Dual, b::Dual) = Dual(a.val - b.val, a.grad - b.grad)
Base.:*(a::Dual, b::Dual) = Dual(a.val * b.val, a.val * b.grad + a.grad * b.val)
Base.:/(a::Dual, b::Dual) = Dual(a.val / b.val, (a.grad * b.val - a.val * b.grad) / (b.val^2))

# Scalar operations
Base.:+(a::Number, b::Dual) = Dual(a + b.val, b.grad)
Base.:+(a::Dual, b::Number) = Dual(a.val + b, a.grad)
Base.:-(a::Number, b::Dual) = Dual(a - b.val, -b.grad)
Base.:-(a::Dual, b::Number) = Dual(a.val - b, a.grad)
Base.:*(a::Number, b::Dual) = Dual(a * b.val, a * b.grad)
Base.:*(a::Dual, b::Number) = Dual(a.val * b, a.grad * b)
Base.:/(a::Number, b::Dual) = Dual(a / b.val, -a * b.grad / (b.val^2))
Base.:/(a::Dual, b::Number) = Dual(a.val / b, a.grad / b)

# Power operations
Base.:^(a::Dual, n::Integer) = Dual(a.val^n, n * a.val^(n-1) * a.grad)
Base.:^(a::Dual, b::Dual) = Dual(a.val^b.val, a.val^b.val * (b.grad * log(a.val) + b.val * a.grad / a.val))

# Common activation functions
function sigmoid(x::Dual)
    s = 1 / (1 + exp(-x.val))
    return Dual(s, s * (1 - s) * x.grad)
end

function sigmoid(x::Number)
    return 1 / (1 + exp(-x))
end

function my_tanh(x::Dual)
    t = Base.tanh(x.val)
    return Dual(t, (1 - t^2) * x.grad)
end

function my_tanh(x::Number)
    return Base.tanh(x)
end

function relu(x::Dual)
    return Dual(max(0, x.val), x.val > 0 ? x.grad : zero(x.grad))
end

function relu(x::Number)
    return max(0, x)
end

# Trigonometric functions
function sin(x::Dual)
    return Dual(Base.sin(x.val), Base.cos(x.val) * x.grad)
end

function cos(x::Dual)
    return Dual(Base.cos(x.val), -Base.sin(x.val) * x.grad)
end

# Gradient computation
function gradient(f, x)
    dual_x = Dual(x, one(x))
    result = f(dual_x)
    return result.grad
end

# Vectorized operations
function gradient(f, xs::AbstractArray)
    n = length(xs)
    grads = zeros(eltype(xs), n)
    for i in 1:n
        xs_dual = [Dual(xs[j], j == i ? one(xs[j]) : zero(xs[j])) for j in 1:n]
        result = f(xs_dual)
        grads[i] = result.grad
    end
    return grads
end

# Explicitly import Base functions
import Base: tanh

export Dual, gradient, sigmoid, my_tanh, relu, sin, cos 