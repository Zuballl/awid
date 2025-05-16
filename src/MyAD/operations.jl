# Dual number type
struct Dual
    val::Float64
    grad::Float64
end

# Conversion
Base.convert(::Type{Float64}, x::Dual) = x.val
Base.convert(::Type{Number}, x::Dual) = x.val

# Basic operations
Base.:+(a::Dual, b::Dual) = Dual(a.val + b.val, a.grad + b.grad)
Base.:+(a::Dual, b::Number) = Dual(a.val + b, a.grad)
Base.:+(a::Number, b::Dual) = Dual(a + b.val, b.grad)

Base.:-(a::Dual, b::Dual) = Dual(a.val - b.val, a.grad - b.grad)
Base.:-(a::Dual, b::Number) = Dual(a.val - b, a.grad)
Base.:-(a::Number, b::Dual) = Dual(a - b.val, -b.grad)

Base.:*(a::Dual, b::Dual) = Dual(a.val * b.val, a.val * b.grad + a.grad * b.val)
Base.:*(a::Dual, b::Number) = Dual(a.val * b, a.grad * b)
Base.:*(a::Number, b::Dual) = Dual(a * b.val, a * b.grad)

Base.:/(a::Dual, b::Dual) = Dual(a.val / b.val, (a.grad * b.val - a.val * b.grad) / (b.val^2))
Base.:/(a::Dual, b::Number) = Dual(a.val / b, a.grad / b)
Base.:/(a::Number, b::Dual) = Dual(a / b.val, -a * b.grad / (b.val^2))

# Power operation
Base.:^(a::Dual, n::Integer) = Dual(a.val^n, n * a.val^(n-1) * a.grad)

# Activation functions for regular numbers
function relu(x::Number)
    return max(0, x)
end

function sigmoid(x::Number)
    return 1.0 / (1.0 + exp(-x))
end

function my_tanh(x::Number)
    return Base.tanh(x)
end

# Activation functions for Dual numbers
function relu(x::Dual)
    if x.val > 0
        return Dual(x.val, x.grad)
    else
        return Dual(0.0, 0.0)
    end
end

function sigmoid(x::Dual)
    s = 1.0 / (1.0 + exp(-x.val))
    return Dual(s, x.grad * s * (1.0 - s))
end

function my_tanh(x::Dual)
    t = Base.tanh(x.val)
    return Dual(t, x.grad * (1.0 - t^2))
end

# Trigonometric functions
function sin(x::Dual)
    return Dual(Base.sin(x.val), x.grad * Base.cos(x.val))
end

function cos(x::Dual)
    return Dual(Base.cos(x.val), -x.grad * Base.sin(x.val))
end

# Gradient computation
function gradient(f, x::Number)
    return f(Dual(x, 1.0)).grad
end

function gradient(f, x::Vector)
    n = length(x)
    grad = zeros(n)
    for i in 1:n
        x_dual = [j == i ? Dual(x[j], 1.0) : x[j] for j in 1:n]
        grad[i] = f(x_dual).grad
    end
    return grad
end 