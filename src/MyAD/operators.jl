export ScalarOp, BroadcastedOp, forward, back, update!

import Base: +, -, *, /, ^, sin, exp, log
import Base.Broadcast: broadcasted
using ..AD: GraphNode, Operator, Constant, Variable

mutable struct ScalarOp{F} <: Operator
    f::F
    inputs::Tuple{GraphNode,Union{GraphNode,Nothing}}
    output::Any
    gradient::Any
    name::String
    ScalarOp(fun, a::GraphNode, b::GraphNode; name=string(fun)) =
        new{typeof(fun)}(fun, (a,b), nothing, nothing, name)
    ScalarOp(fun, a::GraphNode; name=string(fun)) =
        new{typeof(fun)}(fun, (a,nothing), nothing, nothing, name)
end

mutable struct BroadcastedOp{F} <: Operator
    f::F
    inputs::Tuple{GraphNode,Union{GraphNode,Nothing}}
    output::Any
    gradient::Any
    name::String
    BroadcastedOp(fun, a::GraphNode, b::GraphNode; name=string(fun)*".") =
        new{typeof(fun)}(fun, (a,b), nothing, nothing, name)
    BroadcastedOp(fun, a::GraphNode; name=string(fun)*".") =
        new{typeof(fun)}(fun, (a,nothing), nothing, nothing, name)
end


+(x::GraphNode, y::GraphNode) = ScalarOp(+, x, y)
+(x::GraphNode, y::Number)    = ScalarOp(+, x, Constant(y))
+(x::Number,    y::GraphNode) = ScalarOp(+, Constant(x), y)
-(x::GraphNode, y::GraphNode) = ScalarOp(-, x, y)
-(x::GraphNode, y::Number)    = ScalarOp(-, x, Constant(y))
-(x::Number,    y::GraphNode) = ScalarOp(-, Constant(x), y)
*(x::GraphNode, y::GraphNode) = ScalarOp(*, x, y)
*(x::GraphNode, y::Number)    = ScalarOp(*, x, Constant(y))
*(x::Number,    y::GraphNode) = ScalarOp(*, Constant(x), y)
/(x::GraphNode, y::GraphNode) = ScalarOp(/, x, y)
/(x::GraphNode, y::Number)    = ScalarOp(/, x, Constant(y))
/(x::Number,    y::GraphNode) = ScalarOp(/, Constant(x), y)
^(x::GraphNode, y::GraphNode) = ScalarOp(^, x, y)
^(x::GraphNode, y::Number)    = ScalarOp(^, x, Constant(y))
^(x::Number,    y::GraphNode) = ScalarOp(^, Constant(x), y)
sin(x::GraphNode)             = ScalarOp(sin, x)
broadcasted(f, x::GraphNode)  = BroadcastedOp(f, x)

Base.broadcasted(::typeof(+), x::Variable, y::Number) = BroadcastedOp(+, x, Constant(y))
Base.broadcasted(::typeof(+), x::Number, y::Variable) = BroadcastedOp(+, Constant(x), y)
Base.broadcasted(::typeof(+), x::Variable, y::Variable) = BroadcastedOp(+, x, y)
Base.broadcasted(::typeof(-), x::Variable, y::Number) = BroadcastedOp(-, x, Constant(y))
Base.broadcasted(::typeof(-), x::Number, y::Variable) = BroadcastedOp(-, Constant(x), y)
Base.broadcasted(::typeof(-), x::Variable, y::Variable) = BroadcastedOp(-, x, y)
Base.broadcasted(::typeof(*), x::Variable, y::Number) = BroadcastedOp(*, x, Constant(y))
Base.broadcasted(::typeof(*), x::Number, y::Variable) = BroadcastedOp(*, Constant(x), y)
Base.broadcasted(::typeof(*), x::Variable, y::Variable) = BroadcastedOp(*, x, y)
Base.broadcasted(::typeof(/), x::Variable, y::Number) = BroadcastedOp(/, x, Constant(y))
Base.broadcasted(::typeof(/), x::Number, y::Variable) = BroadcastedOp(/, Constant(x), y)
Base.broadcasted(::typeof(/), x::Variable, y::Variable) = BroadcastedOp(/, x, y)


function forward(op::ScalarOp)
    a, b = op.inputs
    if b === nothing
        op.output = op.f(a.output)
    else
        op.output = op.f(a.output, b.output)
    end
end

function forward(op::BroadcastedOp)
    a, b = op.inputs
    if b === nothing
        op.output = op.f.(a.output)
    else
        op.output = op.f.(a.output, b.output)
    end
end


back(::typeof(+),   x, y, g) = (g, g)
back(::typeof(-),   x, y, g) = (g, -g)
back(::typeof(*),   x, y, g) = (g*y, g*x)
back(::typeof(/),   x, y, g) = (g*(1/y), -g*x/(y^2))
back(::typeof(^),   x, y, g) = (g*y*x^(y-1), g*log(abs(x))*x^y)
back(::typeof(sin), x, _, g) = (g*cos(x), nothing)
back(::typeof(exp), x, _, g) = (g .* exp.(x),)
back(::typeof(log), x, _, g) = (g .* (1.0 ./ x),)


function update!(node::Constant, grad)
    return
end

function update!(node::Operator, grad)
    if isnothing(node.gradient)
        node.gradient = grad
    else
        node.gradient .+= grad
    end
end

function update!(node::Variable, grad)
    if isnothing(node.gradient)
        node.gradient = grad
    else
        node.gradient .+= grad
    end
end

function *(A::Matrix{Float64}, x::Variable)
    return Variable(A * x.output)
end

function *(A::Variable, B::Matrix{Float64})
    return Variable(A.output * B)
end