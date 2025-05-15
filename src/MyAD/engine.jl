export topological_sort, forward!, backward!, gradient

using ..AD: GraphNode, Operator, ScalarOp, BroadcastedOp
using ..AD: forward, back, update!

function visit(nothing, visited, order)
    return
end

function visit(n::GraphNode, visited, order)
    if n ∉ visited
        push!(visited, n)
        if n isa Operator
            for child in getfield(n, :inputs)
                visit(child, visited, order)
            end
        end
        push!(order, n)
    end
end

function topological_sort(root::GraphNode)
    visited = Set{GraphNode}()
    order   = GraphNode[]
    visit(root, visited, order)
    return order
end

function forward!(order::Vector{GraphNode})
    for node in order
        if node isa Operator
            if node isa BroadcastedOp
                a = node.inputs[1]
                node.output = similar(a.output)
            end
            forward(node)
        end
        if node isa Operator || node isa Variable
            node.gradient = nothing
        end
    end
    return order[end].output
end

function backward!(order::Vector{GraphNode}; seed=1)
    order[end].gradient = seed

    for node in reverse(order)
        if node isa ScalarOp
            a, b = node.inputs
            x = a.output
            y = b === nothing ? nothing : b.output
            grads = back(node.f, x, y, node.gradient)
            for (c, g) in zip((a, b), grads)
                if c !== nothing && g !== nothing
                    update!(c, g)
                end
            end

        elseif node isa BroadcastedOp
            a = node.inputs[1]
            g_tuple = back(node.f, a.output, nothing, node.gradient)
            update!(a, g_tuple[1])
        end
    end
end


function gradient(f, x::Variable)
    y     = f(x)
    order = topological_sort(y)
    forward!(order)
    backward!(order)
    return x.gradient
end