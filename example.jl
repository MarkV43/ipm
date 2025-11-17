using JuMP, Ipopt, LinearAlgebra

gamma = 1.0
xs = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5]]
ys = [[0.0, 1.0], [1.0, 1.0], [0.5, 1.5]]

model = Model(Ipopt.Optimizer)

@variable(model, a[1:2], start = 1.0)
@variable(model, b)
@variable(model, u[1:length(xs)] >= 0)
@variable(model, v[1:length(ys)] >= 0)

sum_uv = zero(AffExpr)

for ui in u
    add_to_expression!(sum_uv, ui);
end
for vi in v
    add_to_expression!(sum_uv, vi);
end

for (i,x) in enumerate(xs)
    @constraint(model, dot(a, x) - b >= 1 - u[i])
end
for (i,y) in enumerate(ys)
    @constraint(model, dot(a, y) - b <= -1 + v[i])
end

@objective(model, Min,
    sqrt(a[1]^2 + a[2]^2) + gamma*(sum_uv)
)

optimize!(model)

println("status = ", termination_status(model))
println("a = ", value.(a), " b = ", value(b))
println("u = ", value.(u), " v = ", value.(v))
println("obj = ", objective_value(model))

for (x, ui) in zip(xs, u)
    println("constraint = ", value(dot(a, x) - b), " >= ", value(1 - ui))
end
for (y, vi) in zip(ys, v)
    println("constraint = ", value(dot(a, y) - b), " <= ", value(-1 + vi))
end
