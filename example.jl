using JuMP, Ipopt, LinearAlgebra

model = Model(Ipopt.Optimizer)

@variable(model, x[1:2])
# @variable(model, s)
s = 0

@objective(model, Min, -3*x[1]-2*x[2])

@constraint(model, 1 - x[1] + 2 * x[2] >= -s)
@constraint(model, 2 - x[1] + x[2] >= -s)
@constraint(model, 6 - 2 * x[1] + x[2] >= -s)
@constraint(model, 5 - x[1] >= -s)
@constraint(model, 16 - 2 * x[1] - x[2] >= -s)
@constraint(model, 12 - x[1] - x[2] >= -s)
@constraint(model, 21 - x[1] - 2 * x[2] >= -s)
@constraint(model, 10 - x[2] >= -s)
@constraint(model, x[1] >= -s)
@constraint(model, x[2] >= -s)

optimize!(model)

println("status = ", termination_status(model))
println("x = ", value.(x))
# println("s = ", value(s))
println("obj = ", objective_value(model))
