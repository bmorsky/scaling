using DifferentialEquations, Plots, LaTeXStrings

a = 0.7
T = 800
N = 1
β₁ = 1.4
β₂ = 0.7
Y₁ = 1
Y₂ = 1
γ₂ = Y₂*N^β₂

function calc_utility(pop,alpha,γ₁)
    return alpha*log(1.0 + γ₁*(max(pop,0)^(1-β₁))/(Y₁*N^β₁)) + (1-alpha)*log(1.0 + γ₂*(max(pop,0)^(1-β₂))/(Y₂*N^β₂))
end

function replicator!(du,u,p,t)
    α₁, α₂, γ₁ = p
    du[1] = u[1]*(1-u[1])*(calc_utility(u[1],α₁,γ₁) - calc_utility(1-u[1],α₂,γ₁))
end

function negreplicator!(du,u,p,t)
    α₁, α₂, γ₁ = p
    du[1] = -u[1]*(1-u[1])*(calc_utility(u[1],α₁,γ₁) - calc_utility(1-u[1],α₂,γ₁))
end

tspan = (0.0,T)

output1 = zeros(2001)
Θ = 0.8
γ₁ = Θ*Y₁*N^β₁
global count = 1
for α = 0:0.0005:1
    p = [α,a,γ₁]
    prob = ODEProblem(replicator!,[0.5],tspan,p)
    sol = solve(prob)
    output1[count] = sol[end][1]
    global count = count+1
end

plot(collect(0:0.0005:1),output1,colour=RGB(1,113/255,206/255),
    xlabel=L"\alpha_1",ylabel=L"\bar{p}",linewidth=3,
    label = L"Stable equilibria, $\bar{p}$",
    legend=:bottomright, ylims=(-0.03,0.63), size=(300,200))

savefig("bif_Theta_$Θ.pdf")

output2a = zeros(2001)
output2b = []
output2c = []
xvals = []

Θ = 0.6
γ₁ = Θ*Y₁*N^β₁
global count = 1
for α = 0:0.0005:1
    p = [α,a,γ₁]
    prob = ODEProblem(replicator!,[0.49],tspan,p)
    sol = solve(prob)
    output2a[count] = sol[end][1]
    global count = count+1
end

global count = 1
for α = 0:0.0005:1
    p = [α,a,γ₁]
    
    prob = ODEProblem(replicator!,[0.001],tspan,p)
    sol = solve(prob)
    sol2b = sol[end][1]

    prob = ODEProblem(negreplicator!,[0.4],tspan,p)
    sol = solve(prob)
    sol2c = sol[end][1]

    if sol2c ≥ sol2b
        push!(output2b,sol2b)
        push!(output2c,sol2c)
        push!(xvals,α)
    else
        break
    end
end

plot(collect(0:0.0005:1),output2a,colour=RGB(1,113/255,206/255),
    label=false,linewidth=3, ylims=(-0.035,0.735))
plot!(xvals,output2b,colour=RGB(1,113/255,206/255),
    xlabel=L"\alpha_1",ylabel=L"\bar{p}",linewidth=3,
    label = L"Stable equilibria, $\bar{p}$", 
    ylims=(-0.04,0.84))
plot!(xvals,output2c,colour=RGB(1/255,205/255,254/255),
    xlabel=L"\alpha_1",ylabel=L"\bar{p}",linewidth=3,
    label = L"Unstable equilibria, $\bar{p}$",
    legend=:bottomright, ylims=(-0.04,0.84), size=(300,200))

savefig("bif_Theta_$Θ.pdf")