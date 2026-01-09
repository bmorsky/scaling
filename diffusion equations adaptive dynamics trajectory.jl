using DifferentialEquations, Plots, LaTeXStrings

T = 100
N = 1
β₁ = 1.4
β₂ = 0.7
Y₁ = 1
Y₂ = 1
γ₁ = Y₁*N^β₁
γ₂ = Y₂*N^β₂
ϵ = 0.001

using PyCall
@pyimport matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = false
plt.rcParams["image.interpolation"] = "none"

function calc_utility(pop,alpha,theta)
    return alpha*log(1.0 + theta*γ₁*(max(pop,0)^(1-β₁))/(Y₁*N^β₁)) + (1-alpha)*log(1.0 + γ₂*(max(pop,0)^(1-β₂))/(Y₂*N^β₂))
end

function selection_gradient(pop,theta)
    return log(1.0 + theta*γ₁*(max(pop,0)^(1-β₁))/(Y₁*N^β₁)) - log(1.0 + γ₂*(max(pop,0)^(1-β₂))/(Y₂*N^β₂))
end

function replicator!(du,u,p,t)
    α₁, α₂, Θ = p
    du[1] = u[1]*(1-u[1])*(calc_utility(u[1], α₁, Θ) - calc_utility(1-u[1], α₂, Θ))
end

tspan = (0.0,T)

# Define invasion fitness
function s(Θ, r, r1, r2, u0)
    # m = r + 1e-4
    # prob = ODEProblem(replicator!,[u0],tspan,(r1,r2,Θ))
    # sol = solve(prob)
    # resident = calc_utility(sol[end][1],r,Θ)
    resident = calc_utility(u0,r,Θ)
    # mutant = calc_utility(sol[end][1],m)
    # growth_rate = mutant - resident
    growth_rate = selection_gradient(u0,Θ)
    if growth_rate > 0
        return 1
    elseif growth_rate < 0
        return -1
    else
        return 0
    end
end

function trajectory(Θ,r1,r2,ic,overshoot_adj)
    prob = ODEProblem(replicator!,[ic],tspan,(r1,r2,Θ))
    u0 = min(max(solve(prob)[end][1],0),1)
    output = zeros(2000,3)
    for t = 1:2000
        r1_new = r1 + ϵ*s(Θ, r1, r1, r2, u0)
        r2_new = r2 + ϵ*s(Θ, r2, r2, r1, 1-u0)
        if ((r1 > r2 && r1_new < r2_new) || (r1 < r2 && r1_new > r2_new)) && overshoot_adj==true
            r1_new = (r1_new+r2_new)/2
            r2_new = r1_new
            u0 = 0.5
        end
        r1 = min(max(r1_new,0),1)
        r2 = min(max(r2_new,0),1)
        prob = ODEProblem(replicator!,[u0],(0.0,100.0),(r1,r2,Θ))
        u0 = min(max(solve(prob)[end][1],1e-8),1-1e-8)
        u0=solve(prob)[end][1]
        output[t,:] = [r1,r2,u0]
    end
    return output[:,1], output[:,2], output[:,3]
end

theta8_ic9_1, theta8_ic9_2, theta8_ic9_p0 = trajectory(0.8,0.2,0.1,0.9,false)
theta8_ic1_1, theta8_ic1_2, theta8_ic1_p0 = trajectory(0.8,0.2,0.1,0.1,false)
theta8_ic9_1_adj, theta8_ic9_2_adj, theta8_ic9_p0_adj = trajectory(0.8,0.2,0.1,0.9,true)
theta8_ic1_1_adj, theta8_ic1_2_adj, theta8_ic1_p0_adj = trajectory(0.8,0.2,0.1,0.1,true)
theta6_ic9_1, theta6_ic9_2, theta6_ic9_p0 = trajectory(0.6,0.1,0.2,0.9,false)
theta6_ic1_1, theta6_ic1_2, theta6_ic1_p0 = trajectory(0.6,0.1,0.2,0.1,false)

pyplot()

plot(theta8_ic9_1,colour=RGB(1,113/255,206/255),label=L"$\alpha_{1,τ}$",linewidth=3)
plot!(theta8_ic9_2,colour=RGB(1/255,205/255,254/255),label=L"$\alpha_{2,τ}$",linewidth=3,linestyle=:dash)
plot!(theta8_ic9_p0,colour=RGB(5/255,1,161/255),xlabel=L"Evolutionary time, $\tau$",linestyle=:dot,
    label=L"$\bar{p}_{τ}$",linewidth=3,legend=:outerright,ylims=(-0.1,1.1),xlims=(-10,2010), size=(300,200))

savefig("traj_0.8_ic0.9.pdf")

plot(theta8_ic1_1,colour=RGB(1,113/255,206/255),label=L"$\alpha_{1,τ}$",linewidth=3)
plot!(theta8_ic1_2,colour=RGB(1/255,205/255,254/255),label=L"$\alpha_{2,τ}$",linewidth=3,linestyle=:dash)
plot!(theta8_ic1_p0,colour=RGB(5/255,1,161/255),xlabel=L"Evolutionary time, $\tau$",linestyle=:dot,
    label=L"$\bar{p}_{τ}$",linewidth=3,legend=:outerright,ylims=(-0.1,1.1),xlims=(-10,2010), size=(300,200))

savefig("traj_0.8_ic0.1.pdf")

plot(theta8_ic9_1_adj,colour=RGB(1,113/255,206/255),label=L"$\alpha_{1,τ}$",linewidth=3)
plot!(theta8_ic9_2_adj,colour=RGB(1/255,205/255,254/255),label=L"$\alpha_{2,τ}$",linewidth=3,linestyle=:dash)
plot!(theta8_ic9_p0_adj,colour=RGB(5/255,1,161/255),xlabel=L"Evolutionary time, $\tau$",linestyle=:dot,
    label=L"$\bar{p}_{τ}$",linewidth=3,legend=:outerright,ylims=(-0.1,1.1),xlims=(-10,2010), size=(300,200))

savefig("traj_0.8_ic0.9_adj.pdf")

plot(theta8_ic1_1_adj,colour=RGB(1,113/255,206/255),label=L"$\alpha_{1,τ}$",linewidth=3)
plot!(theta8_ic1_2_adj,colour=RGB(1/255,205/255,254/255),label=L"$\alpha_{2,τ}$",linewidth=3,linestyle=:dash)
plot!(theta8_ic1_p0_adj,colour=RGB(5/255,1,161/255),xlabel=L"Evolutionary time, $\tau$",linestyle=:dot,
    label=L"$\bar{p}_{τ}$",linewidth=3,legend=:outerright,ylims=(-0.1,1.1),xlims=(-10,2010), size=(300,200))

savefig("traj_0.8_ic0.1_adj.pdf")

plot(theta6_ic9_1,colour=RGB(1,113/255,206/255),label=L"$\alpha_{1,τ}$",linewidth=3)
plot!(theta6_ic9_2,colour=RGB(1/255,205/255,254/255),label=L"$\alpha_{2,τ}$",linewidth=3,linestyle=:dash)
plot!(theta6_ic9_p0,colour=RGB(5/255,1,161/255),xlabel=L"Evolutionary time, $\tau$",linestyle=:dot,
    label=L"$\bar{p}_{τ}$",linewidth=3,legend=:outerright,ylims=(-0.1,1.1),xlims=(-10,2010), size=(300,200))

savefig("traj_0.6_ic0.9.pdf")

plot(theta6_ic1_1,colour=RGB(1,113/255,206/255),label=L"$\alpha_{1,τ}$",linewidth=3)
plot!(theta6_ic1_2,colour=RGB(1/255,205/255,254/255),label=L"$\alpha_{2,τ}$",linewidth=3,linestyle=:dash)
plot!(theta6_ic1_p0,colour=RGB(5/255,1,161/255),xlabel=L"Evolutionary time, $\tau$",linestyle=:dot,
    label=L"$\bar{p}_{τ}$",linewidth=3,legend=:outerright,ylims=(-0.1,1.1),xlims=(-10,2010), size=(300,200))

savefig("traj_0.6_ic0.1.pdf")