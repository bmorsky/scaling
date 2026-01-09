using DifferentialEquations, Plots, LaTeXStrings

Θ = 0.8#0.6155722065
L = 350
T = 200
N = 1
β₁ = 1.4
β₂ = 0.7
Y₁ = 1
Y₂ = 1
γ₁ = Θ*Y₁*N^β₁
γ₂ = Y₂*N^β₂

using PyCall
@pyimport matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = false
plt.rcParams["image.interpolation"] = "none"

function calc_utility(pop,alpha)
    return alpha*log(1.0 + γ₁*(max(pop,0)^(1-β₁))/(Y₁*N^β₁)) + (1-alpha)*log(1.0 + γ₂*(max(pop,0)^(1-β₂))/(Y₂*N^β₂))
end

function selection_gradient(pop)
    return log(1.0 + γ₁*(max(pop,0)^(1-β₁))/(Y₁*N^β₁)) - log(1.0 + γ₂*(max(pop,0)^(1-β₂))/(Y₂*N^β₂))
end

function replicator!(du,u,p,t)
    α₁, α₂ = p
    du[1] = u[1]*(1-u[1])*(calc_utility(u[1], α₁) - calc_utility(1-u[1], α₂))
end

tspan = (0.0,T)

# Define trait range
traits = range(0.0, 1.0, length=L)

# Define invasion fitness
function s(r, r1, r2, u0)
    # m = r + 1e-4
    prob = ODEProblem(replicator!,u0,tspan,(r1,r2))
    sol = solve(prob)
    resident = calc_utility(sol[end][1],r)
    # println(sol[end][1])
    # mutant = calc_utility(sol[end][1],m)
    # growth_rate = mutant - resident
    growth_rate = selection_gradient(sol[end][1])
    if growth_rate > 0
        return 1
    elseif growth_rate < 0
        return -1
    else
        return 0
    end
end

# Create matrix of invasion fitness values
Z1 = [s(r1, r1, r2, [0.2]) for r1 in traits, r2 in traits]
Z2 = [s(r2, r2, r1, [0.8]) for r1 in traits, r2 in traits]

Z = zeros(L,L)
for i=1:L
    for j=1:L
        if Z1[i,j] > 0 && Z2[i,j] > 0
            Z[i,j] = 1
        elseif Z1[i,j] > 0 && Z2[i,j] < 0
            Z[i,j] = 2
        elseif Z1[i,j] < 0 && Z2[i,j] > 0
            Z[i,j] = 3
        elseif Z1[i,j] < 0 && Z2[i,j] < 0
            Z[i,j] = 4
        end
    end
end

pyplot()

# Plot PIP
heatmap(traits, traits, transpose(Z),
    xlabel=L"\alpha_1",
    ylabel=L"\alpha_2",
    color=[RGB(1,113/255,206/255),RGB(1/255,205/255,254/255),RGB(5/255,1,161/255),RGB(185/255,103/255,1)],
    aspect_ratio=:equal, legend = :none, grid = false, interpolation = false, clims=(1,4))

    mycolours = [RGB(1,113/255,206/255),RGB(1/255,205/255,254/255),RGB(5/255,1,161/255),RGB(185/255,103/255,1)]
    mylabels = [L"S_1'>0,S_2'>0", L"S_1'>0,S_2'<0",L"S_1'<0,S_2'>0",L"S_1'<0,S_2'<0"]

for i in 1:4
    scatter!(xlims=(0,1),ylims=(0,1),grid = false, interpolation = false, [NaN], [NaN], color=mycolours[i], label=mylabels[i], markershape=:rect,
    legend = :outerright,aspect_ratio=:equal, size=(375,250))
end

savefig("PIP_Theta_$Θ.pdf")