
using StaticArrays 
using Quantica
using Arpack
using SparseArrays #Necessário para poder ler sparse matrices do jld
using JLD2 #NPZ do Julia
using FileIO
using LinearAlgebra
using Random 
include("Functions.jl")



#INPUTS

m = parse(Int64, ARGS[1]) #tBLG approximant
r = parse(Int64, ARGS[2]) #tBLG approximant 
P_QP = parse(Int64, ARGS[3]) #Periodic or quasiperiodic
nev = parse(Int64, ARGS[4]) #Number of states in Arpack
RandStack = parse(Int64, Args[5]) #Random Stacking or not


#Geometry Constants
a0 = 2.46 
d_perp = 3.35 
d_cc = 1.42 
Λ = 2.5*d_cc
δ = 0.184*a0

#Hamiltonian parameters
t = 2.7 #eV
t_perp = 0.48  #eV
#Honeycomb lattice vectors
a1 = a0 .* SA[1/2,sqrt(3)/2] #Static array de float64 
a2 = a0 .* SA[-1/2, sqrt(3)/2] #Static Array de Float64

#tBLG periodic approximants


t1,  t2 = MoireVectors(a1,a2,m,r)
L_supercell = norm(t1)

θ = MoireAngle(m,r)
if P_QP == 1
    R = L_supercell * sqrt(3) / 4 #Biggest disk inside unit cell
else
    t1_P, t2_P = MoireVectors(a1,a2,30,1) #Best approximant to the magic angle.
    L_moire = norm(t1_P)
    R = round(L_supercell / L_moire) * sqrt(3) / 4
end

if RandStack == 1
    n1 = rand()
    n2 = rand()
else
    n1 = 0
    n2 = 0
end
#Quantica SETUP

OrbA1 = sublat((0.0,0.0),name=:A1)
OrbB1 = sublat((1/3 * (a1[1]+a2[1]),1/3 * (a1[2]+a2[2])),name=:B1)   

δ_stacking = -1/3 .* (a1 .+ a2) + n1 .* a1 + n2 .* a2 #AB Stacking # + Shift Vector .
OrbA2 = sublat((δ_stacking[1],δ_stacking[2]),name=:A2)
OrbB2 = sublat((δ_stacking[1]+1/3 * (a1[1]+a2[1]),δ_stacking[2] + 1/3 *(a1[2]+a2[2])),name=:B2)
Layer1_UC = lattice(OrbA1, OrbB1, bravais = (a1, a2),dim=2)
Layer2_UC = lattice(OrbA2, OrbB2, bravais = (a1, a2),dim=2)


Layer2_UC = transform(Layer2_UC, r -> RotationMatrix(θ) * r)

Layer1 = supercell(Layer1_UC, region = r -> 0 <= norm(r) <= R)
Layer2 = supercell(Layer2_UC, region = r -> 0 <= norm(r) <= R)

model_graphene = hopping((r,dr) -> HoppingModulation(r,R-4*a0,3*a0,t),range=d_cc+1e-2)
                +onsite(r -> onsitePot(r, R), sublats=:A) +
                + onsite(r -> -onsitePot(r, R), sublats=:B)


h11 = hamiltonian(Layer1, model_graphene)
h22 = hamiltonian(Layer2, model_graphene)

model_inter12 = hopping((r,dr) -> HoppingPerp(r,dr, d_perp , d, δ , t , t_perp, R-4*a0, 3*a0),range = Λ)


H = combine(h11, h22, coupling = model_inter12) 



#Obtain Eigenvalues and EigenVectors

println(" ******* EigenDecomposition Started ******")

σ_ARPACK = 0.0135 #Center of the FlatBand
Vals, Vecs = eigs((H(())), nev=nev, maxiter=1000, tol=1e-4, sigma=σ_ARPACK)
Es = real(Vals)
#Sort Eigenvalues and Eigenvectors 

sorted_indices = sortperm(Es)
Es = Es[sorted_indices]
Vecs = Vecs[1:end, sorted_indices]


println(" ******* EigenDecomposition Finised ******")

#Charge distribution

sites1 = sites(Layer1)
sites2 = sites(Layer2)
rs = LinRange(R/20, R,20)
Resultado_Carga = Vector{Vector{Float64}}()
for i in eachindex(rs)
    push!(Resultado_Carga,ChargeRatio(Es, Vecs, rs[i],sites1, sites2))
end

# #Density of states
sites_Com = [sites1; sites2] 

Energies_DOS = LinRange(0.004*2.7,0.006*2.7, Int64(round((0.002*2.7) / 0.0001)))
DOS_Total = LdosBins(Energies_DOS, Es, Vecs, R,sites_Com)
DOS_Bulk = LdosBins(Energies_DOS, Es, Vecs, 0.8*R,sites_Com)

    
IPR_Bulk = ComputeIpr(Es,Vecs,sites_Com,0.8*R)
IPR_Total = ComputeIpr(Es,Vecs,sites_Com,R)




#Calcular Rhombus e FFT
L_F = Int64(floor(sqrt(length(sites1))))
InvA = [a1[1] a2[1]; a2[2] a2[2]] ^-1
δ_F = -0.5 * L_F .* (a1 .+ a2)

#Implement Find Index of First B 
indexB1 = FindFirstOrbB(sites1,d_cc)
Psi_KA, Psi_KB = SortVecsFFT(nev, L_F, sites1,δ_F, Vecs,InvA,indexB1) # Falta calcular delta, inverse e L_rhombus



#Calculate IPR_K,
IPR_K = ComputeIprk(Psi_KA, Psi_KB, Es)


save("TBG_Results_m=$(m)_r=$(r)_PQP=$(P_QP)_nev=$(nev)_σA=$(σ_ARPACK).jld2",
    "EigenValues",Es,
    "Rs_Charges",rs,
    "Charge",Resultado_Carga,
    "EnergiesDOS",Energies_DOS,
    "DOS_Total",DOS_Total,
    "DOS_Bulk",DOS_Bulk,
    "IPR_Bulk",IPR_Bulk,
    "IPR_Total",IPR_Total,
    "L_F",L_F,
    "IPR_K",IPR_K)