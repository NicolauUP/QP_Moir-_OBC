

using FFTW

function MoireVectors(a1,a2,m,r)
    if gcd(r,3) == 1
        t1 = m .* a1 .+ (m+r) .* a2
        t2 = -(m+r) .* a1 .+ (2*m+r) .* a2
    else
        t1 = (m+r/3) .* a1 .+ r/3 .* a2
        t2 = -r/3 .* a1 + (m + 2*r/3) .* a2
    end
    return t1,t2
end

function MoireAngle(m,r)
    return acos((3*m^2 + 3*m*r + 0.5*r^2)/(3*m^2 + 3*m*r + r^2))
end

function RotationMatrix(θ)
    return SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
end

function HoppingModulation(r,R_max,width,hopping)
    return hopping * 0.5 * (tanh((norm(r) + R_max) / width) - (tanh((norm(r) - R_max) / width)))
end

# function MassModulation(r,R_max,width,value)
#     return value * (1 )
# end
function HoppingPerp(dr, d_perp , d, δ , t , t_perp )
    term1 = (d_perp^2 / (norm(dr)^2 + d_perp^2)) * t_perp * exp((d_perp - sqrt(norm(dr)^2+d_perp^2)) / δ)
    term2 = norm(dr)^2 / (norm(dr)^2 + d_perp^2) * (-t) * exp((d - sqrt(norm(dr)^2+d_perp^2)) / δ)
    return term1 + term2
end


function ChargeRatio(Es, Vecs, r_max,sites1, sites2)
    Result = zeros(Float64, length(Es))

    Condition1 = (norm.(sites1) .<= r_max)
    Condition2 = (norm.(sites2) .<= r_max)
    Condition = [Condition1; Condition2]
    Fraction_Of_Sites = sum(Condition) / length(Condition)

    for i in 1:length(Es)
        Result[i] = real(sum(abs2.(Vecs[1:end, i]) .* Condition) / Fraction_Of_Sites)
    end
    return Result
end

function LdosBins(Energies, EigenValues, Vectors, R_Max,sites)
  Results =  zeros(Float64,length(Energies))
    for i in eachindex(EigenValues)
        if Energies[1] <= EigenValues[i] <= Energies[end]
            indice = floor(length(Energies).*(EigenValues[i] .- Energies[1]) ./(Energies[end] - Energies[1]) )
            for r in eachindex(sites)
                if norm(sites[r]) <= R_Max
                    Results[i] += abs2.(Vectors[r,i])
                    
                end
            end
        end
    end
    return Results ./ (length(sites) * (Energies[2] - Energies[1])) #Normalized by bin and number of sites
end

function ComputeIpr(Es,Vecs,sites,R_Max)
    result = zeros(Float64, length(Es))
    Condition = (norm.(sites) .<= R_Max)
    for i in eachindex(Es)
        result[i] = sum(abs.(Vecs[1:end,i]).^4 .* Condition) / (sum(abs2.(Vecs[1:end,i]) .* Condition).^2)
    end
    return result
end

function FindFirstOrbB(sites1,d_cc)
    N = length(sites1)
    N_teste = Int64(floor(N/2))
    for i in N_teste-100:N_teste+100
        dist_firstA = norm(sites1[1] .- sites1[i])
        if dist_firstA <= 5*d_cc
            NB = i
            return NB
            break
        end
    end
    
end




function SortVecsFFT(nev, L_F ,sites1, δ_F, Vecs, inverse,NB)

    Vectors2D_Rhombus_OrbA = zeros(Complex{Float64}, nev, Int64(L_F),Int64(L_F))
    Vectors2D_Rhombus_OrbB = zeros(Complex{Float64}, nev, Int64(L_F),Int64(L_F))

    for i in eachindex(sites1)
        Index = Int64.(floor.(inverse * (sites1[i] .- δ_F)))
        # println(Index[1] , " ",Index[2]," " , L_F)
        if 0 <= Index[1] < L_F && 0 <= Index[2] < L_F
            if i <= NB
                
                Vectors2D_Rhombus_OrbA[1:end, Index[1]+1,Index[2]+1] = Vecs[i,1:end]
            else    
                
                Vectors2D_Rhombus_OrbB[1:end, Index[1]+1,Index[2]+1] = Vecs[i,1:end]
            end
        end

    end 
    
    Psi_KA = zeros(Complex{Float64}, nev, L_F,L_F)
    Psi_KB = zeros(Complex{Float64}, nev, L_F,L_F)

    for i in 1:nev
        
        Psi_KA[i,1:end,1:end]= abs.(fft(Vectors2D_Rhombus_OrbA[i,1:end,1:end]))
        Psi_KB[i,1:end,1:end]= abs.(fft(Vectors2D_Rhombus_OrbB[i,1:end,1:end]))
    
    end

    return Psi_KA, Psi_KB
end

function ComputeIprk(Psi_KA, Psi_KB, EigenValues)

    resultado = zeros(Float64, length(EigenValues)) 
    for i in eachindex(resultado)
        resultado[i] = sum(abs.(Psi_KA[i,1:end,1:end]).^4 +abs.(Psi_KB[i,1:end,1:end]).^4) /(sum(abs2.(Psi_KA[i,1:end,1:end]) + abs2.(Psi_KB[i,1:end,1:end]) )).^2
    end
return resultado
end
