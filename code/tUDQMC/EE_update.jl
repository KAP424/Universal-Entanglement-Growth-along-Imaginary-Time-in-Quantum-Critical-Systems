

function ctrl_EEicr(path::String,model::_Hubbard_Para,index::Vector{Int64},Sweeps::Int64,λ::Float64,Nλ::Int64,ss::Vector{Matrix{UInt8}},record::Bool)::Vector{Matrix{UInt8}}
    if model.Lattice=="SQUARE"
        name="□"
    elseif model.Lattice=="HoneyComb60"
        name="HC60"
    elseif model.Lattice=="HoneyComb120"
        name="HC120"
    end

    file="$(path)EEicr$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)N$(Nλ)BS$(model.BatchSize).csv"
    atexit() do
        if record
            open(file, "a") do io
                lock(io)
                writedlm(io, O', ',')
                unlock(io)
            end
        end
        writedlm("$(path)ss/SS$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)λ$(Int(round(Nλ*λ))).csv", [ss[1] ss[2]],",")
    end

    rng=MersenneTwister(Threads.threadid()+round(Int,time()*1000))
    elements=(1, 2, 3, 4)

    Gt1=zeros(ComplexF64,model.Ns,model.Ns)
    Gt2=zeros(ComplexF64,model.Ns,model.Ns)
    G01=zeros(ComplexF64,model.Ns,model.Ns)
    G02=zeros(ComplexF64,model.Ns,model.Ns)
    Gt01=zeros(ComplexF64,model.Ns,model.Ns)
    Gt02=zeros(ComplexF64,model.Ns,model.Ns)
    G0t1=zeros(ComplexF64,model.Ns,model.Ns)
    G0t2=zeros(ComplexF64,model.Ns,model.Ns)
    gmInv=zeros(ComplexF64,length(index),length(index))
    detg=0

    tmpO=0
    counter=0
    O=zeros(Sweeps+1)
    O[1]=λ

    I1=I(model.Ns)
    I2=I(length(index))

    
    for loop in 1:Sweeps
        for lt in 1:model.Nt
            if mod(lt,model.WrapTime)==1 || lt==div(model.Nt,2)+1
                Gt1,G01,Gt01,G0t1=G4(model,ss[1],lt,div(model.Nt,2))
                Gt2,G02,Gt02,G0t2=G4(model,ss[2],lt,div(model.Nt,2))
                
                GM=GroverMatrix(G01[index[:],index[:]],G02[index[:],index[:]])
                gmInv=inv(GM)
                detg=abs2(det(GM))^(1/Nλ)
            else
                D1=[model.η[x] for x in ss[1][:,lt]]
                D2=[model.η[x] for x in ss[2][:,lt]]
                Gt1=diagm(exp.(1im*model.α.*D1))*model.eK *Gt1* model.eKinv*diagm(exp.(-1im*model.α.*D1))
                Gt2=diagm(exp.(1im*model.α.*D2))*model.eK *Gt2* model.eKinv*diagm(exp.(-1im*model.α.*D2))
                G0t1=G0t1*model.eKinv*diagm(exp.(-1im*model.α.*D1))
                G0t2=G0t2*model.eKinv*diagm(exp.(-1im*model.α.*D2))
                Gt01=diagm(exp.(1im*model.α.*D1))*model.eK*Gt01
                Gt02=diagm(exp.(1im*model.α.*D2))*model.eK*Gt02

                #####################################################################
                # Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt,div(model.Nt,2))
                # Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt,div(model.Nt,2))
                    
                # if norm(Gt1-Gt1_)+norm(Gt2-Gt2_)+norm(Gt01-Gt01_)+norm(Gt02-Gt02_)+norm(G0t1-G0t1_)+norm(G0t2-G0t2_)>1e-3
                #     println( norm(Gt1-Gt1_),'\n',norm(Gt2-Gt2_),'\n',norm(Gt01-Gt01_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t1-G0t1_),'\n',norm(G0t2-G0t2_) )
                #     error("$lt : WrapTime")
                # end
                #####################################################################


            end

            for x in 1:model.Ns
                b1=transpose(Gt01[x,index[:]]) *(2*G02[index[:],index[:]]-I2)*gmInv
                a1=G0t1[index[:],x]
                Tau1=b1*a1

                sp=Random.Sampler(rng,[i for i in elements if i != ss[1][x,lt]])
                sx1=rand(rng,sp)
                
                Δ1=exp(1im*model.α*(model.η[sx1]-model.η[ss[1][x,lt]]))-1
                r1=1+Δ1*(1-Gt1[x,x])

                p=model.γ[sx1]/model.γ[ss[1][x,lt]]*abs2(r1)^(1-λ)*abs2(r1+Δ1*Tau1)^(λ)

                if rand(rng)<p
                    rho1=Δ1/(r1+Tau1*Δ1)
                    gmInv-=rho1* ( gmInv*a1 .* b1)
                    detg*=abs2(1+Δ1/r1*Tau1)^(1/Nλ)

                    G01+=Δ1/r1* (G0t1[:,x] .* transpose(Gt01[x,:]))
                    Gt01+=Δ1/r1* (Gt1[:,x] .* transpose(Gt01[x,:]))
                    G0t1-=Δ1/r1* (G0t1[:,x] .* transpose( (I1-Gt1)[x,:] ) )
                    Gt1-=Δ1/r1* (Gt1[:,x] .* transpose( (I1-Gt1)[x,:]) )         
                    ss[1][x,lt]=sx1

                    #####################################################################
                    # print('-')
                    # Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt,div(model.Nt,2))
                    
                    # if norm(Gt1-Gt1_)+norm(G01-G01_)+norm(Gt01-Gt01_)+norm(G0t1-G0t1_)>1e-3
                    #     println('\n',norm(Gt1-Gt1_),'\n',norm(G01-G01_),'\n',norm(Gt01-Gt01_),'\n',norm(G0t1-G0t1_))
                    #     error("$lt  $x:,,,asdasdasd")
                    # end
                    #####################################################################
                end



                b2=transpose(Gt02[x,index[:]]) *gmInv
                a2=(2*G01[index[:],index[:]]-I2)*G0t2[index[:],x]
                Tau2=b2*a2

                sp=Random.Sampler(rng,[i for i in elements if i != ss[2][x,lt]])
                sx2=rand(rng,sp)

                Δ2=(exp(1im*model.α*(model.η[sx2]-model.η[ss[2][x,lt]]))-1)
                r2=(1+Δ2*(1-Gt2[x,x]))
                p=model.γ[sx2]/model.γ[ss[2][x,lt]]*(abs2(r2)^(1-λ))*(abs2(r2+Δ2*Tau2)^λ)

                if rand(rng)<p
                    rho2=Δ2/(r2+Tau2*Δ2)
                    gmInv-=rho2* ( gmInv*a2 .*  b2 )
                    detg*=abs2(1+Δ2/r2*Tau2)^(1/Nλ)

                    G02+=Δ2/r2* (G0t2[:,x] .* transpose( Gt02[x,:]))
                    Gt02+=Δ2/r2* (Gt2[:,x] .* transpose( Gt02[x,:]))
                    G0t2-=Δ2/r2* (G0t2[:,x] .* transpose( (I1-Gt2)[x,:]))
                    Gt2-=Δ2/r2* (Gt2[:,x] .* transpose( (I1-Gt2)[x,:])   )      
                    ss[2][x,lt]=sx2
                    #####################################################################
                    # print('-')
                    # Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt,div(model.Nt,2))
                    
                    # if norm(Gt2-Gt2_)+norm(G02-G02_)+norm(Gt02-Gt02_)+norm(G0t2-G0t2_)>1e-3
                    #     println('\n',norm(Gt2-Gt2_),'\n',norm(G02-G02_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t2-G0t2_))
                    #     error("$lt  $x :cmmccmmcm")
                    # end
                    #####################################################################
                end

            end

            ##------------------------------------------------------------------------
            tmpO+=detg
            counter+=1
            ##------------------------------------------------------------------------
        end

        for lt in model.Nt-1:-1:1
            if mod(lt,model.WrapTime)==0 || lt==div(model.Nt,2)
                Gt1,G01,Gt01,G0t1=G4(model,ss[1],lt,div(model.Nt,2))
                Gt2,G02,Gt02,G0t2=G4(model,ss[2],lt,div(model.Nt,2))
                
                GM=GroverMatrix(G01[index[:],index[:]],G02[index[:],index[:]])
                gmInv=inv(GM)
                detg=abs2(det(GM))^(1/Nλ)
            else
                D1=[model.η[x] for x in ss[1][:,lt+1]]
                D2=[model.η[x] for x in ss[2][:,lt+1]]
                Gt1=model.eKinv*diagm(exp.(-1im*model.α.*D1)) *Gt1* diagm(exp.(1im*model.α.*D1))*model.eK 
                Gt2=model.eKinv*diagm(exp.(-1im*model.α.*D2)) *Gt2* diagm(exp.(1im*model.α.*D2))*model.eK
                G0t1=G0t1*diagm(exp.(1im*model.α.*D1))*model.eK
                G0t2=G0t2*diagm(exp.(1im*model.α.*D2))*model.eK
                Gt01=model.eKinv*diagm(exp.(-1im*model.α.*D1))*Gt01
                Gt02=model.eKinv*diagm(exp.(-1im*model.α.*D2))*Gt02

                #####################################################################
                # Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt,div(model.Nt,2))
                # Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt,div(model.Nt,2))
                    
                # if norm(Gt1-Gt1_)+norm(Gt2-Gt2_)+norm(Gt01-Gt01_)+norm(Gt02-Gt02_)+norm(G0t1-G0t1_)+norm(G0t2-G0t2_)>1e-3
                #     println( norm(Gt1-Gt1_),'\n',norm(Gt2-Gt2_),'\n',norm(Gt01-Gt01_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t1-G0t1_),'\n',norm(G0t2-G0t2_) )
                #     error("$lt : WrapTime")
                # end
                #####################################################################
                end

            for x in 1:model.Ns
                b1=transpose(Gt01[x,index[:]]) *(2*G02[index[:],index[:]]-I2)*gmInv
                a1=G0t1[index[:],x]
                Tau1=b1*a1

                sp=Random.Sampler(rng,[i for i in elements if i != ss[1][x,lt]])
                sx1=rand(rng,sp)
                
                Δ1=exp(1im*model.α*(model.η[sx1]-model.η[ss[1][x,lt]]))-1
                r1=1+Δ1*(1-Gt1[x,x])

                p=model.γ[sx1]/model.γ[ss[1][x,lt]]*abs2(r1)^(1-λ)*abs2(r1+Δ1*Tau1)^(λ)

                if rand(rng)<p
                    rho1=Δ1/(r1+Tau1*Δ1)
                    gmInv-=rho1* ( gmInv*a1 .* b1)
                    detg*=abs2(1+Δ1/r1*Tau1)^(1/Nλ)

                    G01+=Δ1/r1* (G0t1[:,x] .* transpose(Gt01[x,:]))
                    Gt01+=Δ1/r1* (Gt1[:,x] .* transpose(Gt01[x,:]))
                    G0t1-=Δ1/r1* (G0t1[:,x] .* transpose( (I1-Gt1)[x,:] ) )
                    Gt1-=Δ1/r1* (Gt1[:,x] .* transpose( (I1-Gt1)[x,:]) )         
                    ss[1][x,lt]=sx1

                    #####################################################################
                    # print('-')
                    # Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt,div(model.Nt,2))
                    
                    # if norm(Gt1-Gt1_)+norm(G01-G01_)+norm(Gt01-Gt01_)+norm(G0t1-G0t1_)>1e-3
                    #     println('\n',norm(Gt1-Gt1_),'\n',norm(G01-G01_),'\n',norm(Gt01-Gt01_),'\n',norm(G0t1-G0t1_))
                    #     error("$lt  $x:,,,asdasdasd")
                    # end
                    #####################################################################
                end



                b2=transpose(Gt02[x,index[:]]) *gmInv
                a2=(2*G01[index[:],index[:]]-I2)*G0t2[index[:],x]
                Tau2=b2*a2

                sp=Random.Sampler(rng,[i for i in elements if i != ss[2][x,lt]])
                sx2=rand(rng,sp)

                Δ2=(exp(1im*model.α*(model.η[sx2]-model.η[ss[2][x,lt]]))-1)
                r2=(1+Δ2*(1-Gt2[x,x]))
                p=model.γ[sx2]/model.γ[ss[2][x,lt]]*(abs2(r2)^(1-λ))*(abs2(r2+Δ2*Tau2)^λ)

                if rand(rng)<p
                    rho2=Δ2/(r2+Tau2*Δ2)
                    gmInv-=rho2* ( gmInv*a2 .*  b2 )
                    detg*=abs2(1+Δ2/r2*Tau2)^(1/Nλ)

                    G02+=Δ2/r2* (G0t2[:,x] .* transpose( Gt02[x,:]))
                    Gt02+=Δ2/r2* (Gt2[:,x] .* transpose( Gt02[x,:]))
                    G0t2-=Δ2/r2* (G0t2[:,x] .* transpose( (I1-Gt2)[x,:]))
                    Gt2-=Δ2/r2* (Gt2[:,x] .* transpose( (I1-Gt2)[x,:])   )      
                    ss[2][x,lt]=sx2
                    #####################################################################
                    # print('-')
                    # Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt,div(model.Nt,2))
                    
                    # if norm(Gt2-Gt2_)+norm(G02-G02_)+norm(Gt02-Gt02_)+norm(G0t2-G0t2_)>1e-3
                    #     println('\n',norm(Gt2-Gt2_),'\n',norm(G02-G02_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t2-G0t2_))
                    #     error("$lt  $x :cmmccmmcm")
                    # end
                    #####################################################################
                end

            end

            ##------------------------------------------------------------------------
            tmpO+=detg
            counter+=1
            ##------------------------------------------------------------------------
        end

        O[loop+1]=tmpO/counter
        tmpO=counter=0

    end
    return ss
end


function EE_dir(path::String,model::_Hubbard_Para,index_arr::Vector{Vector{Int64}},WarmSweeps::Int64,Sweeps::Int64,ss::Vector{Matrix{UInt8}})::Vector{Matrix{UInt8}}
    if model.Lattice=="SQUARE"
        name="□"
    elseif model.Lattice=="HoneyComb60"
        name="HC60"
    elseif model.Lattice=="HoneyComb120"
        name="HC120"
    end
    fid1 = open("$(path)Phy$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)BS$(model.BatchSize).csv", "a")
    fid2 = open("$(path)EEdir$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)BS$(model.BatchSize).csv", "a")
    rng=MersenneTwister(floor(Int,time()))
    elements=(1, 2, 3, 4)

    G1=zeros(ComplexF64,model.Ns,model.Ns)
    G2=zeros(ComplexF64,model.Ns,model.Ns)
    EE=zeros(Float64,1,length(index_arr))
    mA=mB=nn=R0=R1=Ek=0
    counter=0

    for loop in 1:Sweeps+WarmSweeps
        for lt in 1:model.Nt
            if mod(lt,model.WrapTime)==1
                G1=Gτ(model,ss[1],lt)
                G2=Gτ(model,ss[2],lt)
            else
                D1=[model.η[x] for x in ss[1][:,lt]]
                D2=[model.η[x] for x in ss[2][:,lt]]
                G1=diagm(exp.(1im*model.α.*D1))*model.eK *G1* model.eKinv*diagm(exp.(-1im*model.α.*D1))
                G2=diagm(exp.(1im*model.α.*D2))*model.eK *G2* model.eKinv*diagm(exp.(-1im*model.α.*D2))
                
                #####################################################################
                # if norm(G1-Gτ(model,ss[1],lt))+norm(G2-Gτ(model,ss[2],lt))>1e-2 
                #         error("WrapTime")
                #     end
                #####################################################################
            end

            for x in 1:model.Ns
                sp=Random.Sampler(rng,[i for i in elements if i != ss[1][x,lt]])
                sx=rand(rng,sp)
                Δ=exp(1im*model.α*(model.η[sx]-model.η[ss[1][x,lt]]))-1
                r=1+Δ*(1-G1[x,x])

                if rand(rng)<model.γ[sx]/model.γ[ss[1][x,lt]]*abs2(r)
                    G1=G1-Δ/r.*(  G1[:,x]    .*  transpose((I(model.Ns)-G1)[x,:])   )
                    ss[1][x,lt]=sx
                    #####################################################################
                    # if norm(G1-Gτ(model,ss[1],lt))>1e-2
                    #     error("$lt :G1")
                    # end
                    #####################################################################

                end

                sp=Random.Sampler(rng,[i for i in elements if i != ss[2][x,lt]])
                sx=rand(rng,sp)
                Δ=exp(1im*model.α*(model.η[sx]-model.η[ss[2][x,lt]]))-1
                r=1+Δ*(1-G2[x,x])

                if rand(rng)<model.γ[sx]/model.γ[ss[2][x,lt]]*abs2(r)
                    G2=G2-Δ/r.*(  G2[:,x]    .*  transpose((I(model.Ns)-G2)[x,:])   )
                    ss[2][x,lt]=sx
                    #####################################################################
                    # if norm(G2-Gτ(model,ss[2],lt))>1e-2
                    #     error("G2")
                    # end
                    #####################################################################

                end

            end

            if loop>WarmSweeps && abs(lt-model.Nt/2)<=model.WrapTime
                G01=G1[:,:]
                G02=G2[:,:]
                if lt>model.Nt/2
                    for i in lt:-1:div(model.Nt,2)+1
                        D1=[model.η[x] for x in ss[1][:,i]]
                        D2=[model.η[x] for x in ss[2][:,i]]
                        G01= model.eKinv*diagm(exp.(-1im*model.α.*D1)) *G01*  diagm(exp.(1im*model.α.*D1))*model.eK
                        G02= model.eKinv*diagm(exp.(-1im*model.α.*D2)) *G02*  diagm(exp.(1im*model.α.*D2))*model.eK
                    end
                else
                    for i in lt+1:div(model.Nt,2)
                        D1=[model.η[x] for x in ss[1][:,i]]
                        D2=[model.η[x] for x in ss[2][:,i]]
                        G01=diagm(exp.(1im*model.α.*D1))*model.eK *G01* model.eKinv*diagm(exp.(-1im*model.α.*D1))  
                        G02=diagm(exp.(1im*model.α.*D2))*model.eK *G02* model.eKinv*diagm(exp.(-1im*model.α.*D2))  
                    end
                end
                #####################################################################
                # if norm(G01-Gτ(model,ss[1],div(model.Nt,2)))+norm(G02-Gτ(model,ss[2],div(model.Nt,2)))>1e-2
                #     println(norm(G01-Gτ(model,ss[1],div(model.Nt,2))),'\n',norm(G02-Gτ(model,ss[2],div(model.Nt,2))))
                #     error("$lt :G0")
                # end
                #####################################################################

                # symmetric Trotter Decomposition
                G01=model.HalfeK* G01 *model.HalfeKinv
                G02=model.HalfeK* G02 *model.HalfeKinv
            
                ##------------------------------------------------------------------------
                tmp=Magnetism(model,G01)
                mA+=tmp[1]
                mB+=tmp[2]
                tmp=CzzofSpin(model,G01)
                R0+=tmp[1]
                R1+=tmp[2]
                nn+=NN(model,G01)
                Ek+=EK(model,G01)
                counter+=1

                tmp=Magnetism(model,G02)
                mA+=tmp[1]
                mB+=tmp[2]
                tmp=CzzofSpin(model,G02)
                R0+=tmp[1]
                R1+=tmp[2]
                nn+=NN(model,G02)
                Ek+=EK(model,G02)
                counter+=1

                for i in 1:length(index_arr)
                    EE[i]+=abs2( det(GroverMatrix(G01[index_arr[i][:],index_arr[i][:]],G02[index_arr[i][:],index_arr[i][:]])) )
                end
                ##------------------------------------------------------------------------
            end
        end

        for lt in model.Nt-1:-1:1
            if mod(lt,model.WrapTime)==1
                G1=Gτ(model,ss[1],lt)
                G2=Gτ(model,ss[2],lt)
            else
                D1=[model.η[x] for x in ss[1][:,lt+1]]
                D2=[model.η[x] for x in ss[2][:,lt+1]]
                G1=model.eKinv*diagm(exp.(-1im*model.α.*D1)) *G1* diagm(exp.(1im*model.α.*D1))*model.eK 
                G2=model.eKinv*diagm(exp.(-1im*model.α.*D2)) *G2* diagm(exp.(1im*model.α.*D2))*model.eK 
                    
                #####################################################################
                # if norm(G1-Gτ(model,ss[1],lt))+norm(G2-Gτ(model,ss[2],lt))>1e-2 
                #     error("WrapTime")
                # end
                #####################################################################
            end

            for x in 1:model.Ns
                sp=Random.Sampler(rng,[i for i in elements if i != ss[1][x,lt]])
                sx=rand(rng,sp)
                Δ=exp(1im*model.α*(model.η[sx]-model.η[ss[1][x,lt]]))-1
                r=1+Δ*(1-G1[x,x])

                if rand(rng)<model.γ[sx]/model.γ[ss[1][x,lt]]*abs2(r)
                    G1=G1-Δ/r.*(  G1[:,x]    .*  transpose((I(model.Ns)-G1)[x,:])   )
                    ss[1][x,lt]=sx
                    #####################################################################
                    # if norm(G1-Gτ(model,ss[1],lt))>1e-2
                    #     error("$lt :G1")
                    # end
                    #####################################################################

                end

                sp=Random.Sampler(rng,[i for i in elements if i != ss[2][x,lt]])
                sx=rand(rng,sp)
                Δ=exp(1im*model.α*(model.η[sx]-model.η[ss[2][x,lt]]))-1
                r=1+Δ*(1-G2[x,x])

                if rand(rng)<model.γ[sx]/model.γ[ss[2][x,lt]]*abs2(r)
                    G2=G2-Δ/r.*(  G2[:,x]    .*  transpose((I(model.Ns)-G2)[x,:])   )
                    ss[2][x,lt]=sx
                    #####################################################################
                    # if norm(G2-Gτ(model,ss[2],lt))>1e-2
                    #     error("G2")
                    # end
                    #####################################################################

                end

            end

            if loop>WarmSweeps && abs(lt-model.Nt/2)<=model.WrapTime
                G01=G1[:,:]
                G02=G2[:,:]
                if lt>model.Nt/2
                    for i in lt:-1:div(model.Nt,2)+1
                        D1=[model.η[x] for x in ss[1][:,i]]
                        D2=[model.η[x] for x in ss[2][:,i]]
                        G01= model.eKinv*diagm(exp.(-1im*model.α.*D1)) *G01*  diagm(exp.(1im*model.α.*D1))*model.eK
                        G02= model.eKinv*diagm(exp.(-1im*model.α.*D2)) *G02*  diagm(exp.(1im*model.α.*D2))*model.eK
                    end
                else
                    for i in lt+1:div(model.Nt,2)
                        D1=[model.η[x] for x in ss[1][:,i]]
                        D2=[model.η[x] for x in ss[2][:,i]]
                        G01=diagm(exp.(1im*model.α.*D1))*model.eK *G01* model.eKinv*diagm(exp.(-1im*model.α.*D1))  
                        G02=diagm(exp.(1im*model.α.*D2))*model.eK *G02* model.eKinv*diagm(exp.(-1im*model.α.*D2))  
                    end
                end
                #####################################################################
                # if norm(G01-Gτ(model,ss[1],div(model.Nt,2)))+norm(G02-Gτ(model,ss[2],div(model.Nt,2)))>1e-2
                #     println(norm(G01-Gτ(model,ss[1],div(model.Nt,2))),'\n',norm(G02-Gτ(model,ss[2],div(model.Nt,2))))
                #     error("$lt :G0")
                # end
                #####################################################################

                # symmetric Trotter Decomposition
                G01=model.HalfeK* G01 *model.HalfeKinv
                G02=model.HalfeK* G02 *model.HalfeKinv
            
                ##------------------------------------------------------------------------
                tmp=Magnetism(model,G01)
                mA+=tmp[1]
                mB+=tmp[2]
                tmp=CzzofSpin(model,G01)
                R0+=tmp[1]
                R1+=tmp[2]
                nn+=NN(model,G01)
                Ek+=EK(model,G01)
                counter+=1

                tmp=Magnetism(model,G02)
                mA+=tmp[1]
                mB+=tmp[2]
                tmp=CzzofSpin(model,G02)
                R0+=tmp[1]
                R1+=tmp[2]
                nn+=NN(model,G02)
                Ek+=EK(model,G02)
                counter+=1

                for i in 1:length(index_arr)
                    EE[i]+=abs2( det(GroverMatrix(G01[index_arr[i][:],index_arr[i][:]],G02[index_arr[i][:],index_arr[i][:]])) )
                end
                ##------------------------------------------------------------------------
            end
        end
        if loop>WarmSweeps
            writedlm(fid1,[Ek model.U*nn nn mA mB R0 R1]/counter,',')
            writedlm(fid2,EE/counter*2,',')
            mA=mB=nn=R0=R1=Ek=0
            EE=zeros(Float64,1,length(index_arr))
            counter=0
        end

    end
    close(fid1)
    close(fid2)
    return ss
end





function ctrl_EEicr_old(path::String,model::_Hubbard_Para,index::Vector{Int64},Sweeps::Int64,λ::Float64,Nλ::Int64,ss::Vector{Matrix{UInt8}},record::Bool)::Vector{Matrix{UInt8}}
    if model.Lattice=="SQUARE"
        name="□"
    elseif model.Lattice=="HoneyComb60"
        name="HC60"
    elseif model.Lattice=="HoneyComb120"
        name="HC120"
    end

    file="$(path)EEicr$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)N$(Nλ)BS$(model.BatchSize).csv"
    atexit() do
        if record
            open(file, "a") do io
                lock(io)
                writedlm(io, O', ',')
                unlock(io)
            end
        end
        writedlm("$(path)ss/SS$(name)_t$(model.t)U$(model.U)size$(model.site)Δt$(model.Δt)Θ$(model.Θ)λ$(Int(round(Nλ*λ))).csv", [ss[1] ss[2]],",")
    end

    rng=MersenneTwister(Threads.threadid()+round(Int,time()*1000))
    elements=(1, 2, 3, 4)

    Gt1=zeros(ComplexF64,model.Ns,model.Ns)
    Gt2=zeros(ComplexF64,model.Ns,model.Ns)
    G01=zeros(ComplexF64,model.Ns,model.Ns)
    G02=zeros(ComplexF64,model.Ns,model.Ns)
    Gt01=zeros(ComplexF64,model.Ns,model.Ns)
    Gt02=zeros(ComplexF64,model.Ns,model.Ns)
    G0t1=zeros(ComplexF64,model.Ns,model.Ns)
    G0t2=zeros(ComplexF64,model.Ns,model.Ns)
    gmInv=zeros(ComplexF64,length(index),length(index))
    detg=0

    tmpO=0
    counter=0
    O=zeros(Sweeps+1)
    O[1]=λ

    I1=I(model.Ns)
    I2=I(length(index))

    
    for loop in 1:Sweeps
        for lt in 1:model.Nt
            if mod(lt,model.WrapTime)==1 || lt==div(model.Nt,2)+1
                Gt1,G01,Gt01,G0t1=G4(model,ss[1],lt,div(model.Nt,2))
                Gt2,G02,Gt02,G0t2=G4(model,ss[2],lt,div(model.Nt,2))
                
                GM=GroverMatrix(G01[index[:],index[:]],G02[index[:],index[:]])
                gmInv=inv(GM)
                detg=abs2(det(GM))^(1/Nλ)
            else
                D1=[model.η[x] for x in ss[1][:,lt]]
                D2=[model.η[x] for x in ss[2][:,lt]]
                Gt1=diagm(exp.(1im*model.α.*D1))*model.eK *Gt1* model.eKinv*diagm(exp.(-1im*model.α.*D1))
                Gt2=diagm(exp.(1im*model.α.*D2))*model.eK *Gt2* model.eKinv*diagm(exp.(-1im*model.α.*D2))
                G0t1=G0t1*model.eKinv*diagm(exp.(-1im*model.α.*D1))
                G0t2=G0t2*model.eKinv*diagm(exp.(-1im*model.α.*D2))
                Gt01=diagm(exp.(1im*model.α.*D1))*model.eK*Gt01
                Gt02=diagm(exp.(1im*model.α.*D2))*model.eK*Gt02

                #####################################################################
                # Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt,div(model.Nt,2))
                # Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt,div(model.Nt,2))
                    
                # if norm(Gt1-Gt1_)+norm(Gt2-Gt2_)+norm(Gt01-Gt01_)+norm(Gt02-Gt02_)+norm(G0t1-G0t1_)+norm(G0t2-G0t2_)>1e-3
                #     println( norm(Gt1-Gt1_),'\n',norm(Gt2-Gt2_),'\n',norm(Gt01-Gt01_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t1-G0t1_),'\n',norm(G0t2-G0t2_) )
                #     error("$lt : WrapTime")
                # end
                #####################################################################


            end

            for x in 1:model.Ns
                b1=transpose(Gt01[x,index[:]]) *(2*G02[index[:],index[:]]-I2)*gmInv
                a1=G0t1[index[:],x]
                Tau1=b1*a1

                sp=Random.Sampler(rng,[i for i in elements if i != ss[1][x,lt]])
                sx1=rand(rng,sp)
                
                Δ1=exp(1im*model.α*(model.η[sx1]-model.η[ss[1][x,lt]]))-1
                r1=1+Δ1*(1-Gt1[x,x])

                p=model.γ[sx1]/model.γ[ss[1][x,lt]]*abs2(r1)^(1-λ)*abs2(r1+Δ1*Tau1)^(λ)

                if rand(rng)<p
                    rho1=Δ1/(r1+Tau1*Δ1)
                    gmInv-=rho1* ( gmInv*a1 .* b1)
                    detg*=abs2(1+Δ1/r1*Tau1)^(1/Nλ)

                    G01+=Δ1/r1* (G0t1[:,x] .* transpose(Gt01[x,:]))
                    Gt01+=Δ1/r1* (Gt1[:,x] .* transpose(Gt01[x,:]))
                    G0t1-=Δ1/r1* (G0t1[:,x] .* transpose( (I1-Gt1)[x,:] ) )
                    Gt1-=Δ1/r1* (Gt1[:,x] .* transpose( (I1-Gt1)[x,:]) )         
                    ss[1][x,lt]=sx1

                    #####################################################################
                    # print('-')
                    # Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt,div(model.Nt,2))
                    
                    # if norm(Gt1-Gt1_)+norm(G01-G01_)+norm(Gt01-Gt01_)+norm(G0t1-G0t1_)>1e-3
                    #     println('\n',norm(Gt1-Gt1_),'\n',norm(G01-G01_),'\n',norm(Gt01-Gt01_),'\n',norm(G0t1-G0t1_))
                    #     error("$lt  $x:,,,asdasdasd")
                    # end
                    #####################################################################
                end



                b2=transpose(Gt02[x,index[:]]) *gmInv
                a2=(2*G01[index[:],index[:]]-I2)*G0t2[index[:],x]
                Tau2=b2*a2

                sp=Random.Sampler(rng,[i for i in elements if i != ss[2][x,lt]])
                sx2=rand(rng,sp)

                Δ2=(exp(1im*model.α*(model.η[sx2]-model.η[ss[2][x,lt]]))-1)
                r2=(1+Δ2*(1-Gt2[x,x]))
                p=model.γ[sx2]/model.γ[ss[2][x,lt]]*(abs2(r2)^(1-λ))*(abs2(r2+Δ2*Tau2)^λ)

                if rand(rng)<p
                    rho2=Δ2/(r2+Tau2*Δ2)
                    gmInv-=rho2* ( gmInv*a2 .*  b2 )
                    detg*=abs2(1+Δ2/r2*Tau2)^(1/Nλ)

                    G02+=Δ2/r2* (G0t2[:,x] .* transpose( Gt02[x,:]))
                    Gt02+=Δ2/r2* (Gt2[:,x] .* transpose( Gt02[x,:]))
                    G0t2-=Δ2/r2* (G0t2[:,x] .* transpose( (I1-Gt2)[x,:]))
                    Gt2-=Δ2/r2* (Gt2[:,x] .* transpose( (I1-Gt2)[x,:])   )      
                    ss[2][x,lt]=sx2
                    #####################################################################
                    # print('-')
                    # Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt,div(model.Nt,2))
                    
                    # if norm(Gt2-Gt2_)+norm(G02-G02_)+norm(Gt02-Gt02_)+norm(G0t2-G0t2_)>1e-3
                    #     println('\n',norm(Gt2-Gt2_),'\n',norm(G02-G02_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t2-G0t2_))
                    #     error("$lt  $x :cmmccmmcm")
                    # end
                    #####################################################################
                end

            end

            ##------------------------------------------------------------------------
            tmpO+=detg
            counter+=1
            ##------------------------------------------------------------------------
        end

        for lt in model.Nt-1:-1:1
            if mod(lt,model.WrapTime)==0 || lt==div(model.Nt,2)
                Gt1,G01,Gt01,G0t1=G4(model,ss[1],lt,div(model.Nt,2))
                Gt2,G02,Gt02,G0t2=G4(model,ss[2],lt,div(model.Nt,2))
                
                GM=GroverMatrix(G01[index[:],index[:]],G02[index[:],index[:]])
                gmInv=inv(GM)
                detg=abs2(det(GM))^(1/Nλ)
            else
                D1=[model.η[x] for x in ss[1][:,lt+1]]
                D2=[model.η[x] for x in ss[2][:,lt+1]]
                Gt1=model.eKinv*diagm(exp.(-1im*model.α.*D1)) *Gt1* diagm(exp.(1im*model.α.*D1))*model.eK 
                Gt2=model.eKinv*diagm(exp.(-1im*model.α.*D2)) *Gt2* diagm(exp.(1im*model.α.*D2))*model.eK
                G0t1=G0t1*diagm(exp.(1im*model.α.*D1))*model.eK
                G0t2=G0t2*diagm(exp.(1im*model.α.*D2))*model.eK
                Gt01=model.eKinv*diagm(exp.(-1im*model.α.*D1))*Gt01
                Gt02=model.eKinv*diagm(exp.(-1im*model.α.*D2))*Gt02

                #####################################################################
                # Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt,div(model.Nt,2))
                # Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt,div(model.Nt,2))
                    
                # if norm(Gt1-Gt1_)+norm(Gt2-Gt2_)+norm(Gt01-Gt01_)+norm(Gt02-Gt02_)+norm(G0t1-G0t1_)+norm(G0t2-G0t2_)>1e-3
                #     println( norm(Gt1-Gt1_),'\n',norm(Gt2-Gt2_),'\n',norm(Gt01-Gt01_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t1-G0t1_),'\n',norm(G0t2-G0t2_) )
                #     error("$lt : WrapTime")
                # end
                #####################################################################
                end

            for x in 1:model.Ns
                b1=transpose(Gt01[x,index[:]]) *(2*G02[index[:],index[:]]-I2)*gmInv
                a1=G0t1[index[:],x]
                Tau1=b1*a1

                sp=Random.Sampler(rng,[i for i in elements if i != ss[1][x,lt]])
                sx1=rand(rng,sp)
                
                Δ1=exp(1im*model.α*(model.η[sx1]-model.η[ss[1][x,lt]]))-1
                r1=1+Δ1*(1-Gt1[x,x])

                p=model.γ[sx1]/model.γ[ss[1][x,lt]]*abs2(r1)^(1-λ)*abs2(r1+Δ1*Tau1)^(λ)

                if rand(rng)<p
                    rho1=Δ1/(r1+Tau1*Δ1)
                    gmInv-=rho1* ( gmInv*a1 .* b1)
                    detg*=abs2(1+Δ1/r1*Tau1)^(1/Nλ)

                    G01+=Δ1/r1* (G0t1[:,x] .* transpose(Gt01[x,:]))
                    Gt01+=Δ1/r1* (Gt1[:,x] .* transpose(Gt01[x,:]))
                    G0t1-=Δ1/r1* (G0t1[:,x] .* transpose( (I1-Gt1)[x,:] ) )
                    Gt1-=Δ1/r1* (Gt1[:,x] .* transpose( (I1-Gt1)[x,:]) )         
                    ss[1][x,lt]=sx1

                    #####################################################################
                    # print('-')
                    # Gt1_,G01_,Gt01_,G0t1_=G4(model,ss[1],lt,div(model.Nt,2))
                    
                    # if norm(Gt1-Gt1_)+norm(G01-G01_)+norm(Gt01-Gt01_)+norm(G0t1-G0t1_)>1e-3
                    #     println('\n',norm(Gt1-Gt1_),'\n',norm(G01-G01_),'\n',norm(Gt01-Gt01_),'\n',norm(G0t1-G0t1_))
                    #     error("$lt  $x:,,,asdasdasd")
                    # end
                    #####################################################################
                end



                b2=transpose(Gt02[x,index[:]]) *gmInv
                a2=(2*G01[index[:],index[:]]-I2)*G0t2[index[:],x]
                Tau2=b2*a2

                sp=Random.Sampler(rng,[i for i in elements if i != ss[2][x,lt]])
                sx2=rand(rng,sp)

                Δ2=(exp(1im*model.α*(model.η[sx2]-model.η[ss[2][x,lt]]))-1)
                r2=(1+Δ2*(1-Gt2[x,x]))
                p=model.γ[sx2]/model.γ[ss[2][x,lt]]*(abs2(r2)^(1-λ))*(abs2(r2+Δ2*Tau2)^λ)

                if rand(rng)<p
                    rho2=Δ2/(r2+Tau2*Δ2)
                    gmInv-=rho2* ( gmInv*a2 .*  b2 )
                    detg*=abs2(1+Δ2/r2*Tau2)^(1/Nλ)

                    G02+=Δ2/r2* (G0t2[:,x] .* transpose( Gt02[x,:]))
                    Gt02+=Δ2/r2* (Gt2[:,x] .* transpose( Gt02[x,:]))
                    G0t2-=Δ2/r2* (G0t2[:,x] .* transpose( (I1-Gt2)[x,:]))
                    Gt2-=Δ2/r2* (Gt2[:,x] .* transpose( (I1-Gt2)[x,:])   )      
                    ss[2][x,lt]=sx2
                    #####################################################################
                    # print('-')
                    # Gt2_,G02_,Gt02_,G0t2_=G4(model,ss[2],lt,div(model.Nt,2))
                    
                    # if norm(Gt2-Gt2_)+norm(G02-G02_)+norm(Gt02-Gt02_)+norm(G0t2-G0t2_)>1e-3
                    #     println('\n',norm(Gt2-Gt2_),'\n',norm(G02-G02_),'\n',norm(Gt02-Gt02_),'\n',norm(G0t2-G0t2_))
                    #     error("$lt  $x :cmmccmmcm")
                    # end
                    #####################################################################
                end

            end

            ##------------------------------------------------------------------------
            tmpO+=detg
            counter+=1
            ##------------------------------------------------------------------------
        end

        O[loop+1]=tmpO/counter
        tmpO=counter=0

    end
    return ss
end



