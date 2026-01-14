function EK(model,G)
    return (-2*model.t*sum(model.K.*real(G)))/model.Ns
end

function NN(model,G)
    nn=0
    for i in 1:model.Ns
        nn+=(1-G[i,i])*adjoint(G[i,i])
    end
    if imag(nn)>1e-4
        error("complex run into physics@nn!")
    end
    return real(nn)/model.Ns
end


function Magnetism(model,G)
    mA=0
    mB=0
    for i in 1:model.Ns
        if mod(i,2)==0
            mA+=1-2*real(G[i,i])
        else
            mB+=1-2*real(G[i,i])
        end
    end
    return mA/model.Ns*2,mB/model.Ns*2
end




function CzzofSpin(model,G)
    R1,R0=0,0

    if model.Lattice=="SQUARE"
        if length(model.site)==1
            L=div(model.site[1],2)
            C=zeros(ComplexF64,L+1)
            ii=1
            for l in 0:L
                jj=mod1(i+l,model.Ns)

                δ=Int(ii==jj)
                C[l+1]+=(1-G[ii,ii])*(1-G[jj,jj])+(δ-G[jj,ii])*G[ii,jj]
                C[l+1]+=adjoint(G[ii,ii])*adjoint(G[jj,jj])+adjoint(G[jj,ii])*(δ-adjoint(G[ii,jj]))
                C[l+1]-=(1-G[ii,ii])*adjoint(G[jj,jj])
                C[l+1]-=adjoint(G[ii,ii])*(1-G[jj,jj])

                R0+=C[l+1]*(-1)^l
                R1+=C[l+1]*(-1)^l*cos(2π/L/2*l)
            end

            if norm(imag(C))>1e-4
                error("complex run into physics@CC!")
            end
            C=real.(C)
            return real(R0),real(R1),collect(0:div(model.site[1],2)),C
        else
            Lx=model.site[1]
            Ly=model.site[2]
            C=zeros(Float64,Lx,Ly)
            for ix in 1:Lx
                for iy in 1:Ly
                    i=xy_i(model.Lattice,model.site,ix,iy)
                    for lx in 0:Lx-1
                        for ly in 0:Ly-1
                            j=xy_i(model.Lattice,model.site,mod1(ix+lx,Lx),mod1(iy+ly,Ly))
                            δ=Int(i==j)
                            C[lx+1,ly+1]+=(-1)^(ix+iy+lx+ly) *2*real(   G[i,i]*G[j,j]-G[i,j]*G[j,i]+G[i,i]*adjoint(G[j,j])+δ*G[i,j]-G[i,i]-G[j,j]+1/2     )
                        end
                    end
                end
            end
            if norm(imag(C))>1e-4
                error("complex run into physics@CC!")
            end

            C=C/model.Ns^2
            for i in 0:Lx-1
                for j in 0:Ly-1
                    R0+=C[i+1,j+1]
                    R1+=C[i+1,j+1]*cos(2π*( i/Lx + j/Ly ))
                end
            end

            return real(R0),real(R1),C[1,1],C[div(Lx,2),div(Ly,2)]

        end

    elseif  occursin("HoneyComb", model.Lattice)
        Lx=model.site[1]
        Ly=model.site[2]
        C=zeros(Float64,Lx,Ly)

        for ix in 1:Lx
            for iy in 1:Ly
                # ix=iy=1
                i=xy_i(model.Lattice,model.site,ix,iy)-1
                for lx in 0:Lx-1
                    for ly in 0:Ly-1
                        j=xy_i(model.Lattice,model.site,mod1(ix+lx,Lx),mod1(iy+ly,Ly))-1

                        δ=Int(i==j)
                        C[lx+1,ly+1]+=2*real(   G[i,i]*G[j,j]-G[i,j]*G[j,i]+G[i,i]*adjoint(G[j,j])+δ*G[i,j]-G[i,i]-G[j,j]+1/2     )

                        i=i+1
                        δ=Int(i==j)
                        C[lx+1,ly+1]-=2*real(   G[i,i]*G[j,j]-G[i,j]*G[j,i]+G[i,i]*adjoint(G[j,j])+δ*G[i,j]-G[i,i]-G[j,j]+1/2     )

                        j=j+1
                        δ=Int(i==j)
                        C[lx+1,ly+1]+=2*real(   G[i,i]*G[j,j]-G[i,j]*G[j,i]+G[i,i]*adjoint(G[j,j])+δ*G[i,j]-G[i,i]-G[j,j]+1/2     )

                        i=i-1
                        δ=Int(i==j)
                        C[lx+1,ly+1]-=2*real(   G[i,i]*G[j,j]-G[i,j]*G[j,i]+G[i,i]*adjoint(G[j,j])+δ*G[i,j]-G[i,i]-G[j,j]+1/2     )
                    end
                end
            end
        end

        C=C/model.Ns^2
        for i in 0:Lx-1
            for j in 0:Ly-1
                R0+=C[i+1,j+1]
                R1+=C[i+1,j+1]*cos(2π*( i/Lx + j/Ly ))
            end
        end

        # index=sortperm(d)
        # d=d[index]
        # Cd=Cd[index]
        
        # dd=unique(d)
        # Cdd=zeros(Float64,length(dd))
        
        # for i in eachindex(dd)
        #     index=findall(d.==dd[i])
        #     Cdd[i]=mean(Cd[index])
        # end

        return R0,R1,C[1,1],C[div(Lx,2),div(Ly,2)]
    end
end

