# !/bin/bash
current_path=$(pwd)

t=1
U=7
dt=0.03
BS=10
Sweeps=80
Pt=V

L=(6 )
Theta=(3.6 3.3 1.8 1.5)
# Theta=(2.1 2.4 2.7 3.0 3.3 3.6)
# Theta=(0.9 1.2 1.5 1.8 2.1 2.4 2.7 3.0)

for (( l=0; l<${#L[@]}  ; l++ ));  do  
    # 计算 L^1.7 使用自然对数和指数函数：e^(1.7 * ln(L))
    exponent=$(echo "scale=10; 1.7 * l(${L[$l]})" | bc -l)  # ln(L) 是自然对数
    L_power_1_7=$(echo "scale=10; e($exponent)" | bc -l)  # e^exponent
    # 计算 0.2 * L^1.7
    N=$(echo "scale=3; 0.2 * $L_power_1_7+0.9999" | bc)
    N=$(printf "%.0f" "$N")
    echo "L=${L[$l]} :Nlambda is" $N

    for theta in "${Theta[@]}"; do
        for (( i=0; i<N  ; i++ )); do 
            ld=$(echo "scale=5; $i / $N" | bc)  
            sbatch  --cpus-per-task=1 --job-name=SCY${Pt}-U${U}_${L[$l]}_${theta} /home/zxli_1/KAP/tUdata/HC/SCEEjob.sh $current_path $t $U [${L[$l]},${L[$l]}] $theta $dt $BS $Sweeps $N $ld $Pt
        done
    done 
done

