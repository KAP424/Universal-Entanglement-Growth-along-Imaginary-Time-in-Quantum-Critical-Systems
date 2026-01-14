import errno
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import csv
from scipy.stats import sem
from scipy.optimize import curve_fit

def linear_fit(x,y,dy):
    A=np.ones([len(x),2]) 
    A[:,0]=x
    
    para=np.linalg.inv(A.conj().T@A)@A.conj().T@y

    if dy is False:
        yy=para[0] * x + para[1]
        dslope=np.sqrt( np.sum((yy-y)**2)/(len(x)-2)/np.sum((x-np.mean(x))**2) )
    else:
        
        xbar=np.sum(x/dy**2)/np.sum(1/dy**2)
        dslope=1/np.sqrt(  np.sum((x - xbar)**2 / dy**2)  )

    return para, dslope 

def powerfit(x,y):
    initial_guess = [0.3, 1.0, 1.0]

    def model_func(x, a, b, c):
        return a + b * x**(c)
    
    params, _ = curve_fit(
        model_func, x, y,
        p0=initial_guess,
        # bounds=None,
        maxfev=100000
    )
    return params

def sweep_stable(period, file):
    filename = os.path.basename(file)              # 'DP_Lib.py'
    params = extract_numbers(filename)
    if params[0]==120 or params[0]==60:
        U=params[2]
        L=(int(params[3]))
        if params[3]==params[4]:
            t=(params[6])
        else:
            t=(params[5])
    else:
        U=params[1]
        L=(int(params[2]))
        if params[2]==params[3]:
            t=(params[5])
        else:
            t=(params[4])

    data = []
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
            
    data=combine_data(data)
    vals = data[:,1:].astype(float)

    N = data.shape[1] // period
    
    S2 = np.zeros(N)
    dS = np.zeros(N)

    for i in range(N):
        subEE = np.mean(vals[:, 0:(i+1)*period], axis=1)
        dsubEE = np.std(vals[:, 0:(i+1)*period], axis=1, ddof=1) / np.sqrt((i+1)*period)
        S2[i] = -np.log(np.prod(subEE))
        dS[i] = np.sqrt(np.sum((dsubEE/subEE)**2))

    fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=200)
    ax[0].errorbar(range(1, N+1), S2, yerr=dS, fmt='o-', markersize=2, color='blue', ecolor='black', capsize=5,label=f'L={L}')
    ax[0].set_xlabel(f'Sweeps (unit: {period})')
    ax[0].set_ylabel(r'$\Delta S_2$')
    ax[0].set_title(f'U={U}, L={L}, θ={t}, N={params[-2]:.0f}')

    text = f'Final: {S2[-1]:.4f} ± {dS[-1]:.4f}'
    ax[0].text(0.98, 0.02, text,
            transform=ax[0].transAxes,  # 轴坐标(0~1)
            ha='right', va='bottom',
            fontsize=8,
            bbox=dict(boxstyle='round', fc='w', alpha=0.7))

    ax[1].plot(range(1, N+1), dS, 'o-', markersize=2, color='red', label='Error')
    ax[1].set_xlabel(f'Sweeps (unit: {period})')
    ax[1].set_ylabel('Error bar')
    ax[1].set_title(f'U={U}, L={L}, θ={t}, N={params[-2]:.0f}')

    # fig.savefig(os.path.join("C:/Users/admin/Desktop/",f"U={U}L={L}θ={t}N={params[-2]}.png"))
    return fig

def File_Process1(folder_path,throw=0):
    Data = []

    for file in os.listdir(folder_path):
        if not file.endswith(".csv") or "EE" not in file:
            continue
        print(file)
        params = extract_numbers(file)
        if params[0]==120 or params[0]==60:
            U=params[2]
            L=(int(params[3]))
            if params[3]==params[4]:
                t=(params[6])
            else:
                t=(params[5])
        else:
            U=params[1]
            L=(int(params[2]))
            if params[2]==params[3]:
                t=(params[5])
            else:
                t=(params[4])

        
        EE,dEE=File_Process2(file,folder_path,throw)
        Data.append([t, L, EE , dEE])

    Data = pd.DataFrame(Data, columns=['t', 'L', 'EE', 'dEE'])
    return U,Data

def File_Process2(file,folder_path,throw=0):
    full_path = os.path.join(folder_path, file)
    params = extract_numbers(file)
    if params[0]==120 or params[0]==60:
        U=params[2]
        L=(int(params[3]))
        if params[3]==params[4]:
            t=(params[6])
        else:
            t=(params[5])
    else:
        U=params[1]
        L=(int(params[2]))
        if params[2]==params[3]:
            t=(params[5])
        else:
            t=(params[4])

    data = []
    with open(full_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
            
    data=combine_data(data,throw)

    λ = data[:,0]
    # 确保为浮点数并计算样本数 n（data 的第一列是标签）
    vals = data[:,1:].astype(float)
    subEE = np.mean(vals, axis=1)
    n = vals.shape[1]
    # 对样本数太小的情况做保护（n<=1 时无法计算样本标准差）
    if n <= 1:
        # 无法计算 SEM，返回 NaN 向量以便上层捕获或忽略
        dsubEE = np.full(subEE.shape, np.nan)
    else:
        # 使用样本标准差（ddof=1），标准误 = s / sqrt(n)
        s = np.std(vals, axis=1, ddof=1)
        dsubEE = s / np.sqrt(n)

    # 排序处理
    sorted_indices = np.argsort(λ)


    λ = λ[sorted_indices]
    subEE = subEE[sorted_indices]
    dsubEE = dsubEE[sorted_indices]

    EE=-np.log(np.prod(subEE))
    dEE=np.sqrt(np.sum((dsubEE/subEE)**2))

    # 绘图逻辑
    # fig=plt.figure()
    # plt.errorbar(λ, subEE, yerr=dsubEE, fmt='o-', markersize=5, color='blue', ecolor='red', capsize=5,label=f'L={L}')
    # plt.xlim(-0.2, 1)
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$EE_{icr}$')
    # plt.title(f'$\Delta S_2={EE:.4f} \pm {dEE:.4f}$')
    # fig.savefig(os.path.join(folder_path,f"U={params[1]}L={L}θ={t}N={params[-2]}subEE.png"))
    # plt.close()
    return EE,dEE

def File_Process(folder_paths, throw=0):
    Data = pd.DataFrame([ ], columns=['t', 'L', 'EE', 'dEE'])
    
    for count, folder in enumerate(os.listdir(folder_paths)):
        folder_path = os.path.join(folder_paths, folder)
        if not os.path.isdir(folder_path):
            continue

        U,data=File_Process1(folder_path,throw)
        Data = pd.concat([Data, data], ignore_index=True)

    Data.to_csv(os.path.join(folder_paths, "datafit.csv"), index=False)

    return U,Data

def extract_numbers(s):
    """从字符串提取所有数值（支持负数和小数）"""
    return [float(num) for num in re.findall(r'-?\d+\.?\d*', s)]

def combine_data(matrix,throw=0):
    """合并数据表，按标签聚合数据"""
    data_dict = {}
    
    for row in matrix:
        if len(row)==0:
            continue
        label = row[0]   # 保留三位小数
        values = row[throw+1:]
        
        if label not in data_dict:
            data_dict[label] = []
        data_dict[label].extend(values)
    
    # # 构建结果矩阵
    labels = sorted(data_dict.keys())
    max_len = min(len(v) for v in data_dict.values())
    
    combined = np.empty([len(data_dict),max_len+1])
    for index, (key, value) in enumerate(data_dict.items()):
        combined[index,0]=key
        combined[index,1:]=value[:max_len]
    
    return combined

# 数据处理函数
def unit_neardata(data):
    units = []
    for i in range(len(data)):
        neighbors = [j for j in range(len(data)) 
            if max(data[i][1]-data[i][2],data[j][1]-data[j][2]) <= min(data[i][1]+data[i][2],data[j][1]+data[j][2])  ] 
            # if (data[j][1]-data[j][2] <= data[i][1]-data[i][2]<= data[j][1]+data[j][2] or data[j][1]-data[j][2] <= data[i][1]+data[i][2]<= data[j][1]+data[j][2] or
            #     data[i][1]-data[i][2] <= data[j][1]-data[j][2] <= data[i][1]+data[i][2]  or data[i][1]-data[i][2] <= data[j][1]+data[j][2] <= data[i][1]+data[i][2] )] 
        units.append(neighbors)
    LLL=[len(x) for x in units]
    return units[np.where(LLL==np.max(LLL))[0][-1] ]

def phy_PD(folder_path):
    Data = []
    L = []
    t = []
    V = []
    EK , dEK , EV , dEV , R0 , dR0 , R1 , dR1 = [],[],[],[],[],[],[],[] 

    for file in os.listdir(folder_path):
        if not file.endswith(".csv") or "phy" not in file:
            continue
        print(file)
        params = extract_numbers(file)
        if params[0]==120 or params[0]==60:
            V.append(params[2])
            L.append(int(params[3]))
            if params[3]==params[4]:
                t.append(params[6])
            else:
                t.append(params[5])
        else:
            V.append(params[1])
            L.append(int(params[2]))
            if params[2]==params[3]:
                t.append(params[5])
            else:
                t.append(params[4])
        full_path = os.path.join(folder_path, file)
        data = np.array(pd.read_csv(full_path, header=None))
        EK.append( np.mean(data[:,0]) )
        EV.append( np.mean(data[:,1]) )
        dEK.append( sem(data[:,0]) )
        dEV.append( sem(data[:,1]) )

        R = np.mean(data[:,2:6],axis=0)
        dR = sem(data[:,2:6],axis=0)
        Rdelta = np.mean(data[:,6:10],axis=0)
        dRdelta = sem(data[:,6:10],axis=0)

        # R0.append(R[0]+R[1])
        R0.append(R[0]+R[1]-R[2]-R[3])
        dR0.append( np.sqrt(np.sum(dR**2)) )
        # R1.append(Rdelta[0]+Rdelta[1])
        R1.append(Rdelta[0]+Rdelta[1]-Rdelta[2]-Rdelta[3])
        dR1.append( np.sqrt(np.sum(dRdelta**2)) )
        Data.append([V[-1],t[-1], L[-1], EK[-1] , dEK[-1],EV[-1] , dEV[-1] , R0[-1] , dR0[-1] , R1[-1] , dR1[-1] ])

        if np.any(np.isnan(Data[-1])):
            raise ValueError("NaN value found in the file: " + file)

    Data = pd.DataFrame(Data, columns=['V','t', 'L', 'EK', 'dEK', 'EV', 'dEV', 'R0', 'dR0', 'R1', 'dR1'])
    return t[1]/L[1],Data
       
def EEfit_F(x, y, fix, arg):
    # y=a*x*lnx + b*x + c*log(x) + d
    if fix == 0:
        A=np.ones((len(x),4))
        A[:,0]=x*np.log(x)
        A[:,1]=x
        A[:,2]=np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y
    
    elif fix == 1:
        y_adj = y - arg * x * np.log(x)
        A=np.ones((len(x),3))
        A[:,0]=x
        A[:,1]=np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj
    
    elif fix == 2:
        y_adj = y - arg * x
        A=np.ones((len(x),3))
        A[:,0]=x*np.log(x)
        A[:,1]=np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj

    elif fix == 3:
        y_adj = y - arg * np.log(x)
        A=np.ones((len(x),3))
        A[:,0]=x*np.log(x)
        A[:,1]=x
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj
    elif fix == [1,3]:
        y_adj = y - arg[0] * x * np.log(x) - arg[1] * np.log(x)
        A=np.ones((len(x),2))
        A[:,0]=x
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj
    elif fix == [2,3]:
        y_adj = y - arg[0] * x - arg[1] * np.log(x)
        A=np.ones((len(x),2))
        A[:,0]=x*np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj
    elif fix == [1,2]:
        y_adj = y - arg[0] * x *np.log(x) - arg[1] * x
        A=np.ones((len(x),2))
        A[:,0]=np.log(x)
        return np.linalg.inv(A.transpose() @A)@A.transpose()@y_adj

def QuenchFile_Process(folder_paths, throw=0):
    Data = pd.DataFrame([ ], columns=['R', 'L', 'EE', 'dEE'])
    
    for count, folder in enumerate(os.listdir(folder_paths)):
        folder_path = os.path.join(folder_paths, folder)
        if not os.path.isdir(folder_path):
            continue

        data=QuenchFile_Process1(folder_path,throw)
        Data = pd.concat([Data, data], ignore_index=True)

    Data.to_csv(os.path.join(folder_paths, "datafit.csv"), index=False)

    return Data

def QuenchFile_Process1(folder_path,throw=0):
    Data = []

    for file in os.listdir(folder_path):
        if not file.endswith(".csv") or "EE" not in file:
            continue
        print(file)
        params = extract_numbers(file)
        if params[0]==120 or params[0]==60:
            dU=params[3]-params[2]
            L=(int(params[4]))
            if params[4]==params[5]:
                t=(params[8])
            else:
                t=(params[7])
            R=np.abs(dU)/t
        else:
            os.error("QuenchFile_Process1 only supports quench files with initial 60 or 120.")
            # U=params[1]
            # L=(int(params[2]))
            # if params[2]==params[3]:
            #     t=(params[5])
            # else:
            #     t=(params[4])

        
        EE,dEE=File_Process2(file,folder_path,throw)
        Data.append([t,R, L, EE , dEE])

    Data = pd.DataFrame(Data, columns=['t','R', 'L', 'EE', 'dEE'])
    return Data