# 3、寻找泄露的位置（同1使用同一种F检验）
print('Find relevant bits:')
BitChoice = [False] * 16  # 存放泄露位置（子密钥）
for i in range(16):
    BitChoice[i] = True # 暂且认为该位置有泄露。
    m = 0
    Xm = (k[:, 0] & 0x0).astype('uint16') # 用于储存排除了需要验证是否有泄露，以及确定有泄露位置的子密钥的折叠密钥
    for j in range(0, 16):
        if (BitChoice[j] == False): # 当此位置为False时，记录该位置子密钥
            Xm = Xm ^ ((k[:, j] & 0x01).astype('uint16') << m) # 将第j个位置复制到Xm
            m = m + 1
    pv = maf.Full1vsFull2(Traces, Xm, m, Kp, 16) # 判断当前位置是否有影响
    if (pv < threshold): # 判断没有了i位置的X矩阵与完整输入是否有类似的表现
        print('Delete bit {0}, pv={1}'.format(i+bit_start, pv))  # 表示当前位置不涉及泄露
    else:
        print('Cannot delete bit {0}, pv={1}'.format(i+bit_start, pv)) # 表示当前位置有泄露
        BitChoice[i] = False

def Full1vsFull2(self,Y,X1,m1,X2,m2):
    """
    两模型的F检验，主函数有用到
    :param Y: 功耗（单个功耗点的）
    :param X1: 密钥数组（某种方式将所有子密钥合并成一个十进制数）
    :param m1: X1的密钥或者密钥组数量）
    :param X2: 完整的密钥数组
    :param m2: 完整的密钥的个数（16）
    :return:
    """
    maxpv = 0
    [pv, R2, SSE1,SST,F] = ModelAnalysis_Ftest().FullbaseFtest(X1,Y)
    [pv, R2, SSE2,SST,F] = ModelAnalysis_Ftest().FullbaseFtest(X2,Y)
    pv=ModelAnalysis_Ftest().Ftest_nested(2**m1, 2**m2, len(Y), SSE1, SSE2)
    pv = -np.log10(pv)
    if(pv>maxpv):
        maxpv=pv
    return maxpv

def FullbaseFtest(self,X,Y):
    """
    基于F检验的回归分析，检验X对Y的解释能力（即X对Y是否有影响）
    :param X: 输入X，合并过的密钥Kp
    :param Y: Y矩阵，应该是轨迹
    :return: p_value, R2, SSE,SST, F：p值、R²、
    """
    N=len(Y)  # 样本数量
    m=len(set(X)) # X的不同取值个数，数据中密钥有多少个不同值（全随机一般都不一样了，相当于数据量，也不一定）
    #if(N/m<10):
    #    print("Warning: limited samples for each category, unreliable results!")
    XY=sorted(list(zip(X,Y))) # 将X和Y配对，并按X值升序，方便后续分组（第一列代表密钥，第二列代表对应功耗值）
    dictavg = {} # 存储每个X值对应的Y值平均值（每个的格式为：X：Y）
    dictcount={} # 存储X出现的次数（每个的格式为：X：出现次数）
    avg=np.sum(Y)/N # Y的均值
    Xv=XY[0][0] # 当前X值
    sumy=float(XY[0][1]) # 当前Y值的总和
    count=1 # 当前X值出现的次数
    for i in range(1,N):  # 对所有样本进行分组统计
        if(XY[i][0]==Xv):#Same entry    # 如果X值相同
           sumy=sumy+XY[i][1]  # 累加Y值
           count=count+1 # 增加次数
        else: #A new entry  # 遇到新的X值
            dictavg.update({Xv:sumy/count}) # 计算并存储此X的平均Y值
            dictcount.update({Xv:count}) # 存储X的计数
            Xv=XY[i][0] # 更新当前X值
            count=1 # 重置次数
            sumy=float(XY[i][1]) # 重置当前Y值的总和
    #Last key
    dictavg.update({Xv: sumy / count}) # 存储最后一个X值的平均Y值
    dictcount.update({Xv: count}) # 存储最后一个X值的计数
    # dictavg储存了所有种类X对应的Y平均值（对应的Y的和/出现次数）
    # dictcount储存了所有种类X的出现次数
    #Now dictavg has all the means
    #Upper
    # 计算F检验的分子和分母
    Upper=0 # F检验的分子部分，反映了不同 X 值对 Y 的贡献差异
    for x in set(X): # 遍历X的所有不同的取值
        Upper=Upper+dictcount[x]*((dictavg[x]-avg)**2) # 当前x的出现次数*(当前x对应y的均值-所有Y均值)²
    Upper=Upper/float(m-1) # 标准化
    #Lower
    Lower=0 # F 检验的分母部分，衡量每个 Y 值与其对应的 X 平均值之间的差异
    for x,y in XY: # 遍历所有（X，Y）配对
        Lower=Lower+((y-dictavg[x])**2) # 每个样本 y 与其对应 x 的y平均值之差的平方
    Lower=Lower/(N-m) # 标准化

    #F-test  F检验
    F=Upper/Lower # 计算F值
    p_value=f.sf(F,m-1,N-m) # 计算p值，当p小于阈值α，则H0不成立。(数据量N一定要大于数据中不同的数值m，不然跑不出结果)

    #Compute SSE for further test  计算R²和SSE
    SSE=Lower*(N-m)  # 残差平方和
    SST=np.sum((Y-avg)**2) # 总平方和
    R2=1-SSE/SST # 拟合优度

    return [p_value, R2, SSE,SST, F] # 返回F检验结果

def Ftest_nested(self,p1,p2,N, SSE1, SSE2): #单纯的F检验公式函数
    """
    :param p1: 密钥组1数量
    :param p2: 密钥组2数量
    :param N: 轨迹数量
    :param SSE1: 模型1SSE
    :param SSE2: 模型2SSE
    :return:
    """
    #Calculate p-vale
    F=((SSE1-SSE2)/(p2-p1))/(SSE2/(N-p2))
    #print('p1={1},p2={2}, N={3},SSE1={4},SSE2={5},F={0}'.format(F,p1,p2,N,SSE1,SSE2))
    p_value = f.sf(F, p2-p1, N-p2) # 计算p值
    return p_value