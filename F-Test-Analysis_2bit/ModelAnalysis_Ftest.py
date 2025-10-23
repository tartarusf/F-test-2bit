# Using nested F-test analysing power consumption model/operand data-path
# 这部分是F检验的核心函数代码，后面的main主函数大概率是为了
import numpy as np
from scipy.stats import f
from scipy import linalg
import HWEnum
import math
class ModelAnalysis_Ftest(object):  # F检验类
    def PCA(self, data): # 主成分分析（PCA） 算是一种预处理，有降维提取特征的能力
        centered_data = data - np.mean(data)  # 对数据进行中心化处理
        if (np.isnan(centered_data).any()):  # 检查是否有缺失值
            return
        U, S, V = linalg.svd(centered_data, full_matrices=False, lapack_driver='gesvd')
        components = V  # 主成分
        coefficients = np.dot(U, np.diag(S))  # 计算系数
        return coefficients
    # Regression test with all possible values of explanatory variable X
    # Input (X,Y)
    # X: Explanatory variable (i.e. intermediate states in SCA)
    # Y: Response/Dependent variable (i.e. samples on the power trace)
    # Return [p_value, R2, SSE]
    # p_value: statistic p-value of "X does not give a valid model(i.e. Y does not rely on X)". Smaller -> "Y relies on X"
    # R2: Coefficient of determination (https://en.wikipedia.org/wiki/Coefficient_of_determination) Higher->"Y relies on X"
    # SSE:  Residual sum of squares (https://en.wikipedia.org/wiki/Coefficient_of_determination)
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

    #  Regression test with m columns of linear binary variables Xm
    #  Input (Xm,m, Y,B=None)
    #  Xm: the binary matrix of all explanatory variables
    #  m:  the number of column
    #  Y:  Response/Dependent variable (i.e. samples on the power trace)
    #  B:  Moore–Penrose inverse of Xm (avoid computing this repeatedly for all samples)
    #  Return [p_value, R2, SSE]
    #  p_value: statistic p-value of "X does not give a valid model(i.e. Y does not rely on X)". Smaller -> "Y relies on X"
    #  R2: Coefficient of determination (https://en.wikipedia.org/wiki/Coefficient_of_determination) Higher->"Y relies on X"
    #  SSE:  Residual sum of squares (https://en.wikipedia.org/wiki/Coefficient_of_determination)
    def LinearbaseR2(self,Xm,m, Y,B=None): # 基于线性回归的R²计算（适用于m列二值变量Xm）
        N=len(Y) # 样本数量
        avg=np.mean(Y,0) # Y的均值
        #if(B is None):
        #    B = np.linalg.pinv(Xm)
        result=linalg.lstsq(Xm, Y) # 最小二乘法解线性回归，即模型拟合
        beta=result[0] # 回归系数
        SSE=result[1] # 残差平方和
        #beta=np.matmul(B,Y)
        #Ye=np.matmul(Xm,beta)
        SST=sum((Y-avg)**2) # 总平方和
        #for i in range(N):
        #    SSE=SSE+(Y[i]-Ye[i])**2
        #    SST=SST+(Y[i]-avg)**2
        R2=1-SSE/SST # 拟合优度
        #Calculate p-vale 计算F值和p值
        F=((SST-SSE)/(m))/(SSE/(N-m-1))
        p_value = f.sf(F, m , N - m-1) # 计算p值，这里应该是用于评价模拟拟合指数
        return [p_value,R2, SSE] # 返回p值、R²和SSE

    # Ftest for nested basis (https://en.wikipedia.org/wiki/F-test "Regression problems")
    # Input:(p1,p2,N,SSE1,SSE2)
    # p1: number of explanatory variables in Model 1 ("restricted model")
    # p2: number of explanatory variables in Model 2 ("full model") (since Model 1 is part of Model 2, p2>p1)
    # N: number of traces
    # SSE1: Residual sum of squares of Model 1
    # SSE2: Residual sum of squares of Model 2 (since Model 1 is part of Model 2, SSE1>SSE2)
    # return p_value
    # p_value: statistic p-value of "Model 1 has the same explanatory power as Model 2", Smaller -> "Model 2 is significantly better"
    # F检验用于比较嵌套模型（即模型1是模型2的子集）
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
    # Building a binary explanatory variables matrix for input m-bit state X
    # Input: X,m, XL, mode
    # X: Explanatory variable (i.e. intermediate states in SCA)
    # m: bit length of X
    # XL: Current matrix(if none, create a new matrix)
    # mode: 'linear' only add linear term, i.e. each bit of X
    #       'full' add all possible combination from X
    # 构建回归基函数（用于添加线性或全交叉项）
    def AddTerm(self,X,m,XL=None,mode='full'):  # X矩阵交互项的添加？（还不清楚是干嘛的，后面的代码好像也没用）
        if(XL is None): # 如果没有提供XL矩阵，则初始化为全1矩阵
            XL=np.ones(np.size(X)).transpose()
        else:
            XL = XL.transpose()
        if(mode=='linear'): # 只添加线性项，矩阵的第i行为X中所有数据的二进制倒数i位
            for i in range(m):
                t = np.ones(np.size(X), dtype='uint8') # 创建二进制向量，好像是全1向量
                t = t & ((X >> i) & 0x1) # 用按位与运算将第i位的值添加进去t
                XL = np.vstack((XL, t.transpose()))  # 将结果t添加到XL中，算是提取了一种新的特征
        else:  # 添加全交叉项，矩阵的第i行为i的二进制所占有位数对应X中所有数据的二进制倒数i位或联合与值
            for i in range(1,2**m): # 遍历所有可能的组合
                t = np.ones(np.size(X), dtype='uint8') # 创建二进制向量，初始化全1向量
                for j in range(m):
                    if((i>>j)&0x01>0): # 判断第j个位置是否为1
                        t = t & ((X >> j) & 0x1) # 按位与运算，把X的第j列拷贝过去
                XL = np.vstack((XL, t.transpose())) # 将结果添加到XL中
        XL=XL.transpose() # 返回最终的XL矩阵
        return XL

    # 用于LRA（线性回归分析），并计算结果
    # def LRA_Fullbase(self,trs,X,start,end,filename=None,mode="pvalue",append=False):
    #     if (filename != None):
    #         if (append):
    #             fo = open(filename, "a") # 以追加模式打开文件
    #         else:
    #             fo = open(filename, "w") # 以写入模式打开文件
    #     maxpv = 0 # 初始化变量：最大p值
    #     pv = np.zeros((1, end - start))
    #     for i in range(start, end):
    #         if (i % 100 == 0):
    #             print("i=" + str(i))
    #         Y = trs.traces[:, i]
    #         [pv[0, i - start], R2, SSEL,SST,F] = ModelAnalysis_Ftest().FullbaseFtest(X,Y);
    #         pv[0, i - start] = -np.log10(pv[0, i-start]) # 计算-log10(p-value)
    #         if (filename != None):
    #             if(mode=='pvalue'):
    #                 fo.write(str(pv[0, i-start])+'\t') # 写入p值
    #             else:
    #                 fo.write(str(F) + '\t') # 写入F值
    #         if(pv[0, i-start]>maxpv): # 更新最大p值
    #             maxpv=pv[0, i-start]
    #
    #     if (filename != None):
    #         fo.writelines('\n')
    #         fo.close()
    #     return maxpv

    def LRA_Fullbase(self,Y,X):
        """
        这个是主文件使用的，上面那个同名函数暂时无用吧
        :param Y: 功耗轨迹（单个特征点）
        :param X: 合并过的10进制密钥（N个）
        :return: pv：p值，用于衡量X与Y的显著性关联
        """
        [pv, R2, SSEL,SST,F] = ModelAnalysis_Ftest().FullbaseFtest(X,Y)
        pv = -np.log10(pv)
        return pv

    def nCr(self,n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    # 生成回归基矩阵，交互项（这个好像更符合论文的公式，有常数项和交互项）
    def BuildRegressionBasis(self, X, m, degree=None): # 主函数有使用到  需要修改
        """
        就是生成交互项的
        :param X: 输入X（合并的密钥）
        :param m: 应该是16个位置（子密钥个数）
        :param degree: 泄露程度
        :return:
        """
        if (degree == None):
            degree = m # 预设是m，即最大的泄露度
        # 初始化组合数的总数，从常数项开始
        num=1 # 代表总共需要生成的组合数
        for i in range(1,degree+1):
            num=num+ModelAnalysis_Ftest().nCr(m,i) # 计算所有可能的特征组合数量（当前degree的所有组合数量）
        #print('Basis size={0}'.format(num))
        Xe = np.ones((num,np.size(X)), np.dtype('B')).transpose() # 创建一个全1的矩阵，用于存储特征组合（交互数*数据数）
        count = 0
        hwe=HWEnum.HWEnum(m)  # 创建一个HWEnum实例，其中储存着需要用于计算的位置信息（位置为True将用于联合项计算，False不参与）
        while(hwe.Hw<=degree): # 遍历可能的泄露程度项（因为是01密钥，hw的最大值就代表了联合项的最大数，即degree）
        #for i in range(1, 2 ** m):
            t = np.ones(np.size(X), np.dtype('B')) # 初始化一个全1的二进制向量，用来存储当前的特征组合，其长度为数据量
            #bin = '{0:0{1}b}'.format(i, m)[::-1]
            #hw = 0
            # 遍历特征（0到m-1），根据hwe.M[j]的值决定是否选择该特征
            for j in range(0, m): # 遍历16个位置
                #if (bin_i[j] == '1'):
                if(hwe.M[j]): # 会判断这个hwe.M[j]（即第j个子密钥位）是否允许被操作
                    t = t & ((X >> j) & 0x1) # t复制所有密钥中二进制第j个位置（即第j个位置的子密钥）
                #    hw = hw + 1
            #Xe = np.vstack((Xe, t.transpose()))
            # 将当前特征组合t存入Xe矩阵中的相应位置
            Xe[:,count+1]=t # 将第j位的子密钥都保存在Xe的第count+1列（此数组的第一列是全1，用于常数）
            count = count + 1 # 增加特征组合的计数（换到第二个组合项）
            #if(count%100==0):
            #    print('count={0}'.format(count))
            del t # 释放t占用的内存
            hwe.HwNext()  # 用于更新hwe（那些位置要在下一轮用于计算）
        #Xe = Xe.transpose()
        # 返回生成的回归基矩阵Xe和特征组合的总数(count + 1)
        return [Xe, count+1]

    # 用于生成排除某项的交互项（比如排除了X2）
    def BuildRegressionBasis1(self, X, m, NonUsingBit=None):
        num=0
        for i in range(len(NonUsingBit)): # 排除某些项（交互项？）
            if(NonUsingBit[i]==False):
                num=num+1
        num=2**num
        Xe = np.ones((num,np.size(X)), np.dtype('B')).transpose()
        count = 0
        for i in range(1, 2 ** m): # 计算所有可能的特征组合项
            t = np.ones(np.size(X), np.dtype('B'))
            bin_i = '{0:0{1}b}'.format(i, m)[::-1]
            Delete=False
            for j in range(0, m):
                if (bin_i[j] == '1'):
                    if(NonUsingBit[j]): # 排除不需要的特征
                        Delete = True
                        break
                    else:
                        t = t & ((X >> j) & 0x1)
            if (Delete):  # remove
                del t
                continue
            else:
                #Xe = np.vstack((Xe, t.transpose()))
                Xe[:,count+1]=t
                count = count + 1
                if(count%10==0):
                    print('count={0}'.format(count))
                del t
        #Xe = Xe.transpose()
        return [Xe, count+1]
    # 计算每个数据点的 p 值和 R² 值，并将它们写入指定的文件中（不涉及F检验）
    def LRA(self,trs,X,m,start,end,filename=None,mode="pvalue",append=False):
        if (filename != None):
            if(append): # 打开文件
                fo = open(filename, "a")
            else:
                fo = open(filename, "w")
        maxpv = 0
        #pv = np.zeros((1, end - start))
        #if(B is None):
        #    B = np.linalg.pinv(X)
        # 调用 LinearbaseR2 方法进行回归分析，排除了某个子密钥（筛选后）
        [pv, R2, SSEL] = ModelAnalysis_Ftest().LinearbaseR2(X, m, trs.traces[:, start:end])
        for i in range(start, end): # 计算p值
            pv[ i - start] = -np.log10(pv[i - start])
            if (filename != None):
                if (mode == 'pvalue'):
                    fo.write(str(pv[i - start]) + '\r\n')
                else:
                    fo.write(str(R2[i - start]) + '\r\n')
            if(pv[i-start]>maxpv):
                maxpv=pv[i-start]
        if (filename != None):
            fo.close()
    def LRA(self,Y,X,m):
        [pv, R2, SSEL] = ModelAnalysis_Ftest().LinearbaseR2(X, m, Y)
        pv = -np.log10(pv)
        return pv

    # 全模型和排除子密钥后的受限模型F检验，并根据 F 测试计算每个数据点的 p 值。
    def LRAvsFull(self,trs,X,m,XL,ml,start,end,filename=None,append=False):
        if (filename != None):
            if(append):
                fo = open(filename, "a")
            else:
                fo = open(filename, "w")
        maxpv = 0
        # 输入了XL简化后的特征矩阵
        [pv, R2, SSEL] = ModelAnalysis_Ftest().LinearbaseR2(XL, ml, trs.traces[:, start:end])
        for i in range(start, end):
            #if (i % 100 == 0):
            #    print("i=" + str(i))
            Y = trs.traces[:, i]
            [pv, R2, SSE,SST,F] = ModelAnalysis_Ftest().FullbaseFtest(X,Y)
            pv=ModelAnalysis_Ftest().Ftest_nested(ml, 2**m, len(Y), SSEL[i-start], SSE)
            pv = -np.log10(pv)
            if (filename != None):
                fo.write(str(pv) + '\r\n')
            if(pv>maxpv):
                maxpv=pv
        if (filename != None):
            fo.close()
        return maxpv
    # 注意：主文件使用的是下面这个版本
    def LRAvsFull(self,Y,X,m,XL,ml):
        [pv, R2, SSEL] = ModelAnalysis_Ftest().LinearbaseR2(XL, ml, Y)
        [pv, R2, SSE,SST,F] = ModelAnalysis_Ftest().FullbaseFtest(X,Y)
        pv=ModelAnalysis_Ftest().Ftest_nested(ml, 2**m, len(Y), SSEL, SSE)
        pv = -np.log10(pv)
        return pv

    # （主函数没用到）比较两个不同X矩阵全模型的显著性差异
    # def Full1vsFull2(self,trs,X1,m1,X2,m2,start,end,filename=None,append=False):
    #     if (filename != None):
    #         if(append):
    #             fo = open(filename, "a")
    #         else:
    #             fo = open(filename, "w")
    #     maxpv = 0
    #     for i in range(start, end):
    #         #if (i % 100 == 0):
    #         #    print("i=" + str(i))
    #         Y = trs.traces[:, i]
    #         [pv, R2, SSE1,SST,F] = ModelAnalysis_Ftest().FullbaseFtest(X1,Y)
    #         [pv, R2, SSE2,SST,F] = ModelAnalysis_Ftest().FullbaseFtest(X2,Y)
    #         pv=ModelAnalysis_Ftest().Ftest_nested(2**m1, 2**m2, len(Y), SSE1, SSE2)
    #         pv = -np.log10(pv)
    #         if (filename != None):
    #             fo.write(str(pv) + '\r\n')
    #         if(pv>maxpv):
    #             maxpv=pv
    #     if (filename != None):
    #         fo.close()
    #     return maxpv
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

    # def LRA1vsLRA2(self,trs,XL1,m1,XL2,m2,start,end,filename=None,append=False):
    #     if (filename != None):
    #         if(append):
    #             fo = open(filename, "a")
    #         else:
    #             fo = open(filename, "w")
    #     maxpv = 0
    #     [pv, R2, SSEL1] = ModelAnalysis_Ftest().LinearbaseR2(XL1, m1, trs.traces[:, start:end])
    #     [pv, R2, SSEL2] = ModelAnalysis_Ftest().LinearbaseR2(XL2, m2, trs.traces[:, start:end])
    #     for i in range(start, end):
    #         if (i % 100 == 0):
    #             print("i=" + str(i))
    #         Y = trs.traces[:, i]
    #         pv=ModelAnalysis_Ftest().Ftest_nested(m1, m2, len(Y), SSEL1[i], SSEL2[i])
    #         pv = -np.log10(pv)
    #         if (filename != None):
    #             fo.write(str(pv) + '\r\n')
    #         if(pv>maxpv):
    #             maxpv=pv
    #     if (filename != None):
    #         fo.close()
    #     return maxpv
    def LRA1vsLRA2(self,Y,XL1,m1,XL2,m2):
        """
        :param Y: 轨迹
        :param XL1: 交互密钥矩阵1
        :param m1: 密钥组合数1
        :param XL2:交互密钥矩阵2
        :param m2: 密钥组合数2
        :return:
        """
        [pv, R2, SSEL1] = ModelAnalysis_Ftest().LinearbaseR2(XL1, m1, Y)
        [pv, R2, SSEL2] = ModelAnalysis_Ftest().LinearbaseR2(XL2, m2, Y)
        pv=ModelAnalysis_Ftest().Ftest_nested(m1, m2, len(Y), SSEL1, SSEL2)
        pv = -np.log10(pv)
        return pv

if __name__ == '__main__':
        # instr=10 # 未知变量
        point=500 # 可能是点位置？
        threshold=5  # 检测阈值
        file = 'E:\\PZH\\code\\F-Test-Analysis\\data\\SR01_10W_9267_2024_11_23_11h32m44s_0.98612.mat'
        # trs = TRS_Reader.TRS_Reader(  # 数据读取，但缺失代码
        #     "ModelFtest_Instr" + str(instr) + "_2000Samples_WithEOR_Rand2.trs")
        # trs.read_header()
        N = 10000 # 数据量
        # trs.read_traces(N) # 读取功耗点
        data = np.load(file) # 改为自己的输入
        traces = data['traces'][0:N,:]
        # keys = data['key'][0:N,:]
        plaintext = data['plaintext'][0:N,:]
        # Op1 = (trs.plaintext[:, 0]) & 0x03  # 选取了第一份子密钥？
        # Op2 = (trs.plaintext[:, 1]) & 0x03
        # Op3 = (trs.plaintext[:, 2]) & 0x03
        # Op4 = (trs.plaintext[:, 3]) & 0x03

        Op = (plaintext[:, 15])   # 选取了第一份子密钥？
        Op1 = (plaintext[:, 0]) & 0x03  # 选取了第一份子密钥？并且中取二进制的最低两位
        Op2 = (plaintext[:, 1]) & 0x03
        Op3 = (plaintext[:, 2]) & 0x03
        Op4 = (plaintext[:, 3]) & 0x03   # 选取了第一份子密钥？

        # Op1 = (keys[:, 0]) & 0x03  # 选取了第一份子密钥？并且中取二进制的最低两位
        # Op2 = (keys[:, 1]) & 0x03
        # Op3 = (keys[:, 2]) & 0x03
        # Op4 = (keys[:, 3]) & 0x03

        Op12 = (Op2 << 2) ^ Op1  # 将op2左移两位后与op1异或？这个op12应该是op1与op2的合并即联合
        X = (Op4.astype(dtype='uint16')<<6) ^ (Op3.astype(dtype='uint16')<<4) ^ Op12 # 把op12与op3和op4拼凑在一起，确实是完整的X输入，但只有8位
        Y = traces[:, point]

        #Example 1: the following lines verify if Y depends on Op1
        [pv_F, R2_F, SSE_F, SST_F, F_F] = ModelAnalysis_Ftest().FullbaseFtest(X, Y)
        [pv_L, R2_L, SSE_L, SST_L, L_F] = ModelAnalysis_Ftest().FullbaseFtest(X>>2, Y)#remove Op1 移除第一个影响
        pv =ModelAnalysis_Ftest().Ftest_nested(len(set(X>>2)), len(set(X)), N, SSE_L, SSE_F)
        pv = -np.log10(pv)
        if(pv>threshold):
            print("Sample does depend on Op1!")  # 检验结果位Op1是否对模型有重要影响
        else:
            print("Sample does not depend on Op1!")

        # Example 2: the following lines verify whether there are interaction with Op1 and Op2\3\4
        [pv_F, R2_F, SSE_F, SST_F, F_F] = ModelAnalysis_Ftest().FullbaseFtest(X, Y)
        XL=ModelAnalysis_Ftest().AddTerm(X>>2, 6, None,'full')#Construct a full model for op2\3\4 包含op234的模型
        XL = ModelAnalysis_Ftest().AddTerm(Op1, 2, XL, 'full')  # Adding a full model for op1 alone, so no interaction between Op1 and Op2/3/4  # 只包含op1的模型
        [pv_L, R2_L, SSE_L] = ModelAnalysis_Ftest().LinearbaseR2(XL, np.size(XL,1), Y)# Regression with this restricted model
        pv = ModelAnalysis_Ftest().Ftest_nested(np.size(XL,1), len(set(X)), N, SSE_L, SSE_F)
        pv = -np.log10(pv)
        if (pv > threshold):
            print("There are interaction between Op1 and Op2/3/4!")
        else:
            print("There are not interaction between Op1 and Op2/3/4!")


        # Example 3: the following lines tested whether the influence of Op1 is linear
        # 验证 Op1 对 Y 的影响是否为线性。
        [pv_F, R2_F, SSE_F, SST_F, F_F] = ModelAnalysis_Ftest().FullbaseFtest(Op1, Y)#Full model for op1
        XL=ModelAnalysis_Ftest().AddTerm(Op1, 2, None,'linear')#Construct a linear model for op1 op1的线性模型
        [pv_L, R2_L, SSE_L] = ModelAnalysis_Ftest().LinearbaseR2(XL, np.size(XL,1), Y)# Regression with this restricted model
        pv = ModelAnalysis_Ftest().Ftest_nested(np.size(XL,1), len(set(Op1)), N, SSE_L, SSE_F)
        pv = -np.log10(pv)
        if (pv > threshold):
            print("Influence of Op1 is non-linear!")
        else:
            print("Influence of Op1 is possibly linear!")