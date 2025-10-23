# 这份代码貌似是用于模拟AES加密的运行的，并生成模拟功耗数据
# 更正：生成模拟数据后，将进行F检验的相关实验，此代码才是与论文最相关的代码
import ModelAnalysis_Ftest
import numpy as np
import Ttest
import timeit
import h5py
import multiprocessing
import numpy as np
from functools import partial
import time
Rcon = (   # Rcon: AES密钥调度中使用的常数
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)

def HW(X): #计算汉明重量的函数
        y = np.zeros(len(X)).astype(float)
        for i in range(len(X)):
                y[i] = bin(X[i]).count("1")
        return y
def xtime(x):  # 用于模拟AES加密过程中的（行？）位移操作
        MSB = (x&0x80)>>7
        x=x<<1
        x=x^(0x1b*MSB)  # 异或操作与Rcon常数
        return x
class RetroAnalysis_MaskedAES(object):  # 定义RetroAnalysis_MaskedAES类
        global sbox
        sbox = [  # S-box，用于AES中的字节替换操作
                0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
                0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
                0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
                0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
                0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
                0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
                0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
                0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
                0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
                0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
                0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
                0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
                0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
                0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
                0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
                0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]
        def Sbox(self, X):  # AES S-box替换操作
                #y = np.zeros(len(X)).astype(int)
                #for i in range(len(X)):
                #        y[i] = sbox[X[i]]
                #return y

                return np.array(sbox)[X.astype(int)]
        #Generate simulation AES traces:
        #Trace has 6 time points:
        #1) HW(Plain[0:4])
        #2) HW(Roundkey[1, 0:4])
        #3) HW(SboxOutput[1, 4])
        #4) HD(SboxOutput[1, 6], SboxOutput[1, 10])
        #5) HW(MCOutput[1, 8])
        #6) HW(MCOutput[2, 12])
        def SimulationOrace(self,Plain,Key):  # 模拟AES加密过程，并生成泄露的跟踪数据
                """
                :param Plain: 明文
                :param Key: 密钥
                :return: Cipher,Traces：密文，轨迹
                """
                Traces = np.zeros((np.size(Plain, 0), 6))

                #key schedule first
                key = np.copy(Key)
                Rkey = np.zeros((np.size(Key, 0), 11, 16),dtype='uint8')
                Rkey[:,0,:]=Key
                for r in range(10):
                        temp = np.copy(key[:, 12: 16])
                        temp[:, 0: 3]=np.copy(key[:, 13: 16])
                        temp[:, 3]=np.copy(key[:, 12])

                        temp = self.Sbox(temp)


                        temp[:, 0]=temp[:, 0]^Rcon[r+1]


                        key[:, 0: 4]=key[:, 0: 4]^temp
                        key[:, 4: 8]=key[:, 4: 8]^key[:, 0: 4]
                        key[:, 8: 12]=key[:, 8: 12]^key[:, 4: 8]
                        key[:, 12: 16]=key[:, 12: 16]^key[:, 8: 12]

                        Rkey[:, r + 1,:]=key
                #Encryption start
                # Leaky 1: HW[Plain[0:4]]
                Traces[:,0]=HW(Plain[:,0])+HW(Plain[:,1])+HW(Plain[:,2])+HW(Plain[:,3])
                # Leaky 2: HW[Roundkey[1,0:4]]
                Traces[:,1]=HW(Rkey[:,0,0])+HW(Rkey[:,0,1])+HW(Rkey[:,0,2])+HW(Rkey[:,0,3])
                #ADK0
                State = Plain^Rkey[:, 0, :]
                #10 rounds encryption
                NewState=np.copy(State)
                for r in range(1,11):
                        State = self.Sbox(State)
                        if (r == 1):
                            #Leaky 3: HW(Sout[:, 4])
                            Traces[:, 2]=HW(State[:,4])
                            #Leaky 4: HD(SboxOutput[1, 6], SboxOutput[1, 10])
                            Traces[:, 3]=HW(State[:,6]^State[:,10])
                        #ShiftRow
                        NewState[:, 0]=State[:, 0]
                        NewState[:, 1]=State[:, 5]
                        NewState[:, 2]=State[:, 10]
                        NewState[:, 3]=State[:, 15]
                        NewState[:, 4]=State[:, 4]
                        NewState[:, 5]=State[:, 9]
                        NewState[:, 6]=State[:, 14]
                        NewState[:, 7]=State[:, 3]
                        NewState[:, 8]=State[:, 8]
                        NewState[:, 9]=State[:, 13]
                        NewState[:, 10]=State[:, 2]
                        NewState[:, 11]=State[:, 7]
                        NewState[:, 12]=State[:, 12]
                        NewState[:, 13]=State[:, 1]
                        NewState[:, 14]=State[:, 6]
                        NewState[:, 15]=State[:, 11]
                        State = NewState

                        if(r<10):
                                #MixColumn
                                for j in range(4):
                                        t = np.copy(State[:, 4 * j])
                                        temp = State[:, 4 * j]^State[:, 4*j + 1]
                                        Tm = np.copy(temp)
                                        temp = State[:, 4 * j + 2]^temp
                                        temp = State[:, 4 * j + 3]^ temp
                                        Tm = xtime(Tm)
                                        Tm = Tm^temp
                                        State[:, 4 * j ]=State[:, 4 * j ]^Tm

                                        Tm = State[:, 4 * j + 1]^State[:, 4 * j + 2]
                                        Tm = xtime(Tm)
                                        Tm = Tm^temp
                                        State[:, 4 * j + 1]=State[:, 4 * j + 1]^Tm

                                        Tm = State[:, 4 * j + 2]^State[:, 4 * j + 3]
                                        Tm = xtime(Tm)
                                        Tm = Tm^temp
                                        State[:, 4 * j + 2]=State[:, 4 * j + 2]^Tm

                                        Tm = State[:, 4 * j + 3]^t
                                        Tm = xtime(Tm)
                                        Tm = Tm^temp
                                        State[:, 4 * j + 3]=State[:, 4 * j + 3]^Tm
                        #Leaky 5: HW(MCOutput[1,8])
                        if(r==1):
                                Traces[:, 4]=HW(State[:,8])
                        if (r == 2):
                        #Leaky 6: HW(MCOutput[2, 12])
                                Traces[:, 5]=HW(State[:,12])
                        #Add Round Key
                        State =State^Rkey[:, r ,:]
                Cipher=State
                return [Cipher,Traces] # 返回大概是密文和模拟轨迹

        def TVLA_Normal(self, N=100000, sigma2=16,filename="TVLA_normal.txt"):
                """
                T检验评估，使用固定密钥、随机和固定明文，轨迹的特征点位6
                1.生成第一组密钥固定、明文固定和随机的轨迹并做t检验（t检验一般是使用明文固定和随机的两组数据做的）
                2.生成第二组密钥固定、明文固定和随机的轨迹并做t检验
                :param N: 需要生成轨迹的数量
                :param sigma2: 噪声的方差，用于模拟噪声的大小
                :param filename: 储存文件名（应该包含路径）
                :return: no return
                """
                fo = open(filename, "w") # 写入文件操作
                N_T = int(N / 2) #
                print('Part 1: fixed-versus-random plaintext, fixed key TVLA:')
                print('Trial 1:') # 使用第一组轨迹做实验？
                print('Generating random trace set:')
                Plain = np.random.randint(0, 256, (N_T, 16)).astype('uint8')  # Random Plaintext 随机明文！
                Key = np.random.randint(0, 256, (1, 16)).astype('uint8')  # one fixed key 固定密钥！
                Key = np.tile(Key, (N_T, 1))
                [Cipher, TracesR] = self.SimulationOrace(Plain, Key) # 根据上述的明文密钥生成密文和轨迹
                TracesR = TracesR + np.random.randn(N_T, 6) * np.sqrt(sigma2)  # Add noise 添加噪声
                print('Generating fixed trace set:')
                Plain = np.zeros((N_T, 16)).astype('uint8')  # Fixed Plaintext 生成固定的明文！
                # Same key
                [Cipher, TracesF] = self.SimulationOrace(Plain, Key) # 固定明文和密钥！
                TracesF = TracesF + np.random.randn(N_T, 6) * np.sqrt(sigma2)  # Add noise
                # Ttest
                tvla1 = Ttest.Ttest(6) # 创建的一个Ttest对象，样本的维度是6（为什么是6）
                for i in range(N_T): # 检验每一条轨迹
                        tvla1.UpdateTrace(TracesR[i, :], True)  # 将随机明文数据传递到tvla1
                        tvla1.UpdateTrace(TracesF[i, :], False) # 将固定明文数据传递到tvla1
                print('Trial 2:') # 使用第二组
                print('Generating random trace set:')
                Plain = np.random.randint(0, 256, (N_T, 16)).astype('uint8')  # Random Plaintext
                Key = np.random.randint(0, 256, (1, 16)).astype('uint8')  # one fixed key
                Key = np.tile(Key, (N_T, 1))
                [Cipher, TracesR] = self.SimulationOrace(Plain, Key)
                TracesR = TracesR + np.random.randn(N_T, 6) * np.sqrt(sigma2)  # Add noise
                print('Generating fixed trace set:')
                Plain = np.zeros((N_T, 16)).astype('uint8')  # Fixed Plaintext
                # Same key
                [Cipher, TracesF] = self.SimulationOrace(Plain, Key)
                TracesF = TracesF + np.random.randn(N_T, 6) * np.sqrt(sigma2)  # Add noise
                # Ttest
                tvla2 = Ttest.Ttest(6)
                for i in range(N_T):
                        tvla2.UpdateTrace(TracesR[i, :], True)
                        tvla2.UpdateTrace(TracesF[i, :], False)
                print('Result summary: fixed-versus-random plaintext, fixed key')
                print('Leaking:')
                T1 = tvla1.GetT(1) #获取tvla1的统计量T值
                T2 = tvla2.GetT(1) #获取tvla2的统计量T值
                for i in range(6): # 判断每个特征点的泄露情况
                        if (T1[i] > 6 and T2[i] > 6):
                                print('Sample {0}'.format(i))
                                fo.write('Sample {0}: Leak!\r\n'.format(i))
                        else:
                                fo.write('Sample {0}: Not Leak\r\n'.format(i))
                fo.close()

        def TVLA_Key(self, N=100000, sigma2=16,filename="TVLA_key.txt"):
                """
                这部分也是T检验，但使用的是固定明文、随机和固定密钥，轨迹的特征点数量位6
                :param N:
                :param sigma2:
                :param filename:
                :return:
                """
                fo = open(filename, "w")
                N_T = int(N / 2)
                print('\n\n\nPart 2: fixed-versus-random key, fixed plaintext TVLA:')
                print('Trial 1:') # 生成第一份数据
                print('Generating random trace set:')
                Plain = np.random.randint(0, 256, (1, 16)).astype('uint8')  # One fixed plaintext（固定密钥1）
                Plain = np.tile(Plain, (N_T, 1))
                Key = np.random.randint(0, 256, (N_T, 16)).astype('uint8')  # random key（这里用的又是正常的8位的子密钥）[N,子密钥数量(16)]
                [Cipher, TracesR] = self.SimulationOrace(Plain, Key)
                TracesR = TracesR + np.random.randn(N_T, 6) * np.sqrt(sigma2)  # Add noise
                print('Generating fixed trace set:')
                Key = np.zeros((N_T, 16)).astype('uint8')  # Fixed Plaintext
                # Same key
                [Cipher, TracesF] = self.SimulationOrace(Plain, Key)
                TracesF = TracesF + np.random.randn(N_T, 6) * np.sqrt(sigma2)  # Add noise
                # Ttest
                tvla1 = Ttest.Ttest(6)
                for i in range(N_T):
                        tvla1.UpdateTrace(TracesR[i, :], True)
                        tvla1.UpdateTrace(TracesF[i, :], False)
                print('Trial 2:') # 生成第二份数据
                print('Generating random trace set:')
                Plain = np.random.randint(0, 256, (1, 16)).astype('uint8')  # One fixed plaintext（固定密钥2）
                Plain = np.tile(Plain, (N_T, 1))
                Key = np.random.randint(0, 256, (N_T, 16)).astype('uint8')  # random key
                [Cipher, TracesR] = self.SimulationOrace(Plain, Key)
                TracesR = TracesR + np.random.randn(N_T, 6) * np.sqrt(sigma2)  # Add noise
                print('Generating fixed trace set:')
                Key = np.zeros((N_T, 16)).astype('uint8')  # Fixed Plaintext
                # Same key
                [Cipher, TracesF] = self.SimulationOrace(Plain, Key)
                TracesF = TracesF + np.random.randn(N_T, 6) * np.sqrt(sigma2)  # Add noise
                # Ttest
                tvla2 = Ttest.Ttest(6)
                for i in range(N_T):
                        tvla2.UpdateTrace(TracesR[i, :], True)
                        tvla2.UpdateTrace(TracesF[i, :], False)
                print('Result summary: fixed-versus-random key, fixed plaintext')
                print('Leaking:')
                T1 = tvla1.GetT(1)
                T2 = tvla2.GetT(1)
                for i in range(6):
                        if (T1[i] > 6 and T2[i] > 6):
                                print('Sample {0}'.format(i))
                                fo.write('Sample {0}: Leak!\r\n'.format(i))
                        else:
                                fo.write('Sample {0}: Not Leak\r\n'.format(i))
                fo.close()

        def OnePointCollapsedFtest(self,point,Traces,k,Kp,start_key):
                """
                论文中的F检验方法
                :param point: 要检测的功耗点
                :param Traces: 输入轨迹
                :param k: 密钥矩阵（N*16）
                :param Kp: 将16个位置的密钥合并成1个位置10进制密钥（N）a
                :param fo: 储存文件名路径
                :return:
                """
                write_data = []
                print('Sample {0}'.format(point))
                write_data.append('Sample {0}\r\n'.format(point)) # 输入当前点
                threshold = 3
                Traces = Traces[:, point]
                maf = ModelAnalysis_Ftest.ModelAnalysis_Ftest() # 模型分析实例
                print('Find degree:')
                degree=-1 # 初始化泄露程度d？
                # 1、回归分析，用来评估第一个位置子密钥Kp是否对轨迹有显著影响（可能是论文中全模型与朴素模型对比）
                # 这部分使用的标准的F检验是代码新加的，在原论文中貌似没有提及，主要用于验证输入X是否能解释Y
                pv = maf.LRA_Fullbase(Traces, Kp)
                print('当前功耗点的pv为：{0}'.format(pv))
                # if (fo is not None):
                #         fo.write('当前功耗点的pv为：{0}\r\n'.format(pv))
                write_data.append('当前功耗点的pv为：{0}\r\n'.format(pv))

                if(pv<threshold): # 将pv小于阈值，则没有泄露
                        print('Not a leak') # 输出没有泄露

                        write_data.append('没有泄露\r\n')
                        # if (fo is not None): # 将结果写到输入的文件路径中
                        #         fo.write('Not a leak\r\n')
                        return write_data

                # 2、找到泄露程度d（这部分用的F检验公式是论文公式提到的，算是标准的F检验公式）  # 改变一下，只找泄露度为2和1的点（数量太多了）
                # [Ke4, count4] = maf.BuildRegressionBasis(Kp, 16, 4) # 构建回归基（交互项），泄露程度为4
                # # Ke4为X的交互矩阵，count是交互项数
                #
                # pv=maf.LRAvsFull(Traces,Kp, 16, Ke4, count4) # 计算当前泄露度为4的交互模型的p值
                # print('16-4, Fpv={0}'.format(pv))
                # if(pv>threshold): # 如果p值大于阈值，说明有明显差异，则泄露程度等于16
                #         degree=16 # 最初的子密钥
                # else: # 如果p值小于阈值，泄露度为4的模型与全模型表现相当

                # 改成了只检测degree为2和1的情况
                [Ke2, count2] = maf.BuildRegressionBasis(Kp, 16, 2) # 以谢露程度2来构建交互矩阵
                        # pv=maf.LRA1vsLRA2(Traces,Ke2, count2, Ke4, count4)
                        # print('4-2, Fpv={0}'.format(pv))
                        # if (pv > threshold): # 如果p值大于阈值，说明有明显差异，则泄露程度等于4
                        #         degree = 4
                        # else: # 如果p值小于阈值，泄露度为4的模型与全模型表现相当
                [Ke1, count1] = maf.BuildRegressionBasis(Kp, 16, 1)
                pv = maf.LRA1vsLRA2(Traces,Ke1, count1, Ke2, count2)
                print('2-1, Fpv={0}'.format(pv))
                if (pv > threshold):
                        degree = 2
                else:
                        degree=1
                print('Degree={0}\n\n\n'.format(degree))
                write_data.append('Degree={0}\r\n'.format(degree))
                # if(fo is not None):
                #         fo.write('Degree={0}\r\n'.format(degree))
                if(degree==-1): # 泄露程度为-1，即没检测出泄露
                    print('Not a leak')
                    write_data.append('没有泄露\r\n')
                    # if (fo is not None):
                    #         fo.write('Not a leak\r\n')
                    return write_data
                # if(degree==16):  # 泄露程度为最高
                #     print('All bits relevant')
                #     write_data.append('All bits relevant\r\n')

                    # if (fo is not None):
                    #         fo.write('All bits relevant\r\n') # 所有位置都相关
                    return write_data

                # Find relevant bits
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
                                print('Delete bit {0}, pv={1}'.format(i+start_key, pv))  # 表示当前位置不涉及泄露
                        else:
                                print('Cannot delete bit {0}, pv={1}'.format(i+start_key, pv)) # 表示当前位置有泄露
                                BitChoice[i] = False
                if all(BitChoice): # 用于检查是否没有位置泄露，以防止后面报错
                        print("没有位置造成泄露\n")
                        write_data.append('没有泄露\r\n')
                        return write_data

                X = (k[:, 0] & 0x0).astype('uint16') # 可能保留了泄露的位置的密钥值
                l = 0
                print('Relevant bits:\t')
                write_data.append('Relevant bits:\r\n')
                # if (fo is not None):
                #         fo.write('Relevant bits:\r\n')
                Rb=[] # 用于保存有泄露的位置
                for i in range(0, 16):
                        if (BitChoice[i] == False): # 等于这个值的位置都是有泄露的
                                X = X ^ ((k[:, i] & 0x01).astype('uint16') << l)
                                print('bit {0}\t'.format(i+start_key))
                                write_data.append('bit {0}\t'.format(i+start_key))
                                # if (fo is not None):
                                #         fo.write('bit {0}\t'.format(i))
                                Rb.append(i)
                                l = l + 1  # 有泄露的位置的数量
                print('\n')
                write_data.append('\r\n')
                # if (fo is not None):
                #         fo.write('\r\n')

                # 4.基于 Hamming 权重选择最佳项
                iList = [] # 储存符合条件的组合
                hwList = [] # 储存符合HW的组合
                for i in range(1, 2 ** l): # 遍历有可能的泄露组合项
                        # compute hw
                        hw = 0 # 初始化hw为0
                        for j in range(l):
                                if ((i >> j) & 0x1 == 1): # 检查第i个组合的第j位是否为1
                                        hw = hw + 1 # 计算当前第i个组合项的hw（单个子密钥应该是二进制的1位）
                        if (hw > degree): # 第i个组合hw超过泄露程度，则排除
                                continue
                        else:
                                iList.append(i) # 储存第i个组合的值
                                hwList.append(hw) # 储存其汉明重量
                iList, hwList = zip(*sorted(zip(iList, hwList)))

                # Test each term
                # 5.测试剩余可能泄露项的的相关性
                find=False
                for i in iList: # 遍历符合条件的位组合
                        Xm = (k[:, 0] & 0x0).astype('uint16') # 初始化全0矩阵
                        m = 0 # 初始化最终有泄露的组合数
                        for j in range(l): # 遍历每一个泄露的位置j
                                if ((i >> j) & 0x1 == 1): # 如果第i个项的第j给位置位1
                                        Xm = Xm ^ (((X >> j) & 0x01) << m) # 通过按位操作，更新Xm
                                        m = m + 1 # 增加m，表示一个新的泄露组合已被添加到Xm
                        pv = maf.LRA_Fullbase(Traces, Xm) # 测试当前组合与Y的p值
                        print('Term {0}, pv={1}'.format(i, pv))

                        if (pv > threshold and find==False):  # 证明该项与泄露有关
                                # find=True # 将此项标记为已经搜查
                                write_data.append('Target term {0}, pv={1}, using:\r\n'.format(i, pv))
                                # if (fo is not None):
                                #         # 输出当前的组合i和p值
                                #         fo.write('Target term {0}, pv={1}, using:\r\n'.format(i, pv))
                                for j in range(l):
                                        # 输出相应的比特位
                                        if ((i >> j) & 0x1 == 1):
                                                #print('bit {0}\t'.format(j))
                                                print('bit {0}\t'.format(Rb[j])++start_key)
                                                write_data.append('bit {0}\t'.format(Rb[j])+start_key)
                                                # if (fo is not None):
                                                #         fo.write('bit {0}\t'.format(Rb[j]))
                                print('\n')
                                write_data.append('\r\n')
                                # if (fo is not None):
                                #         fo.write('\r\n')
                # fo.flush()

                return write_data


        def CollapsedFtest_Key(self, N=100000, sigma2=16,filename="Ftest_key.txt"): # 使用内部生成的模拟数据
                fo = open(filename, "w")
                print('\n\n\nPart 3: Collasped F-test for key')
                print('Generating trace set:')
                Plain = np.zeros((N, 16)).astype('uint8')  # all zero[数据量N,子明文个数(16)] 生成全0明文矩阵
                Key = np.random.randint(0, 2, (N, 16)).astype('uint8')  # random key[数据量N,子密钥个数(16)] 生成随机密钥（只有两个值0或者1），在评测领域使用特例是没有问题的
                k=np.copy(Key) # 复制密钥副本
                k0=np.packbits(k[:,0:8], axis=1)  #每个数据子密钥前8位打包（即将8位合并变为10进制），这是第一份子密钥k0（长为数据量，每个代表当前这份数据前8位的值）
                k1=np.packbits(k[:,8:16], axis=1) # 第二份子密钥k1与上面类似
                Kp=(k0.astype('uint16')<<8)^k1  # 每个数据密钥的十进制值
                Key[np.where(Key==1)]=0x52 # 密钥中值为1的部分改为0x52（十六进制，即十进制的82）
                Key[np.where(Key == 0)] = 0x7d # 密钥中为0的部分改为0x7d（十进制的125）
                [Cipher, Traces] = self.SimulationOrace(Plain, Key)
                Traces = Traces + np.random.randn(N, 6) * np.sqrt(sigma2)  # Add noise  轨迹矩阵[数据量*功耗点数(6)],这里6个特征点的原因是因为模拟功耗直接生成了最关键的特征点，因此更多的功耗点没有作用
                # 这前面应该都是数据生成（问题：为什么01的密钥作为输入，变化过的密钥（将其中的0和1替换）用于生成轨迹）
                for i in range(6): # 分别循环每个特征点实验
                        print('Sample {0}'.format(i))
                        fo.write('Sample {0}\r\n'.format(i))
                        self.OnePointCollapsedFtest(Traces[:,i], k, Kp[:,0],fo)
                        # 这里的输入分别是，第i特征点的轨迹，密钥矩阵（N*16），转成10进制的密钥（N），和储存文件名
                fo.close()

        # def CollapsedFtest_Key_data(self, Traces, Key, filename="Ftest_key.txt"):  # 使用外部的数据作为输入
        #         fo = open(filename, "w")
        #         print('\n\n\nPart 3: Collasped F-test for key')
        #         k=np.copy(Key) # 复制密钥副本(要转变成uint8类型)
        #         k0=np.packbits(k[:,0:8], axis=1)  #每个数据子密钥前8位打包（即将8位合并变为10进制），这是第一份子密钥k0（长为数据量，每个代表当前这份数据前8位的值）
        #         k1=np.packbits(k[:,8:16], axis=1) # 第二份子密钥k1与上面类似
        #         Kp=(k0.astype('uint16')<<8)^k1  # 每个数据密钥的十进制值
        #         N_all, point = Traces.shape # 取出轨迹矩阵的数量和特征点数
        #         for i in range(point): # 分别循环每个特征点实验
        #                 print('Sample {0}'.format(i))
        #                 fo.write('Sample {0}\r\n'.format(i))
        #                 self.OnePointCollapsedFtest(Traces[:,i], k, Kp[:,0],fo)
        #                 # 这里的输入分别是，第i特征点的轨迹，密钥矩阵（N*16），转成10进制的密钥（N），和储存文件名
        #         fo.close()



        def CollapsedFtest_Key_data(self, Traces, Key, start_key, filename="Ftest_key.txt"):  # 使用外部的数据作为输入
                # fo = open(filename, "w")
                print('\n\n\nPart 3: Collasped F-test for key')
                k=np.copy(Key) # 复制密钥副本(要转变成uint8类型)
                k0=np.packbits(k[:,0:8], axis=1)  #每个数据子密钥前8位打包（即将8位合并变为10进制），这是第一份子密钥k0（长为数据量，每个代表当前这份数据前8位的值）
                k1=np.packbits(k[:,8:16], axis=1) # 第二份子密钥k1与上面类似
                Kp=(k0.astype('uint16')<<8)^k1  # 每个数据密钥的十进制值
                N_all, point = Traces.shape # 取出轨迹矩阵的数量和特征点数

                chunk_size = 1000  # 根据你的内存和数据大小调整
                num_chunks = (point + chunk_size - 1) // chunk_size

                manager = multiprocessing.Manager()  # 创建一个Manager对象，用于创建共享内存对象
                results_dict = manager.dict()  # 创建一个共享字典，用于在多个进程之间共享结果

                # 1、串行化运行
                # for i in range(point): # 分别循环每个特征点实验  17.6分钟
                #         # print('Sample {0}'.format(i))
                #         results_dict[i] = self.OnePointCollapsedFtest(i, Traces, k, Kp[:,0])
                        # 这里的输入分别是，第i特征点的轨迹，密钥矩阵（N*16），转成10进制的密钥（N），和储存文件名

                # 2、并行化运行
                # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                #         partial_func = partial(self.process_point, Traces=Traces, k=k, Kp=Kp[:, 0],results_dict = results_dict)
                #         pool.map(partial_func, range(point))

                # 2.5、并行分块改进，不然内存爆炸  14.8分钟
                with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                        for i in range(num_chunks):
                                start_index = i * chunk_size
                                end_index = min((i + 1) * chunk_size, point)
                                partial_func = partial(
                                        self.process_chunk,
                                        Traces=Traces[:, start_index:end_index],  # 传递数据块
                                        k=k,
                                        Kp=Kp,
                                        start_key =start_key,
                                        results_dict=results_dict,
                                        start_index=start_index
                                )
                                pool.map(partial_func, range(end_index - start_index))

                # 将结果写入文件
                with open(filename, 'w') as outfile:  # 打开文件准备写入
                        for i in range(point):  # 按照顺序遍历每个索引
                                try:
                                        for item in results_dict[i]:  # 遍历每个索引对应的写入内容列表
                                                # outfile.write(item + '\n')  # 写入内容到文件
                                                outfile.write(item)  # 写入内容到文件
                                except KeyError as e:
                                        print(f"警告: 索引 {i} 出现 KeyError. 可能的原因：数据处理过程中出现错误。")

        def process_point(self, i, Traces, k, Kp, start_key, results_dict):
                result = self.OnePointCollapsedFtest(i, Traces, k, Kp, start_key)
                results_dict[i] = result # 直接存储结果

        def process_chunk(self, i, Traces, k, Kp, results_dict, start_index):
                # 在这里 i 是相对于块的索引, 使用 start_index + i 来得到全局索引
                global_index = start_index + i
                result = self.OnePointCollapsedFtest(global_index, Traces, k, Kp[:, 0])
                results_dict[global_index] = result  # 直接存储结果

        def SimulationTest(self,N=10000,sigma2=16):  # 原本的检验代码，使用的是的模拟数据！！
                # self.TVLA_Normal(N, sigma2)
                # self.TVLA_Key(N, sigma2)
                self.CollapsedFtest_Key(N, sigma2)

        def SimulationTestdata(self,file,N=-1): # 改进后代码，使用外部提供的真实数据！！
                data = np.load(file)
                # self.TVLA_Normal(N, sigma2)
                # self.TVLA_Key(N, sigma2)
                self.CollapsedFtest_Key_data(data, N)




if __name__ == '__main__':
        threshold = 3  # 检验的阈值
        # file = 'E:\\PZH\\code\\F-Test-Analysis\\data\\SR01_66W_9267_2024_11_28_22h53m26s_0.98612.mat'
        file = 'E:\\PZH\\code\\F-Test-Analysis\\data\\SR32bit01_66W_9267_2024_12_12_21h21m56s_0.98574_byte32.mat'
        np.seterr(divide='ignore') # 用于设置 NumPy 处理浮点数错误的方式，特别是处理除以零的情况。通过将 divide='ignore'，它会告诉 NumPy 忽略除以零的警告
        start_time =timeit.default_timer() # 开始时间
        # RetroAnalysis_MaskedAES 类的实例化
        # 调用其 SimulationTest 方法
        # sys.argv[1] 和 sys.argv[2] 表示从命令行传入的第一个和第二个参数
        # RetroAnalysis_MaskedAES().SimulationTest(int(sys.argv[1]), int(sys.argv[2]))
        # RetroAnalysis_MaskedAES().SimulationTest(1000000, 16)  # 不要需要传入参数的版本
        N = 660000 # 选取的数据量（注意：数据量一定要多余密钥的所有组合数，不然没有检验结果）
        point = 9267
        start_key = 16 # 密钥检测结束的位置
        end_key = start_key + 16 # 密钥检测结束的位置
        with h5py.File(file, 'r') as f:
                Key = f['key'][0:N,start_key:end_key]
                Key = Key.astype(np.uint8)
                # plaintext = f['plaintext'][0:N,0:Sample]
                Traces = f['traces'][0:N,0:point]
        filename = "测试数据.txt"
        RetroAnalysis_MaskedAES().CollapsedFtest_Key_data(Traces, Key, start_key, filename)  # 直接做F检验
        end_time = timeit.default_timer() # 结束时间
        print("program takes {0} minutes process time".format((end_time-start_time)/60.0))





