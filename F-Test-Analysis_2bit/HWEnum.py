# Enumeration following the HW order (from HW=0 to HW=n)
# 这个类主要用于生成和管理特定形式的二进制数列，尤其是通过指定的方式枚举不同数量的 True 元素（即Hw），并在每次调用时通过 HwNext 方法更新状态。
class HWEnum(object):
    M=None
    dim=0  # 组合的最大个数
    pstart=-1 # 上次记录第一个True的位置，组合起始点
    pend=-1 # 记录最后一个True的位置，组合的终点
    Hw=-1
    def __init__(self,d,start=1):
        """
        :param d:子密钥的位数，这里每个数据有16个子密钥
        :param start: 开始计算的位置，默认为1
        """
        self.M = [False]*d # 初始化全False矩阵，代表每一项都没检查过？
        self.dim = d
        self.pstart = 0
        self.pend = start - 1
        self.Hw = 1  # 默认的汉明重量
        for i in range(start,self.dim):
            self.M[i] = False # start之后的位置的重置为False
        for i in range(0,start): # 从开始检查的start位置都标为True
            self.M[i] = True # start之前的位置变为True
    def HwNext(self):
        if (self.pstart != self.dim - self.Hw): #Hw does not change after update  # 判断此泄露程度为d的联合项是否都计算完毕了
            if (self.pstart == self.pend):#Special case 只有一个True值，即单个组合时候
                self.M[self.pstart + 1] = True # 将其下一项标为True
                for i in range(0,self.pstart+1):
                    self.M[i]=False # 前面的项都标为False
                self.pend = self.pstart + 1 # 各位置往后推进一位，用于下一次计算
                self.pstart = self.pstart + 1 # pend和pstart都改为pstart+1
                while (((self.pend + 1) < self.dim) and (self.M[self.pend + 1] != False)): # 单个组合遍历到底了才触发
                    self.pend = self.pend + 1 # 到最后一位则pend再加一位（下一次判断则不会进入单项的构造）
            else:#pstart != pend # 证明进入了联合项的循环
                self.M[self.pend + 1] = True
                for i in range(0,self.pend-self.pstart):
                    self.M[i] = True
                for i in range(self.pend-self.pstart, self.pend +1):
                    self.M[i] = False
                # Update pstart and pend
                self.pend = (self.pend - self.pstart - 1)
                self.pstart = 0
        else: #00...0011...11  # 当进入一个泄露程度组合项
            if (self.Hw == self.dim):
                self.Hw=self.Hw+1 # Hw加1，即进入下一个d项构造
                return
            for i in range(self.Hw+1):
                self.M[i]=True
            for i in range(self.Hw+1,self.dim):
                self.M[i]=False
            self.pstart = 0
            self.pend = self.Hw
            self.Hw = self.Hw + 1;
    def PrintState(self):
        print("M=")
        for i in range(self.dim):
            if(self.M[i]):
                print("1\t",end = '')
            else:
                print("0\t",end = '')
        print("Hw={0}\n".format(self.Hw))
    def ReturnNo(self):
        No=[0]*self.Hw
        j=0
        for i in range(self.dim):
            if (self.M[i]):
                No[j] = i
                j=j+1
            if (j == self.Hw):
                break
        return No
    def ReturnNum(self):
        Num = 0
        for i in range(self.dim-1,-1,-1):
            if (self.M[i]):
                self.Num =self.Num+1
            if (i != 0):
                Num = Num << 1
        return Num
if __name__ == "__main__":
    h=HWEnum(6)
    while(h.Hw<4):
        h.PrintState()
        h.HwNext()