import numpy as np 
class binary_arithmetic_coding:
    def __init__(self) -> None:
        self.low = [0] * 256
        self.high = [1] * 256
        self.bitstream = []
        self.sequences = []
        
    def dec2bin(self, x): 
        x -= int(x) 
        bins = [] 
        while len(bins) < 256: 
            x *= 2 
            bins.append(1 if x>=1. else 0) 
            x -= int(x) 
        return bins
    
    def bin2dec(self, b): 
        d = 0 
        for i, x in enumerate(b): 
            d += 2**(-i-1)*x 
        return d
    
    def encode(self, p, sym):#p预测为1的概率
        if p >= 0.5:
            mps = 1 
            lps = 0
            pmps = p
        else:
            mps = 0 
            lps = 1
            pmps = 1 - p
        low = self.bin2dec(self.low)
        high = self.bin2dec(self.high)
        l = high - low
        
        if mps == sym:
            high = low + l * pmps
            newlow = self.low
            newhigh = self.dec2bin(high)
        else: 
            low  = low + l * pmps
            newlow = self.dec2bin(low)
            newhigh = self.high 
         
        for i, (l_b, h_b) in enumerate(zip(newlow, newhigh)):
            if l_b != h_b:
                break
            self.bitstream += [l_b]   
        self.high = newhigh[i:] + [0]*(i)
        self.low  = newlow[i:]  + [0]*(i)
        
    def end(self):
        for i, b in enumerate(self.low):
            if i != 0 and b == 0:
                break
            self.bitstream += [b]
        self.bitstream += [1]
        
             
    def decode(self, p):
        if p >= 0.5:
            mps = 1 
            lps = 0
            pmps = p
        else:
            mps = 0 
            lps = 1
            pmps = 1 - p
        low = self.bin2dec(self.low)
        high = self.bin2dec(self.high)
        l = high - low
        limit = low + l * pmps
        
        value = self.bin2dec(self.bitstream[:256]) 
        if value <= limit:
            self.sequences.append(mps) 
            newlow = self.low
            newhigh = self.dec2bin(limit)
        else:
            self.sequences.append(lps) 
            newlow = self.dec2bin(limit)
            newhigh =self.high 
        for i, (l_b, h_b) in enumerate(zip(newlow, newhigh)):
            if l_b != h_b:
                break 
        self.high = newhigh[i:] + [0]*(i)
        self.low  = newlow[i:]  + [0]*(i) 
        self.bitstream = self.bitstream[i:]   
            
    def reset(self):
        self.low = [0] * 256
        self.high = [1] * 256
        
if __name__ == "__main__":
    coder = binary_arithmetic_coding() 
    P = np.random.random(160)
    S = np.random.randint(low=0, high=2, size=160)
    P[S==1] = 2e-6
    P[S==0] = 1-2e-6
    for i, (p, s) in enumerate(zip(P,S)): 
        print(i) 
        coder.encode(p, s)    
        # print(coder.low)
        # print(coder.high) 
        # print('-------------------------------------------------------------------------------------------------')
    coder.end()
    print(len(coder.bitstream))
    coder.reset()
    print()
    for p in P: 
        coder.decode(p) 
        # print(coder.low)
        # print(coder.high) 
        # print('-------------------------------------------------------------------------------------------------')
    res = coder.sequences == S 
    # print(res)
    print(np.sum(res))
    pass
