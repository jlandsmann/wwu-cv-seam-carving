import torch
import torch.nn.functional as fn

def lbp(x):
    #pad image for 3x3 mask size
    x = fn.pad(input=x, pad = [1, 1, 1, 1], mode='constant')
    b=x.shape
    M=b[1]
    N=b[2]
    
    y=x
    #select elements within 3x3 mask 
    # y00  y01  y02
    # y10  y11  y12
    # y20  y21  y22
    
    y00=y[:,0:M-2, 0:N-2]
    y01=y[:,0:M-2, 1:N-1]
    y02=y[:,0:M-2, 2:N  ]
    #     
    y10=y[:,1:M-1, 0:N-2]
    y11=y[:,1:M-1, 1:N-1]
    y12=y[:,1:M-1, 2:N  ]
    #
    y20=y[:,2:M, 0:N-2]
    y21=y[:,2:M, 1:N-1]
    y22=y[:,2:M, 2:N ]      
    
       
    
    # Comparisons 
    # 1 ---------------------------------
    bit=torch.ge(y01,y11)
    tmp=torch.mul(bit,torch.tensor(1)) 
    
    # 2 ---------------------------------
    bit=torch.ge(y02,y11)
    val=torch.mul(bit,torch.tensor(2))
    val=torch.add(val,tmp)    
    
    # 3 ---------------------------------
    bit=torch.ge(y12,y11)
    tmp=torch.mul(bit,torch.tensor(4))
    val=torch.add(val,tmp)
    
    # 4 --------------------------------- 
    bit=torch.ge(y22,y11)
    tmp=torch.mul(bit,torch.tensor(8))   
    val=torch.add(val,tmp)
    
    # 5 ---------------------------------
    bit=torch.ge(y21,y11)
    tmp=torch.mul(bit,torch.tensor(16))   
    val=torch.add(val,tmp)
    
    # 6 ---------------------------------
    bit=torch.ge(y20,y11)
    tmp=torch.mul(bit,torch.tensor(32))   
    val=torch.add(val,tmp)
    
    # 7 ---------------------------------
    bit=torch.ge(y10,y11)
    tmp=torch.mul(bit,torch.tensor(64))   
    val=torch.add(val,tmp)
    
    # 8 ---------------------------------
    bit=torch.ge(y00,y11)
    tmp=torch.mul(bit,torch.tensor(128))   
    val=torch.add(val,tmp)    
    return val.float()