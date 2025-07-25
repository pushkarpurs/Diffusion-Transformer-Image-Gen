import torch
import torch.nn as nn

def get_time_embedding(time_steps, temb_dim=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if time_steps.dim() == 2 and time_steps.shape[1] == 1:
        time_steps = time_steps.squeeze(1)
    half_dim = temb_dim // 2
    exponents = torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
    scale_factors = 10000 ** exponents
    scaled_time = time_steps[:, None] / scale_factors 
    time_embedding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1) 
    return time_embedding

def get_positional_embedding(seq_len, dim=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half_dim = dim // 2
    exponents = torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim
    scale_factors = 10000 ** exponents
    positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    scaled_positions = positions / scale_factors
    pos_embedding = torch.cat([torch.sin(scaled_positions), torch.cos(scaled_positions)], dim=-1)
    return pos_embedding

class convdownone(nn.Module):
    def __init__(self):
        super().__init__()
        self.upchannels=nn.Conv2d(3,4,kernel_size=3,stride=1, padding=1)
        self.silu=nn.SiLU()
    def forward(self,x):
        x=self.upchannels(x)
        x=self.silu(x)
        return x

class convdowntwo(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsize=nn.Conv2d(4,4,kernel_size=4,stride=2, padding=1)
        self.silu=nn.SiLU()
    def forward(self,x):
        x=self.downsize(x)
        x=self.silu(x)
        return x

class convupone(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsize=nn.ConvTranspose2d(8,8,4,2,1)
        self.downchannels=nn.Conv2d(8,4,3,1,1)
        self.silu=nn.SiLU()
    def forward(self,x,y):
        x=torch.cat((x,y),1)
        x=self.upsize(x)
        x=self.downchannels(x)
        x=self.silu(x)
        return x

class convuptwo(nn.Module):
    def __init__(self):
        super().__init__()
        self.downchannels=nn.Conv2d(8,3,3,1,1)
        self.silu=nn.SiLU()
    def forward(self,x,y):
        x=torch.cat((x,y),1)
        x=self.downchannels(x)
        x=self.silu(x)
        return x

class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(4*4*4, 768)
        self.silu=nn.SiLU()
    def forward(self, x):
        B,C,H,W = x.shape
        P=4
        x=x.unfold(2, P, P).unfold(3, P, P)
        x=x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * P * P)
        x=self.proj(x)
        x=self.silu(x)
        pos_embedding=get_positional_embedding(x.shape[1])
        x=x+pos_embedding.unsqueeze(0)
        return x
    
class LinearReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm=nn.LayerNorm(768)
        self.proj=nn.Linear(768, 4*4*4)
        self.silu=nn.SiLU()
    def forward(self, x):
        B,C,H =x.shape
        x=self.layernorm(x)
        x=self.proj(x)
        x=self.silu(x)
        x=x.view(B, 8, 8, 4, 4, 4)
        x=x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x=x.view(B, 4, 32, 32)
        return x

class DiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1=nn.LayerNorm(768)
        self.timestep1=nn.Sequential(
            nn.Linear(768, 768*2),
            nn.SiLU(),
            nn.Linear(768*2, 768*2)
        )
        self.attention=nn.MultiheadAttention(768, 12, batch_first=True)
        self.timestep2=nn.Sequential(
            nn.Linear(768, 768),
            nn.SiLU(),
        )
        self.norm2=nn.LayerNorm(768)
        self.timestep3=nn.Sequential(
            nn.Linear(768, 768*2),
            nn.SiLU(),
            nn.Linear(768*2, 768*2)
        )
        self.pointwise=nn.Sequential(
            nn.Linear(768, 768*4),
            nn.SiLU(),
            nn.Linear(768*4, 768)
        )
        self.timestep4=nn.Sequential(
            nn.Linear(768, 768),
            nn.SiLU(),
        )
    def forward(self, x, tx): 
        skip1=x
        x=self.norm1(x)
        t=get_time_embedding(tx)
        t1=self.timestep1(t)
        gamma1,beta1=t1.chunk(2, dim=-1)
        x=x*gamma1.unsqueeze(1)+beta1.unsqueeze(1)
        x=self.attention(x, x, x)[0]
        t2=self.timestep2(t)
        x=x*t2.unsqueeze(1)
        x=x+skip1
        skip2=x
        x=self.norm2(x)
        t3=self.timestep3(t)
        gamma3,beta3=t3.chunk(2, dim=-1)
        x=x*gamma3.unsqueeze(1)+beta3.unsqueeze(1)
        x=self.pointwise(x)
        t4=self.timestep4(t)
        x=x*t4.unsqueeze(1)
        x=x+skip2
        return x
    
class DiT(nn.Module):
    def __init__(self,ditblocks=5):
        super().__init__()
        self.d1=convdownone()
        self.d2=convdowntwo()
        self.d3=convdowntwo()
        self.d4=convdowntwo()
        self.patch=Patchify()
        self.ditblock=nn.ModuleList([DiTBlock() for _ in range(ditblocks)])
        self.linear=LinearReshape()
        self.u1=convupone()
        self.u3=convupone()
        self.u4=convupone()
        self.u2=convuptwo()
        self.ditblocks=ditblocks
    def forward(self, x, t):
        a=[]
        x=self.d1(x)
        a.append(x)
        x=self.d2(x)
        a.append(x)
        x=self.d3(x)
        a.append(x)
        x=self.d4(x)
        a.append(x)
        x=self.patch(x)
        for i in range(self.ditblocks):
            x=self.ditblock[i](x,t)
        x=self.linear(x)
        x=self.u1(x,a.pop())
        x=self.u3(x,a.pop())
        x=self.u4(x,a.pop())
        x=self.u2(x,a.pop())
        return x