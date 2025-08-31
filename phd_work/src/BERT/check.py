import torch 

def random_mask( x: torch.Tensor, probability: float, mask_v: float = -10) -> torch.Tensor:
    # рызыгрывать вероятности а не индексы. 
    device = x.device

    token_mask = (x[:,:,0:1] != mask_v).to(device)  # [batch, maxlen]
    token_mask[:, 0] = False  # исключаем первый токен

    index = torch.sum(token_mask, dim=1)[:,0] # one dim -> batch
    token_mask[torch.arange(index.size(0)), index] = False
    
    probability_tensor = torch.rand_like(x[:,:,0:1]) # batch, len, 1
    probability_tensor = probability_tensor*token_mask # zero in supportive tokens
    token_mask = torch.where(probability_tensor>(1-probability), 1, 0).to(device)
    # to shape -> batch, len, 6
    token_mask = torch.repeat_interleave(token_mask, 6, dim=2).to(device).to(torch.bool)
    return token_mask

x=torch.randn(3,21,6)
mask_v = -10
x[0,10:]=mask_v
x[1,10:]=mask_v
x[2,10:]=mask_v
token_mask = random_mask(x, 0.3, mask_v)
print(token_mask, token_mask.shape)

