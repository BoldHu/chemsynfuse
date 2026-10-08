"""Bidirectional Mamba and molecular reconstruction/contrastive objectives."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .constants import PAD, BOS, EOS

class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (T, B, H)
        # self.pe: (1, max_len, d_model)
        # self.pe[:, :T, :]: (1, T, d_model)
        # permute(1, 0, 2): (T, 1, d_model)
        pos_enc = self.pe[:, :x.size(0), :].permute(1, 0, 2)  # => (T, 1, H)
        x = x + pos_enc
        return self.dropout(x)

def reverse_valid(x, lengths):
    """Reverse [0,L), including BOS/EOS, leave right-hand PAD positions in place."""
    if x.ndim<2 or lengths.shape!=(x.shape[0],) or (lengths<0).any() or (lengths>x.shape[1]).any():
        raise ValueError('Invalid lengths')
    pos=torch.arange(x.shape[1],device=x.device).expand(x.shape[0],-1)
    indices=torch.where(pos<lengths[:,None],lengths[:,None]-1-pos,pos)
    indices=indices.reshape(*indices.shape,*([1]*(x.ndim-2))).expand_as(x)
    return x.gather(1,indices)

def validate_tokens(ids):
    if ids.ndim!=2:raise ValueError('Expected batch-first token IDs')
    valid=ids!=PAD;lengths=valid.sum(1)
    expected=torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None]
    if (lengths<3).any() or not torch.equal(valid,expected):raise ValueError('Nonempty molecule with right padding required')
    if (ids[:,0]!=BOS).any() or (ids.gather(1,(lengths-1)[:,None]).flatten()!=EOS).any():
        raise ValueError('BOS/EOS boundary contract violated')
    return valid,lengths

class BidirectionalBlock(nn.Module):
    def __init__(self, hidden, state=16, share_directions=True):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            import selective_scan_cuda
        except ImportError as exc:
            raise RuntimeError('Real mamba_ssm CUDA backend required; no substitute') from exc
        self.forward_scan=Mamba(d_model=hidden,d_state=state,d_conv=4,expand=2,use_fast_path=True)
        self.backward_scan=None if share_directions else Mamba(d_model=hidden,d_state=state,d_conv=4,expand=2,use_fast_path=True)
        self.mix=nn.Linear(2*hidden,hidden)
        self.norm=nn.LayerNorm(hidden)

    def forward(self,x,valid,lengths):
        if not x.is_cuda:raise RuntimeError('Verified CUDA Mamba path only')
        forward=self.forward_scan(x)
        scanner=self.forward_scan if self.backward_scan is None else self.backward_scan
        backward=reverse_valid(scanner(reverse_valid(x,lengths)),lengths)
        result=self.norm(x+self.mix(torch.cat([forward,backward],dim=-1)))
        return result.masked_fill(~valid[:,:,None],0.)

def pool_hidden(hidden,ids):
    valid=(ids!=PAD)&(ids!=BOS)&(ids!=EOS)
    mean=(hidden*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
    maximum=hidden.masked_fill(~valid[:,:,None],-torch.inf).max(1).values
    lengths=(ids!=PAD).sum(1)
    last=hidden.gather(1,(lengths-1)[:,None,None].expand(-1,1,hidden.shape[-1])).squeeze(1)
    return torch.cat([mean,maximum,hidden[:,0],last],dim=-1)

class MolecularPretrainer(nn.Module):
    def __init__(self,vocab_size,hidden=64,layers=2,state=16,share_directions=True,
                 encoder_type='bimamba',dropout=.1,projection_dim=32):
        super().__init__()
        self.config=dict(vocab_size=vocab_size,hidden=hidden,layers=layers,state=state,
                         share_directions=share_directions,encoder_type=encoder_type,
                         dropout=dropout,projection_dim=projection_dim)
        if encoder_type != 'bimamba':
            raise ValueError('Only the ChemSynFuse bidirectional Mamba encoder is included')
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.decoder_pe = PositionalEncoding(hidden, dropout)
        layer = nn.TransformerDecoderLayer(hidden, 4, hidden)
        self.decoder = nn.TransformerDecoder(layer, 1, norm=nn.LayerNorm(hidden))
        self.output = nn.Linear(hidden, vocab_size)
        self.blocks = nn.ModuleList([BidirectionalBlock(hidden, state, share_directions) for _ in range(layers)])
        self.projection=nn.Sequential(nn.Linear(4*hidden,hidden),nn.GELU(),nn.Linear(hidden,projection_dim))

    def states(self,ids):
        valid,lengths=validate_tokens(ids)
        x=self.embed(ids).masked_fill(~valid[:,:,None],0.)
        for block in self.blocks:
            x = block(x, valid, lengths)
        return x

    def encode(self,src):
        """Old notebook API: (T,B) IDs -> (B,4H) Tensor, without detaching gradients."""
        if src.ndim!=2:raise ValueError('encode expects T,B')
        if src.shape[1]==0:return self.embed.weight.new_empty((0,4*self.config['hidden']))
        ids=src.transpose(0,1)
        return pool_hidden(self.states(ids),ids)

    def decode(self,memory,source_ids,decoder_input):
        tgt=self.decoder_pe(self.embed(decoder_input).transpose(0,1))
        causal=torch.triu(torch.ones(tgt.shape[0],tgt.shape[0],dtype=torch.bool,device=tgt.device),diagonal=1)
        result=self.decoder(tgt,memory.transpose(0,1),tgt_mask=causal,
                            tgt_key_padding_mask=decoder_input==PAD,memory_key_padding_mask=source_ids==PAD)
        return self.output(result).transpose(0,1)

    def forward(self,a,b,canonical_target):
        memory=self.states(a);other=self.states(b)
        logits=self.decode(memory,a,canonical_target[:,:-1])
        return logits,self.projection(pool_hidden(memory,a)),self.projection(pool_hidden(other,b))

    @torch.no_grad()
    def generate(self,source_ids,max_length):
        memory=self.states(source_ids)
        seq=torch.full((source_ids.shape[0],1),BOS,dtype=torch.long,device=source_ids.device)
        ended=torch.zeros(source_ids.shape[0],dtype=torch.bool,device=source_ids.device)
        for _ in range(max_length-1):
            logits=self.decode(memory,source_ids,seq)[:,-1]
            logits[:,[PAD,BOS,4]]=-torch.inf
            nxt=logits.argmax(1);nxt=torch.where(ended,torch.full_like(nxt,EOS),nxt)
            seq=torch.cat([seq,nxt[:,None]],1);ended|=nxt==EOS
            if ended.all():break
        return seq

def molecular_contrastive(a,b,identities,temperature=.1):
    """Symmetric multi-positive objective; identical identities are never negatives.

    All 2B anchors have every other same-identity view as a positive. No self-pair.
    Positive log probabilities are averaged per anchor, then over both directions.
    """
    if temperature<=0 or a.shape!=b.shape or len(identities)!=a.shape[0]:raise ValueError('Invalid contrastive batch')
    unique={v:i for i,v in enumerate(sorted(set(identities)))}
    labels=torch.tensor([unique[v] for v in identities]*2,device=a.device)
    z=F.normalize(torch.cat([a,b],0).float(),dim=-1)
    eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
    positive=(labels[:,None]==labels[None,:])&~eye
    negative=(labels[:,None]!=labels[None,:])&~eye
    logits=(z@z.T)/temperature
    logits=logits.masked_fill(eye,-torch.inf)
    logp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    loss=-(logp.masked_fill(~positive,0.).sum(1)/positive.sum(1)).mean()
    return loss,dict(positive_pairs=int(positive.sum()),negative_pairs=int(negative.sum()),same_identity_negative_pairs=0)
