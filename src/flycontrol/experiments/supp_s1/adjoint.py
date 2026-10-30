from flycontrol.experiments.supp_s1 import native as old
import torch
from larva.ant_transfer.reward_transfer_v2.policy import segment_product
def adjoint(graph,x):return segment_product(x,graph.b_source,graph.b_weight,graph.b_lengths)
def output_gradient(core,h):
 with torch.enable_grad():
  v=h.detach().requires_grad_(True);cmd=torch.tanh(old.mean_from_rates(core,torch.tanh(v)))
  return torch.stack([torch.autograd.grad(cmd[:,i].sum(),v,retain_graph=i<5)[0] for i in range(6)],1).detach()
def propagate(core,h,obs,start,end):
 for t in range(start+1,end+1):
  dr=core.fixed_graph(torch.tanh(h));dr[:,:2279]+=torch.nn.functional.linear(obs[:,t,-172:].to(h.dtype),core.input_proj.weight.to(h.dtype),core.input_proj.bias.to(h.dtype));h=.5*h+.5*torch.tanh(dr)
 x=torch.tanh(h)[:,-1290:];ln,lin=core.output_layer
 z=torch.nn.functional.layer_norm(x,(1290,),ln.weight.to(h.dtype),ln.bias.to(h.dtype),ln.eps)
 raw=torch.nn.functional.linear(z,lin.weight.to(h.dtype),lin.bias.to(h.dtype))
 return torch.tanh(3*torch.tanh(raw/3))
