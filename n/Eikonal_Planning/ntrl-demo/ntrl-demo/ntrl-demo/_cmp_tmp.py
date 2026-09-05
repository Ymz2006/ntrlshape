import numpy as np, torch, sys
sys.path.append(".")
sys.path.append("/workspace/baselines/ntrl-demo/dataprocessing")
from scipy.spatial import cKDTree
import speed_sampling_gpu_kdtree_normal_mesh as K
from dataprocessing.preprocess_obj import (load_obj, sample_surface_points,
    tetrahedralize_shape, evaluate_placements, generate_radius_surface_points)
MARGIN, OFFSET = 0.05, 0.005
V,F,names = load_obj("datasets/3dshape/env1.obj")
bmn,bmx=V.min(0),V.max(0); c=0.5*(bmn+bmx); scale=float((bmx-bmn).max()); Vn=(V-c)/scale
mask=np.array(["null" not in str(n).lower() for n in names]); Fe=F[mask]; ns=names[mask]
flip=np.array(["wall" in str(n).lower() for n in ns])
env_pts=sample_surface_points(Vn,Fe,10000).astype(np.float32)
v_obs,n_obs=K.sample_surface_points_with_normals(Vn,Fe,100000,flip_faces=flip,mode="face")
v_obs=v_obs.astype(np.float32); n_obs=n_obs.astype(np.float32)
Vs,Fsh,_=load_obj("datasets/3dshape/rectangle.obj")
sc=0.5*(Vs.min(0)+Vs.max(0)); Vsl=(Vs-sc)/scale
sp=sample_surface_points(Vsl,Fsh,2000).astype(np.float32)
TV,TT,TF=tetrahedralize_shape(Vsl,Fsh)
tetv=torch.tensor(TV[TT],dtype=torch.float32); facev=torch.tensor(TV[TF],dtype=torch.float32)
rp,rb=generate_radius_surface_points(Vsl,Fsh,1000,10)
kd=cKDTree(v_obs); vo=torch.tensor(v_obs); no=torch.tensor(n_obs); spt=torch.tensor(sp)
torch.manual_seed(1)
bb=torch.tensor(v_obs.max(0)); bbn=torch.tensor(v_obs.min(0))
N=3000
P=K.sample_placements(N,bb,bbn,"cpu")
d_kd,n_kd,_=K.shape_obstacle_distance(P,spt,kd,vo,no)
ep=torch.tensor(env_pts)
fr=[];de=[];an=[];ne=[]
for i in range(0,N,250):
    f,d,a,n,_,_=evaluate_placements(P[i:i+250],tetv,facev,ep,rp,rb,"cpu")
    fr.append(f);de.append(d);an.append(a);ne.append(n)
free=torch.cat(fr);d_ex=torch.cat(de);ang=torch.cat(an);n_ex=torch.cat(ne)
band_kd=(d_kd>OFFSET)&(d_kd<MARGIN); band_ex=free&(d_ex>OFFSET)&(d_ex<MARGIN)
print("== %d random placements =="%N)
print("exact-free %d | kd d>0 %d | sign agree %.4f"%(int(free.sum()),int((d_kd>0).sum()),float(((d_kd>0)==free).float().mean())))
print("band_exact %d | band_kd %d | both %d"%(int(band_ex.sum()),int(band_kd.sum()),int((band_ex&band_kd).sum())))
print("kd-band rows exact calls COLLIDING: %.4f"%(float((band_kd&~free).sum())/max(int(band_kd.sum()),1)))
s=band_ex&band_kd
print("== %d rows both accept =="%int(s.sum()))
e=(d_kd[s]-d_ex[s]).abs()
print("dist |err| mean %.5f p99 %.5f (d_ex mean %.4f)"%(e.mean(),np.percentile(e.numpy(),99),d_ex[s].mean()))
sk=np.clip(d_kd[s].numpy()/MARGIN,OFFSET/MARGIN,1); se=np.clip(d_ex[s].numpy()/MARGIN,OFFSET/MARGIN,1)
print("speed(kd) vs speed_dists(obj): |err| mean %.4f p99 %.4f"%(np.abs(sk-se).mean(),np.percentile(np.abs(sk-se),99)))
sb=(se+ang[s].numpy()/np.pi)/2
print("speed(kd) vs speed.npy(obj blend): |err| mean %.4f p99 %.4f ; blend mean %.3f vs kd mean %.3f"%(np.abs(sk-sb).mean(),np.percentile(np.abs(sk-sb),99),sb.mean(),sk.mean()))
def cs(a,b):
    a=a/(a.norm(dim=1,keepdim=True)+1e-12); b=b/(b.norm(dim=1,keepdim=True)+1e-12)
    cc=(a*b).sum(1); return cc.mean().item(), float(np.degrees(np.arccos(cc.clamp(-1,1).numpy())).mean())
print("normal full6 cos %.4f (%.1f deg) | trans cos %.4f (%.1f deg) | rot cos %.4f (%.1f deg)"%(cs(n_kd[s],n_ex[s])+cs(n_kd[s][:,:3],n_ex[s][:,:3])+cs(n_kd[s][:,3:],n_ex[s][:,3:])))
print("angles rad: min %.3f max %.3f mean %.3f -> speed_angles mean %.3f"%(ang[s].min(),ang[s].max(),ang[s].mean(),(ang[s]/np.pi).mean()))
