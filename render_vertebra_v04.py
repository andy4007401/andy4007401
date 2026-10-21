from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out=Path(__file__).resolve().parent.parent/'outputs'/'vertebra_v04'
m=np.load(out/'mesh_preview_data.npz')
v,f=m['vertices'],m['faces']
n=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
vn=np.zeros_like(v)
for i in range(3): np.add.at(vn,f[:,i],n)
vn/=np.maximum(np.linalg.norm(vn,axis=1)[:,None],1e-12)
fn=vn[f].mean(axis=1)
fn/=np.maximum(np.linalg.norm(fn,axis=1)[:,None],1e-12)
def panel(size,right,up,depth,title):
    right,up,depth=[np.asarray(a,dtype=float) for a in (right,up,depth)]
    for a in (right,up,depth): a/=np.linalg.norm(a)
    basis=np.stack([right,up,depth],axis=1)
    p=v@basis
    scale=min((size[0]-90)/np.ptp(p[:,0]),(size[1]-100)/np.ptp(p[:,1]))
    p[:,:2]-=(p[:,:2].max(axis=0)+p[:,:2].min(axis=0))/2
    p[:,0]=p[:,0]*scale+size[0]/2
    p[:,1]=-p[:,1]*scale+size[1]/2+10
    im=Image.new('RGB',size,(40,44,49)); d=ImageDraw.Draw(im)
    light=np.array([-.45,.65,.75]);light/=np.linalg.norm(light)
    nc=fn@basis
    intensity=.48+.50*np.maximum(nc@light,0)
    base=np.array([235,222,194])
    colors=np.clip(intensity[:,None]*base,0,255).astype(np.uint8)
    for j in np.argsort(p[f,2].mean(axis=1)):
        if nc[j,2]<-.1: continue
        d.polygon([tuple(q) for q in p[f[j],:2]],fill=tuple(colors[j]))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    d.text((24,18),title,font=font,fill=(217,223,230))
    return im
im=Image.new('RGB',(1800,1400),(30,33,38))
im.paste(panel((850,850),(1,0,0),(0,0,1),(0,1,0),'TOP | rim, canal and paired facets'),(0,0))
im.paste(panel((850,550),(0,0,-1),(0,1,0),(1,0,0),'SIDE | curved spinous blade'),(0,850))
depth=np.array([.9,1.35,1.1]);depth/=np.linalg.norm(depth)
right=np.cross((0,1,0),depth);right/=np.linalg.norm(right)
up=np.cross(depth,right)
im.paste(panel((950,1400),right,up,depth,'THORACIC VERTEBRA | concept v04'),(850,0))
im.save(out/'Vertebra_Curves_v04_Preview.png')
print(out/'Vertebra_Curves_v04_Preview.png')
