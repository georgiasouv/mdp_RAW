import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as pat

ANN='/cifs/Shares/Raw_Bayer_Datasets/NOD/annotations_raw/Nikon/raw_new_Nikon750_val.json'
PACK='/scratch/INC1526354/nod/packed/Nikon'
OUT='/networkhome/WMGDS/souval_g/raw-mdp'

d=json.load(open(ANN))
cats={c['id']:c['name'] for c in d['categories']}
col={'person':'red','bicycle':'lime','car':'cyan'}

# group annotations by image
from collections import defaultdict
by_img=defaultdict(list)
for a in d['annotations']:
    by_img[a['image_id']].append(a)

# pick 2 images (different from DSC_0830) that have boxes and exist as .npy
import os
stems=[i['file_name'][:-4] for i in d['images']]
picks=[]
for s in stems:
    if s=='DSC_0830': continue
    if os.path.exists(f'{PACK}/{s}.npy') and by_img[s]:
        picks.append(s)
    if len(picks)==2: break

for stem in picks:
    arr=np.load(f'{PACK}/{stem}.npy')
    g=arr[1].astype(np.float32); g=np.clip(g/np.percentile(g,99)*255,0,255).astype(np.uint8)
    anns=by_img[stem]
    fig,ax=plt.subplots(figsize=(12,8)); ax.imshow(g,cmap='gray')
    for a in anns:
        x,y,w,h=a['bbox']; name=cats[a['category_id']]
        ax.add_patch(pat.Rectangle((x/2,y/2),w/2,h/2,fill=False,edgecolor=col.get(name,'yellow'),lw=2))
        ax.text(x/2,y/2-5,name,color=col.get(name,'yellow'),fontsize=10,weight='bold')
    ax.set_title(f'{stem}  ({len(anns)} boxes)')
    plt.savefig(f'{OUT}/check_{stem}.png',dpi=80,bbox_inches='tight')
    print(f'saved check_{stem}.png  boxes:', [(cats[a['category_id']],a['bbox']) for a in anns])