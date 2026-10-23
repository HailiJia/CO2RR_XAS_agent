"""Enlarged H-visible local views from frozen audit arrays; no name-based labels."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.contact_checks import minimum_image_vectors
BASE=ROOT/'study/extensions_v2/chemical_audit'
COLORS={'Cu':'#4B92C3','Au':'#DDB644','C':'#333333','O':'#D84E47','H':'#FAFAFA'}

def main():
    rows=json.loads((BASE/'geometry_views_data.json').read_text());out=BASE/'geometry_views';out.mkdir(exist_ok=True)
    selected=[r for r in rows if r['stage']=='relaxed' and (r['configuration'].startswith(('CHO/','COH/')))]
    manifest=[]
    for n,r in enumerate(selected):
        s=r['structure'];p=np.array(s['positions']);cell=np.array(s['cell']);el=np.array(s['atoms']);ads=np.flatnonzero(np.isin(el,['C','H','O']))
        f,axs=plt.subplots(1,max(1,len(r['audit']['components'])),figsize=(4.3*max(1,len(r['audit']['components'])),4.2),squeeze=False)
        for ax,m in zip(axs.flat,r['audit']['components']):
            ids=np.array(m['atom_indices'])-1;center=p[ids].mean(axis=0);q=minimum_image_vectors(p-center,cell,(True,True,False))
            # Oblique projection preserves height and shows all H explicitly.
            xy=np.column_stack([q[:,0]-.35*q[:,1],q[:,2]+.28*q[:,1]])
            nearby=np.flatnonzero((np.linalg.norm(q,axis=1)<4)&np.isin(el,['Cu','Au']));draw=np.r_[nearby,ids]
            for i,j in m['bonds']:
                a,b=int(i)-1,int(j)-1;ax.plot(*xy[[a,b]].T,c='#444444',lw=2,zorder=2)
            for i in draw:
                ax.scatter(*xy[i],s=190 if el[i]=='H' else 240 if el[i] in ['C','O'] else 130,c=COLORS[el[i]],edgecolor='#222222',linewidth=1,zorder=4 if el[i]=='H' else 3)
                if i in ids:ax.annotate(f'{el[i]}{i+1}',xy[i],xytext=(7,7),textcoords='offset points',fontsize=9,fontweight='bold',zorder=5)
            ax.set_aspect('equal');ax.set_xlim(-3.5,3.5);ax.set_ylim(-3,3.5);ax.set_title(m['species']+'; indices are 1-based',fontsize=10);ax.set_xlabel('Oblique local coordinate (Å)');ax.set_ylabel('Height + oblique coordinate (Å)')
        f.suptitle(r['configuration']+' — relaxed connectivity',fontsize=10);f.tight_layout()
        stem=f'{n:02d}_'+r['configuration'].replace('/','__')
        for ext in ['pdf','png']:f.savefig(out/(stem+'.'+ext),dpi=240,bbox_inches='tight')
        plt.close(f);manifest.append({'configuration':r['configuration'],'geometry_path':r['path'],'figure':stem,'meaning':'full-graph components; H white with black border; covalent bonds dark; local metals Cu blue/Au gold'})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
