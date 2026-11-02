"""Render the archived numeric reference comparisons without changing processing.

Run after reference_benchmark/process.py when recomputing the numeric comparisons.
Consumes exact plotted arrays; no shifts, widths, fits, or values change here.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
METHODS=['fdmnes_green','fdmnes_fdm','feff','vasp_pbe','vasp_gw']
LABELS=['FDMNES green','FDMNES fdm','FEFF','VASP PBE','VASP GW*']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
TITLES={'native':'Native energy axes','common':'Cu-calibrated offsets transferred to the oxides',
        'shape_aligned':'Separate alignment for each calculation',
        'sigma1_sensitivity':'Additional Gaussian broadening: σ = 1 eV sensitivity'}

def main():
    d=pd.read_csv(ROOT/'reference_benchmark/plotted_arrays.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    for view,title in TITLES.items():
        f,axes=plt.subplots(3,5,figsize=(13.8,7.8),sharex=True,sharey='row')
        for i,mat in enumerate(['Cu','Cu2O','CuO']):
            row=d[(d.material==mat)&(d.view==view)]
            ymin=-.25; ymax=1.10
            for j,(method,color) in enumerate(zip(METHODS,COLORS)):
                ax=axes[i,j];q=row[row.method==method]
                if q.empty:
                    ax.set_axis_off()
                    continue
                ax.plot(q.energy_eV,q.experiment/q.experiment.max(),c='#171717',lw=1.6)
                ax.plot(q.energy_eV,q.calculation/q.calculation.max(),c=color,lw=1.45,
                        ls='--' if mat=='CuO' and method=='fdmnes_fdm' else '-')
                ax.set(xlim=(8980,9020),ylim=(ymin,ymax))
                ax.set_xticks([8980,9000,9020]);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
                ax.tick_params(labelsize=9)
                if i==0:ax.set_title(LABELS[j],fontweight='bold',color=color,pad=12,fontsize=10.7)
                if j==0:ax.set_ylabel({'Cu':'Cu','Cu2O':'Cu₂O','CuO':'CuO'}[mat],fontweight='bold',fontsize=12)
                if i==2:ax.set_xlabel('Photon energy (eV)',fontsize=9.5)
        f.suptitle(title,fontsize=16,y=.982)
        f.supylabel('Intensity (a.u.)',x=.012,fontsize=11)
        f.text(.5,.025,'Black: experiment. Colors: calculations. Traces scaled to unit maximum. Blank cells: missing data or insufficient overlap.\n'
               '*“GW” is an unverified directory label. Dashed CuO FDM: single site; incomplete final-state processing.',
               ha='center',fontsize=9.3,linespacing=1.5)
        f.subplots_adjust(left=.07,right=.99,top=.88,bottom=.135,hspace=.25,wspace=.18)
        for ext in ['pdf','svg','png']:
            f.savefig(ROOT/'figures'/f'SI_reference_grid_{view}.{ext}',dpi=450)
        plt.close(f)
    print('Rendered four reference grids; numeric arrays unchanged')
if __name__=='__main__':main()
