"""Original scientific schematics and figures, drawn from retained simulation outputs."""
from pathlib import Path
import json
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle, Polygon
from physics import C, forward, gas_properties, DETUNING

ROOT=Path(__file__).resolve().parent; R=ROOT/'results'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
S=json.loads((R/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':11.5,
 'axes.labelsize':10.5,'legend.fontsize':9.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
 'savefig.facecolor':'white','figure.facecolor':'white','axes.grid':False})
BLUE='#245E91'; TEAL='#16817A'; RED='#BB4A43'; GOLD='#B08022'; PURPLE='#75549B'
COLORS=[GOLD,RED,BLUE,PURPLE,TEAL]
METHODS=list(S['main'])
SHORT=['Nominal T, p','Plug-in T, p','Joint uncertainty','In-phase only','Single frequency']

def save(fig,name):
    buffer=io.BytesIO()
    fig.savefig(buffer,format='png',dpi=320,bbox_inches='tight',pad_inches=.08)
    from PIL import Image
    with Image.open(io.BytesIO(buffer.getvalue())) as check:check.verify()
    temporary=F/(name+'.tmp')
    temporary.write_bytes(buffer.getvalue());temporary.replace(F/(name+'.png'))
    fig.savefig(F/(name+'.svg'),bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
def box(ax,xy,w,h,text,color=BLUE,fs=10):
    ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.015,rounding_size=0.035',
       facecolor=color+'12',edgecolor=color,lw=1.3))
    artist=ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=fs,color='#202830')
    ax.figure.canvas.draw()
    renderer=ax.figure.canvas.get_renderer()
    available=abs(ax.transData.transform((xy[0]+w,xy[1]))[0]-ax.transData.transform(xy)[0])*.95
    while artist.get_window_extent(renderer).width>available and fs>8.1:
        fs-=.25;artist.set_fontsize(fs);ax.figure.canvas.draw()
def arrow(ax,a,b,color='#59636E',label=None,offset=(0,.035)):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=color))
    if label:ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,fontsize=8.7,ha='center',va='center',color=color)
def panel(ax,label): ax.text(-.12,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=12)

# Proposed apparatus: all paths are schematic; no hardware implementation is claimed.
fig,ax=plt.subplots(figsize=(8.0,4.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,(.015,.72),.21,.16,'Pump: 2121.8 nm\nPower modulation',BLUE)
box(ax,(.015,.46),.21,.16,'Probe: 1550 nm\nCoherent phase\nreadout',TEAL,9.5)
box(ax,(.275,.60),.15,.17,'Optical\ncombiner',BLUE)
arrow(ax,(.225,.80),(.275,.71),BLUE);arrow(ax,(.225,.54),(.275,.66),TEAL)
box(ax,(.47,.57),.24,.22,'Hollow-core fibre\nH₂ / N₂ gas cell\n7 m; core radius 15 µm',TEAL)
arrow(ax,(.425,.685),(.47,.685),BLUE)
box(ax,(.49,.88),.20,.09,'Gas inlet / outlet',TEAL,9.5);arrow(ax,(.59,.88),(.59,.79),TEAL)
box(ax,(.485,.37),.21,.11,'Auxiliary T, p\nwith uncertainty',GOLD,9.5)
arrow(ax,(.59,.57),(.59,.48),GOLD)
box(ax,(.765,.59),.22,.18,'Interferometer\nBalanced detector\nComplex lock-in (I + iQ)',BLUE,9.5)
arrow(ax,(.71,.685),(.765,.685),TEAL)
ax.plot([.225,.25,.25,.75,.75,.765],[.51,.51,.30,.30,.64,.64],color=TEAL,lw=1.3,ls='--')
ax.text(.29,.32,'Probe reference arm',fontsize=8.8,color=TEAL)
box(ax,(.765,.11),.22,.22,'Joint physics inversion\nConcentration + interval\nDiscrepancy / ambiguity\nreview flag',BLUE,9.1)
arrow(ax,(.875,.59),(.875,.33),BLUE,'49-point\nscans',(.064,0))
arrow(ax,(.695,.415),(.765,.25),GOLD)
# Antiresonant cross-section icon; wall boundary is an idealisation.
center=(.18,.17); aspect=8.0/4.8
ax.add_patch(Ellipse(center,.16,.16*aspect,fill=False,color='#8996A3',lw=1.5))
for angle in np.linspace(0,2*np.pi,6,endpoint=False):
    pos=np.array(center)+.058*np.array([np.cos(angle),aspect*np.sin(angle)])
    ax.add_patch(Ellipse(pos,.044,.044*aspect,fill=False,color='#8996A3',lw=1.1))
ax.add_patch(Ellipse(center,.077,.077*aspect,color=TEAL,alpha=.12));ax.add_patch(Ellipse(center,.033,.033*aspect,color=BLUE,alpha=.25))
ax.annotate('Gas-filled core\nGaussian optical modes',xy=center,xytext=(.36,.12),fontsize=9.2,
 arrowprops={'arrowstyle':'-','lw':1},ha='left',va='center')
ax.text(.015,.015,'Proposed measurement configuration; schematic, not to scale.',fontsize=9,color='#59636E')
save(fig,'fig01_apparatus')

# Data and inference flow, showing the distinct truth and inverse discretisations.
fig,ax=plt.subplots(figsize=(8,4.6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,(.02,.79),.27,.17,'Frequency design\n10 triplets; 18 design states\nWorst marginal uncertainty',BLUE,9.7)
box(ax,(.37,.79),.27,.17,'Synthetic truth\n240 radial finite volumes\nT, p, gain, relaxation varied',TEAL,9.7)
box(ax,(.71,.79),.27,.17,'Complex spectra +\nuncertain auxiliaries\nCorrelated noise; drift',GOLD,9.7)
arrow(ax,(.29,.875),(.37,.875));arrow(ax,(.64,.875),(.71,.875))
box(ax,(.71,.48),.27,.17,'Whiten noise\nProject linear backgrounds\nKeep I and Q quadratures',BLUE,9.7)
arrow(ax,(.845,.79),(.845,.65))
box(ax,(.37,.48),.27,.17,'Constrained joint inversion\n40 Bessel modes\nFive concentration starts',BLUE,9.7)
arrow(ax,(.71,.565),(.64,.565))
box(ax,(.02,.48),.27,.17,'Independent calibration\n120 cases: interval scale\nand residual threshold',GOLD,9.7)
arrow(ax,(.155,.79),(.155,.65));arrow(ax,(.29,.565),(.37,.565),GOLD)
box(ax,(.37,.15),.27,.18,'Estimate + local interval\nAlternative-branch audit\nResidual and width gate',BLUE,9.7)
arrow(ax,(.505,.48),(.505,.33))
box(ax,(.02,.15),.27,.18,'Reportable estimate\nFinite interval and\nacceptable model fit',TEAL,9.7)
box(ax,(.71,.15),.27,.18,'Review flag\nWide interval, ambiguity\nor model discrepancy',RED,9.7)
arrow(ax,(.37,.24),(.29,.24),TEAL);arrow(ax,(.64,.24),(.71,.24),RED)
ax.text(.5,.02,'Independent test: 320 cases  •  four challenge scenarios: 64 cases each',ha='center',fontsize=9.5)
save(fig,'fig02_workflow')

# Non-monotonic concentration response and frequency-dependent phase signatures.
z=np.load(R/'concentration_response.npz'); fig,axs=plt.subplots(1,2,figsize=(8,3.65),layout='constrained')
for k,f in enumerate(z['frequencies_Hz']):
    axs[0].plot(100*z['x'],abs(z['complex_phase'][:,k])*1000,label=f'{f/1000:g} kHz',color=[BLUE,TEAL,RED][k])
    axs[1].plot(100*z['x'],np.angle(-z['complex_phase'][:,k],deg=True),color=[BLUE,TEAL,RED][k])
for x in [float(z['alias_low']),float(z['alias_high'])]:
    axs[0].axvline(100*x,color='#6C7580',ls='--',lw=1)
target=abs(forward([float(z['alias_low']),300,100000,0,0,0],z['frequencies_Hz'][:1])[0,24])*1000
axs[0].scatter(100*np.array([z['alias_low'],z['alias_high']]),[target,target],s=35,color='#151C23',zorder=4)
axs[0].text(.38,.055,f'0.5-kHz peak alias:\n{100*z["alias_low"]:.2f}% and {100*z["alias_high"]:.2f}%',transform=axs[0].transAxes,fontsize=9.1)
axs[0].set(xlabel='Hydrogen mole fraction (%)',ylabel='Peak phase amplitude (mrad)')
axs[0].legend(loc='lower center',bbox_to_anchor=(.50,1.015),ncol=3,fontsize=8.7)
axs[1].set(xlabel='Hydrogen mole fraction (%)',ylabel='Phase of −Φ (degrees)')
panel(axs[0],'(a)');panel(axs[1],'(b)');save(fig,'fig03_concentration_alias')

# Frequency response separates heat diffusion and concentration-dependent relaxation.
z=np.load(R/'frequency_response.npz');fig,axs=plt.subplots(1,2,figsize=(8,3.65),layout='constrained')
for k,x in enumerate(z['x']):
    label=f'{100*x:.0f}% H₂';color=[BLUE,TEAL,GOLD,RED][k]
    axs[0].loglog(z['frequencies_Hz']/1000,abs(z['complex_phase'][k])*1000,label=label,color=color)
    axs[1].semilogx(z['frequencies_Hz']/1000,np.unwrap(np.angle(-z['complex_phase'][k]))*180/np.pi,color=color)
for ax in axs:
    for f in S['frequencies_Hz']:ax.axvline(f/1000,ls=':',lw=.8,color='#A4ABB2')
    ax.set_xlabel('Modulation frequency (kHz)')
axs[0].set_ylabel('Peak phase amplitude (mrad)');axs[0].legend();axs[1].set_ylabel('Phase of −Φ (degrees)')
panel(axs[0],'(a)');panel(axs[1],'(b)');save(fig,'fig04_frequency_response')

# Thermal field in gas core; the absolute scale is directly computed, not illustrative data.
z=np.load(R/'illustrative_thermal_field.npz');fig,axs=plt.subplots(1,2,figsize=(8,3.65),layout='constrained')
g=np.linspace(-C.radius,C.radius,241);xx,yy=np.meshgrid(g,g);rad=np.sqrt(xx**2+yy**2)
field=np.interp(rad,z['radius_m'],abs(z['complex_temperature_K'][0]),left=abs(z['complex_temperature_K'][0,0]),right=0)*1e3
field=np.ma.masked_where(rad>C.radius,field)
im=axs[0].pcolormesh(xx*1e6,yy*1e6,field,cmap='magma',shading='auto',rasterized=True)
axs[0].set_aspect('equal');axs[0].set(xlabel='Transverse position (µm)',ylabel='Transverse position (µm)',title='0.5 kHz; 30% H₂; 300 K; 100 kPa')
fig.colorbar(im,ax=axs[0],label='Temperature amplitude (mK)',shrink=.83)
for j,f in enumerate(z['frequencies_Hz']):
    axs[1].plot(z['radius_m']/C.radius,abs(z['complex_temperature_K'][j])*1e3,label=f'{f/1000:g} kHz',color=[BLUE,TEAL,RED][j])
axs[1].set(xlabel='Normalised radial position r / R',ylabel='Temperature amplitude (mK)');axs[1].legend()
panel(axs[0],'(a)');panel(axs[1],'(b)');save(fig,'fig05_thermal_field')

# Numerical verification and design selection in one compact figure.
d=pd.read_csv(R/'numerical_verification.csv');design=pd.read_csv(R/'frequency_design.csv').sort_values('worst_std_x')
fig,axs=plt.subplots(1,2,figsize=(8,4.25),layout='constrained')
for check,marker,color in [('Finite volume radial refinement','o',BLUE),('Bessel modal truncation','s',TEAL)]:
    sub=d[d.check==check];axs[0].loglog(sub.resolution,sub.relative_error,marker=marker,color=color,label='Finite volume' if marker=='o' else 'Bessel modes')
axs[0].set(xlabel='Radial cells or Bessel modes',ylabel='Relative complex-response error');axs[0].legend()
labels=[' / '.join(f'{row[k]/1000:g}' for k in ['f1_Hz','f2_Hz','f3_Hz']) for _,row in design.iterrows()]
axs[1].barh(np.arange(10),100*design.worst_std_x,color=[BLUE]+['#BBC6CF']*9)
axs[1].set(yticks=np.arange(10),yticklabels=labels,xlabel='Worst local standard uncertainty (pp)',ylabel='Frequency triplet (kHz)');axs[1].invert_yaxis()
panel(axs[0],'(a)');panel(axs[1],'(b)');save(fig,'fig06_verification_design')

# Main independent test, including concentration-dependent failures.
d=pd.read_csv(R/'test_estimates.csv');fig,axs=plt.subplots(2,2,figsize=(8,6.1),layout='constrained')
for ax,index in zip(axs.flat,[0,1,2,4]):
    sub=d[d.method==METHODS[index]]
    scatter=ax.scatter(100*sub.true_x,100*sub.estimated_x,c=sub.stratum,cmap='viridis',s=15,alpha=.7,vmin=0,vmax=3)
    ax.plot([0,100],[0,100],ls='--',color='#606C77',lw=1)
    ax.set(xlim=(-2,100),ylim=(-2,100),xlabel='True hydrogen concentration (%)',ylabel='Estimated concentration (%)',title=SHORT[index])
    ax.text(.04,.91,f'MAE = {S["main"][METHODS[index]]["mae_pp"]:.2f} pp',transform=ax.transAxes,fontsize=9.5)
for ax,label in zip(axs.flat,['(a)','(b)','(c)','(d)']):panel(ax,label)
save(fig,'fig07_test_scatter')

fig,axs=plt.subplots(2,2,figsize=(8,5.6),layout='constrained');idx=np.arange(5)
means=np.array([S['main'][m]['mae_pp'] for m in METHODS]);ci=np.array([S['bootstrap'][m]['mae_pp_ci95'] for m in METHODS])
axs[0,0].bar(idx,means,color=COLORS,yerr=np.array([means-ci[:,0],ci[:,1]-means]),capsize=3)
axs[0,0].set_ylabel('Concentration MAE (pp)')
raw=np.array([S['main'][m]['raw_coverage'] for m in METHODS])*100;scaled=np.array([S['main'][m]['coverage'] for m in METHODS])*100
axs[0,1].bar(idx-.17,raw,width=.34,color='#AAB9C6',label='Local 1.96×SE');axs[0,1].bar(idx+.17,scaled,width=.34,color=BLUE,label='Calibration-scaled')
axs[0,1].axhline(95,ls='--',color=RED,lw=1);axs[0,1].set(ylim=(0,104),ylabel='Empirical interval coverage (%)');axs[0,1].legend(fontsize=8.5,loc='lower right')
axs[1,0].bar(idx,[S['main'][m]['median_width_pp'] for m in METHODS],color=COLORS);axs[1,0].set_ylabel('Median interval width (pp)')
axs[1,1].bar(idx,[100*S['main'][m]['accepted_fraction'] for m in METHODS],color=COLORS);axs[1,1].set(ylabel='Reportable acquisitions (%)',ylim=(0,100))
for ax,label in zip(axs.flat,['(a)','(b)','(c)','(d)']):
    ax.set(xticks=idx,xticklabels=['Nominal','Plug-in','Joint','I only','Single']);ax.tick_params(axis='x',rotation=28);panel(ax,label)
save(fig,'fig08_metrics')

# Conditional errors, coverage by concentration stratum, and interval widths.
fig,axs=plt.subplots(1,2,figsize=(8,3.8),layout='constrained');labels=['0.20–5.00','5.00–30.00','30.00–65.00','65.00–95.00'];xx=np.arange(4)
for offset,index in zip([-.25,0,.25],[1,2,4]):
    values=[S['strata'][METHODS[index]][str(k)]['mae_pp'] for k in range(4)]
    axs[0].bar(xx+offset,values,width=.24,color=COLORS[index],label=SHORT[index])
    vals=[S['strata'][METHODS[index]][str(k)]['coverage']*100 for k in range(4)]
    axs[1].plot(xx,vals,marker=['s','o','^'][[1,2,4].index(index)],color=COLORS[index],label=SHORT[index])
for ax in axs:ax.set(xticks=xx,xticklabels=labels,xlabel='True hydrogen concentration band (%)');ax.tick_params(axis='x',rotation=20)
axs[0].set_ylabel('Concentration MAE (pp)');axs[0].legend(fontsize=8.5)
axs[1].set(ylabel='Empirical interval coverage (%)',ylim=(45,102));axs[1].axhline(95,color=RED,ls='--',lw=1)
panel(axs[0],'(a)');panel(axs[1],'(b)');save(fig,'fig09_strata')

# Prespecified profile scans; use one case from each concentration stratum.
d=pd.read_csv(R/'concentration_profiles.csv');fits=pd.read_csv(R/'test_estimates.csv');fig,axs=plt.subplots(2,2,figsize=(8,5.1),layout='constrained')
for ax,j in zip(axs.flat,[5,85,165,245]):
    sub=d[d.test_id==j];best=fits[(fits.method=='Joint uncertainty')&(fits.id==j)].iloc[0]
    ax.plot(100*sub.x,sub.delta_cost,color=BLUE,lw=1.5);ax.axhline(3.84146,color=RED,ls='--',lw=1,label='Δ objective = 3.84')
    ax.axvline(100*best.true_x,color=TEAL,ls=':',lw=1.4,label='True x');ax.axvline(100*best.estimated_x,color=GOLD,ls='-.',lw=1.2,label='Estimated x')
    ax.set(ylim=(0,18),xlim=(0,98),xlabel='Constrained concentration (%)',ylabel='Profile objective difference',title=f'Test case {j}; true {100*best.true_x:.2f}%')
axs[0,0].legend(fontsize=8.2)
for ax,label in zip(axs.flat,['(a)','(b)','(c)','(d)']):panel(ax,label)
save(fig,'fig10_profiles')

# All specified stress tests are reported, including biased-calibration failures.
names=list(S['challenges']);labels=['Wider T, p\nuncertainty','Biased T, p\nreadings','Thermal /\nspectral\nmismatch','4° phase\noffset'];xx=np.arange(4)
fig,axs=plt.subplots(1,2,figsize=(8,3.9),layout='constrained')
axs[0].bar(xx,[S['challenges'][m]['mae_pp'] for m in names],color=[BLUE,GOLD,RED,PURPLE]);axs[0].set_ylabel('Concentration MAE (pp)')
axs[1].bar(xx-.17,[S['challenges'][m]['coverage']*100 for m in names],width=.34,color=BLUE,label='Interval coverage')
axs[1].bar(xx+.17,[S['challenges'][m]['accepted_fraction']*100 for m in names],width=.34,color=TEAL,label='Reportable fraction')
axs[1].axhline(95,color=RED,ls='--',lw=1);axs[1].set(ylabel='Acquisitions (%)',ylim=(0,105));axs[1].legend(fontsize=8.5,loc='lower left')
for ax in axs:ax.set(xticks=xx,xticklabels=labels);ax.tick_params(axis='x',labelsize=8.8)
panel(axs[0],'(a)');panel(axs[1],'(b)');save(fig,'fig11_challenges')

# One actual noisy complex scan: the fitted physics curve and projected nuisance backgrounds.
data=np.load(R/'test_data.npz');fits=pd.read_csv(R/'test_estimates.csv');j=165;row=fits[(fits.method=='Joint uncertainty')&(fits.id==j)].iloc[0]
t=[row.estimated_x,row.T_estimated_K,row.p_estimated_Pa,row.log_gain_estimated,row.log_relaxation_estimated,row['shift_estimated_cm-1']]
from physics import Inverse, whiten
inverse=Inverse(data['frequencies_Hz']);clean=forward(t,data['frequencies_Hz']);res=data['complex_phase'][j]-clean
# The whitened QR solution restores the analytically eliminated complex baseline.
A=np.c_[np.ones(49),DETUNING]; Aw=whiten(A.T,C.noise_rad).real.T
background=np.array([A@np.linalg.lstsq(Aw,whiten(e,C.noise_rad),rcond=None)[0] for e in res]);fitted=clean+background
fig,axs=plt.subplots(2,3,figsize=(8,4.65),layout='constrained')
for k,f in enumerate(data['frequencies_Hz']):
    for q,fun in enumerate([np.real,np.imag]):
        ax=axs[q,k];ax.scatter(DETUNING,fun(data['complex_phase'][j,k])*1e3,s=9,color='#6C7680',alpha=.7,label='Noisy observation')
        ax.plot(DETUNING,fun(fitted[k])*1e3,color=BLUE,lw=1.5,label='Fitted model + baseline')
        ax.set(xlabel='Line detuning (cm⁻¹)',ylabel='I phase (mrad)' if q==0 else 'Q phase (mrad)')
        if q==0:ax.set_title(f'{f/1000:g} kHz')
handles,legend_labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='outside upper center',ncol=2,fontsize=9,frameon=False)
save(fig,'fig12_complex_fit')

(F/'figure_manifest.json').write_text(json.dumps({'figures':sorted(p.name for p in F.glob('*.png')),
 'origin':'Original vector schematics and plots from executed retained numerical outputs; no generated or copied data graphics.'},indent=2))
print('Wrote',len(list(F.glob('*.png'))),'figures as PNG and SVG')
