"""Independent retained-output audit; does not rerun or tune the fitted estimators."""
from pathlib import Path
import json, zipfile, re
import numpy as np
import pandas as pd
from scipy.stats import chi2
from PIL import Image
from lxml import etree
from physics import C, forward, spectroscopic, gas_properties, fv_thermal

ROOT=Path(__file__).resolve().parent; R=ROOT/'results'
S=json.loads((R/'summary.json').read_text())
d=pd.read_csv(R/'test_estimates.csv');cal=pd.read_csv(R/'calibration_estimates.csv')
assert len(d)==1600 and len(cal)==600
assert d.groupby('method').size().eq(320).all()
assert d.groupby(['method','stratum']).size().eq(80).all()
assert np.isfinite(d.select_dtypes('number').to_numpy()).all()
audit=[]
for method in S['main']:
    a=d[d.method==method].sort_values('id');e=a.estimated_x.to_numpy()-a.true_x.to_numpy()
    mae=float(np.mean(abs(e))*100);rmse=float(np.sqrt(np.mean(e**2))*100)
    cover=float(((a.true_x>=a.lower_x)&(a.true_x<=a.upper_x)).mean())
    accepted=(a.red_residual<=S['discrepancy_thresholds'][method])&(a.branch_gap>chi2.ppf(.95,1))&(a.upper_x-a.lower_x<=.15)
    wrong=int(np.sum(accepted.to_numpy()&(abs(e)>.05)))
    for value,key in [(mae,'mae_pp'),(rmse,'rmse_pp'),(cover,'coverage'),(int(accepted.sum()),'accepted'),(wrong,'wrong_accepted_5pp')]:
        assert np.isclose(value,S['main'][method][key],atol=1e-10),(method,key,value)
    assert np.array_equal(accepted.to_numpy().astype(int),a.accepted.to_numpy())
    audit.append({'method':method,'mae_pp_recomputed':mae,'coverage_recomputed':cover,'accepted_recomputed':int(accepted.sum())})
profiles=pd.read_csv(R/'concentration_profiles.csv');profile_checks=[]
challenge_checks=[]
challenges=pd.read_csv(R/'challenge_estimates.csv')
assert len(challenges)==256
for scenario,a in challenges.groupby('scenario'):
    e=(a.estimated_x-a.true_x).to_numpy();accepted=a.accepted.to_numpy().astype(bool)
    checks={'mae_pp':float(np.mean(abs(e))*100),'coverage':float(((a.true_x>=a.lower_x)&(a.true_x<=a.upper_x)).mean()),
      'median_width_pp':float(np.median(a.upper_x-a.lower_x)*100),'accepted':int(accepted.sum()),
      'wrong_accepted_5pp':int(np.sum(accepted&(abs(e)>.05)))}
    for key,value in checks.items():assert np.isclose(value,S['challenges'][scenario][key],atol=1e-10),(scenario,key)
    challenge_checks.append({'scenario':scenario,**checks})
for j,a in profiles.groupby('test_id'):
    best=d[(d.method=='Joint uncertainty')&(d.id==j)].iloc[0]
    improvement=float(best.cost-a.profile_cost.min())
    assert improvement<1e-3,('Profile finds a better main objective',j,improvement)
    profile_checks.append({'case_id':int(j),'main_minus_best_profile_cost':improvement})
design=pd.read_csv(R/'frequency_design.csv');best=design.loc[design.worst_std_x.idxmin()]
assert np.allclose([best.f1_Hz,best.f2_Hz,best.f3_Hz],S['frequencies_Hz'])
raw=np.load(R/'test_data.npz');assert raw['complex_phase'].shape==(320,3,49)
assert raw['single_equal_time'].shape==(320,1,49)
assert raw['auxiliary'].shape==(320,4)
max_depth=max(float(max(spectroscopic(*t[:3],shift=t[5]))*C.length) for t in raw['theta'])
alias=[S['alias_low'],S['alias_high']]
signals=[abs(forward([x,300,100000,0,0,0],[500])[0,24]) for x in alias]
assert np.isclose(*signals,rtol=1e-10,atol=1e-15)
for p in (ROOT/'figures').glob('*.png'):
    with Image.open(p) as im:im.verify()
manifest=json.loads((ROOT/'manuscript_manifest.json').read_text())
assert manifest['equation_numbers']==[str(i) for i in range(1,35)]
docx=ROOT.parent.parent/'output/Hydrogen_Photothermal_Uncertainty_Manuscript.docx'
with zipfile.ZipFile(docx) as z:
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    tree=etree.fromstring(z.read('word/document.xml'))
    native_math=len(tree.xpath('.//m:oMath',namespaces=ns))
    assert native_math>=34
    math_text=''.join(tree.xpath('.//m:t/text()',namespaces=ns))
    assert '\\' not in math_text
    assert len(z.namelist())>10 and z.testzip() is None
    assert len(tree.xpath('.//w:drawing',namespaces=ns))==12
    assert len(tree.xpath('.//w:tbl',namespaces=ns))==10 # nine numbered tables plus algorithm
results={'status':'passed','independently_recomputed_metrics':audit,'challenge_metrics_audit':challenge_checks,'profile_objective_audit':profile_checks,
 'maximum_test_optical_depth':max_depth,'scalar_alias_amplitudes_rad':signals,'native_editable_equations':native_math,
 'numerical_checks':'Mesh, modal and analytical checks retained in results/numerical_verification.csv and summary.json',
 'scope':'Internal numerical and document integrity only; no experimental validation.'}
(R/'integrity_audit.json').write_text(json.dumps(results,indent=2))
print(json.dumps({'status':'passed','main_rows':len(d),'calibration_rows':len(cal),'profile_cases':len(profile_checks),'maximum_test_optical_depth':max_depth,'native_equations':native_math},indent=2))
