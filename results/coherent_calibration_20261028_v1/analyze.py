"""Independent policy replay, split/freeze checks, paired effects and rank diagnostics."""
import csv
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]; RUN=ROOT/'runs'/OUT.name
assert json.loads((RUN/'status.json').read_text())['status']=='complete'
config=json.loads((RUN/'config_resolved.json').read_text()); settings=config['calibration']
frozen=json.loads((RUN/'frozen.json').read_text())
for name,digest in frozen['files_sha256'].items():
    assert hashlib.sha256((RUN/name).read_bytes()).hexdigest()==digest
resume=json.loads((RUN/'resume_provenance.json').read_text())
assert hashlib.sha256((RUN/'resume_source.py').read_bytes()).hexdigest()==resume['resume_source_sha256']
with zipfile.ZipFile(RUN/'source.zip') as zipped:
    original=ast.parse(zipped.read('scripts/probe_coherent_calibration.py').decode())
current=ast.parse((RUN/'resume_source.py').read_text(encoding='utf-8'))
old_functions={n.name:n for n in ast.walk(original) if isinstance(n,ast.FunctionDef)}
new_functions={n.name:n for n in ast.walk(current) if isinstance(n,ast.FunctionDef)}
for name in ['delay_score','free_from_prepared','terms','transform','fit_step','step','infer','evaluate']:
    assert ast.dump(old_functions[name],include_attributes=False)==ast.dump(new_functions[name],include_attributes=False),name
observed=torch.load(RUN/'predictions/screen_observations.pt',map_location='cpu',weights_only=True)
assert observed['created_at_unix']>=frozen['frozen_at_unix']
assert observed['frozen_sha256']==hashlib.sha256((RUN/'frozen.json').read_bytes()).hexdigest()
assert len(set(observed['geometry_ids']))==32
assert not set(observed['geometry_ids'])&set(frozen['training_geometry_ids']+frozen['validation_geometry_ids'])
assert not set(frozen['training_geometry_ids'])&set(frozen['validation_geometry_ids'])
parent=ROOT/'runs'/settings['parent_run']
physical=torch.load(parent/'inputs/physical_parent.pt',map_location='cpu',weights_only=True)
assert not set(observed['geometry_ids'])&set(physical['geometry_ids']['train']+physical['geometry_ids']['validation'])
matrix=physical['contexts'][0]['matrix']
del physical
rows=list(csv.DictReader((OUT/'metrics.csv').open()))
diagnostics=list(csv.DictReader((OUT/'diagnostics.csv').open()))
maximum_score_error=0.; maximum_policy_error=0.; maximum_fit_error=0.; solver_rows=[]
for context in range(5):
    train=torch.load(RUN/f'cache/train_{context}.pt',map_location='cpu',weights_only=True)
    saved=torch.load(RUN/f'predictions/screen_{context}.pt',map_location='cpu',weights_only=True)
    for split,entry in [('train',train),('screen',saved)]:
        residual=entry['optimality_residual'].numpy()
        solver_rows.append({'split':split,'snr_db':[-10,0,10,20,30][context],
            'iterations':settings['sbl_iterations'],'threshold':config['sbl']['tolerance'],
            'outputs':len(residual),'not_at_threshold':int(np.sum(residual>config['sbl']['tolerance'])),
            'minimum_residual':float(residual.min()),'maximum_residual':float(residual.max())})
    assert saved['geometry_ids']==observed['geometry_ids']
    assert train['geometry_ids']==frozen['training_geometry_ids']
    gamma=train['gamma'].numpy().astype(float); model=train['model']
    powers=np.take_along_axis(gamma,model['indices'].numpy(),1)
    score=np.clip((powers*model['weights'].numpy().sum(-1)).sum(-1)/gamma.sum(-1),1e-6,1-1e-6)
    np.testing.assert_allclose(score,train['score'].numpy(),rtol=1e-12)
    logits=np.log(score/(1-score)); choice=frozen['choices'][str(context)]
    np.testing.assert_allclose([logits.mean(),logits.std()],
        [choice['score_logit_mean'],choice['score_logit_std']],rtol=1e-10,atol=1e-12)
    z=(logits-logits.mean())/logits.std()
    h=train['truth'].numpy().astype(complex); free=train['free'].numpy().astype(complex)
    change=train['coherent'].numpy().astype(complex)-free
    energy=(abs(h)**2).sum((1,2)); c=(abs(change)**2).sum((1,2))/energy
    b=((h-free).conj()*change).real.sum((1,2))/energy
    validation=torch.load(parent/f'cache/validation_{context}.pt',map_location='cpu',weights_only=True)
    vh=validation['truth'].numpy().astype(complex); vf=validation['outputs']['rho_0'].numpy().astype(complex)
    vd=validation['outputs']['rho_0.95'].numpy().astype(complex)-vf
    ve=(abs(vh)**2).sum((1,2)); vb=((vh-vf).conj()*vd).real.sum((1,2))/ve
    vc=(abs(vd)**2).sum((1,2))/ve; vbefore=(abs(vh-vf)**2).sum((1,2))/ve
    vg=validation['gamma'].numpy().astype(float)
    vs=(np.take_along_axis(vg,validation['model']['indices'].numpy(),1)*validation['model']['weights'].numpy().sum(-1)).sum(-1)/vg.sum(-1)
    vs=np.clip(vs,1e-6,1-1e-6); vz=(np.log(vs/(1-vs))-choice['score_logit_mean'])/choice['score_logit_std']
    for name in ['constant','score']:
        fit=choice[name]; ridge=fit['ridge']; coef=np.array(fit['coefficients'])
        x=np.ones((len(z),1)) if name=='constant' else np.column_stack([np.ones(len(z)),z])
        target=np.zeros(len(coef));target[0]=1
        penalty=ridge*c.sum()
        design=np.concatenate([np.sqrt(c)[:,None]*x,np.sqrt(penalty)*np.eye(len(coef))])
        response=np.concatenate([b/np.sqrt(c),np.sqrt(penalty)*target])
        alternative=np.linalg.lstsq(design,response,rcond=None)[0]
        maximum_fit_error=max(maximum_fit_error,float(abs(alternative-coef).max()))
        np.testing.assert_allclose(coef,alternative,rtol=1e-8,atol=1e-9)
        options=[]
        for candidate_ridge in settings['ridge_grid']:
            penalty=candidate_ridge*c.sum()
            design=np.concatenate([np.sqrt(c)[:,None]*x,np.sqrt(penalty)*np.eye(len(coef))])
            response=np.concatenate([b/np.sqrt(c),np.sqrt(penalty)*target])
            beta=np.linalg.lstsq(design,response,rcond=None)[0]
            lam=np.clip(beta[0]+(beta[1]*vz if len(beta)>1 else 0),*settings['step_bounds'])
            options.append((float(np.mean(vbefore+lam**2*vc-2*lam*vb)),-candidate_ridge))
        assert -min(options)[1]==ridge,(context,name,options,ridge)
    truth=saved['truth'].numpy().astype(complex); f=saved['free'].numpy().astype(complex)
    d=saved['coherent'].numpy().astype(complex)-f
    z=(np.log(saved['score'].numpy()/(1-saved['score'].numpy()))-choice['score_logit_mean'])/choice['score_logit_std']
    for method,prediction in saved['outputs'].items():
        output=prediction.numpy(); score=(abs(output-truth)**2).sum((1,2))/(abs(truth)**2).sum((1,2))
        recorded=[r for r in rows if r['split']=='screen' and int(r['context'])==context and r['method']==method]
        assert [r['geometry_id'] for r in recorded]==observed['geometry_ids']
        maximum_score_error=max(maximum_score_error,float(np.max(abs(score-[float(r['nmse_linear']) for r in recorded])/score)))
        if method in ['free','fixed_coherent']:
            lam=np.full(len(z),0. if method=='free' else 1.)
        else:
            coef=choice[method.split('_')[0]]['coefficients']
            lam=np.clip(coef[0]+(coef[1]*z if len(coef)>1 else np.zeros_like(z)),*settings['step_bounds'])
        expected=f+lam[:,None,None]*d
        error=float((abs(expected-output)**2).sum()/max((abs(output)**2).sum(),1e-30))
        maximum_policy_error=max(maximum_policy_error,error)
        assert error<1e-20
assert maximum_score_error<1e-10

# Direct independent-prior equations check the new restoration shortcut at full size.
sys.path.insert(0,str(ROOT/'src'))
from dma_ce.physical_study import case_hardware
_,_,hardware=case_hardware(config,config['cases'][0])
dictionary=hardware[0]['dictionary']; ad=matrix.cuda().to(torch.complex128)
maximum_direct_free_difference=0.
for context in [0,4]:
    saved=torch.load(RUN/f'predictions/screen_{context}.pt',map_location='cpu',weights_only=True)
    gamma=saved['gamma'][:1].cuda().to(torch.float64)
    y=saved['observation'][:1].cuda().to(torch.complex128)
    noise=saved['variance'][:1].cuda().to(torch.float64)
    chunks=[]
    for first in range(0,len(ad),4):
        d=dictionary[first:first+4].cuda().to(torch.complex128)
        phi=ad[first:first+4]@d
        covariance=(phi[None]*gamma[:,None,None])@phi.mH+noise[:,None,None,None]*torch.eye(ad.shape[1],dtype=ad.dtype,device=ad.device)
        solved=torch.linalg.solve(covariance,y[:,first:first+4,:,None])
        coefficients=gamma[:,None,:,None]*(phi.mH[None]@solved)
        chunks.append((d[None]@coefficients).squeeze(-1).cpu())
    direct=torch.cat(chunks,dim=1).numpy(); recorded=saved['free'].numpy().astype(complex)[:1]
    difference=float((abs(direct-recorded)**2).sum()/(abs(direct)**2).sum())
    maximum_direct_free_difference=max(maximum_direct_free_difference,difference)
    assert difference<1e-10
del hardware,dictionary,ad

def write_csv(name,values):
    with (OUT/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)

draws=np.random.default_rng(settings['bootstrap_seed']).integers(0,32,(settings['bootstrap_samples'],32))
summary=[]; effects=[]
for split in ['train','validation','screen']:
    for snr in [-10,0,10,20,30]:
        scores={}
        for method in ['free','fixed_coherent','constant_calibrated','score_calibrated']:
            part=[r for r in rows if r['split']==split and float(r['snr_db'])==snr and r['method']==method]
            assert len(part)==32
            scores[method]=np.array([float(r['nmse_linear']) for r in part])
            lam=np.array([float(r['step']) for r in part])
            summary.append({'split':split,'snr_db':snr,'method':method,'scenes':len(part),
                'mean_nmse_linear':float(scores[method].mean()),'mean_nmse_db':float(10*np.log10(scores[method].mean())),
                'step_mean':float(lam.mean()),'step_min':float(lam.min()),'step_max':float(lam.max()),
                'step_upper_fraction':float(np.mean(lam>=2))})
        if split=='screen':
            for candidate,reference in [('fixed_coherent','free'),('constant_calibrated','fixed_coherent'),
                    ('score_calibrated','fixed_coherent'),('score_calibrated','constant_calibrated')]:
                x,y=scores[candidate],scores[reference]
                low,high=np.quantile(10*np.log10(y[draws].mean(1)/x[draws].mean(1)),[.025,.975])
                effects.append({'snr_db':snr,'candidate':candidate,'reference':reference,
                    'improvement_db':float(10*np.log10(y.mean()/x.mean())),
                    'ci95_low_db':float(low),'ci95_high_db':float(high),'scene_win_rate':float(np.mean(x<y))})

def spearman(x,y):
    def rank(v):
        return (v[None,:]<v[:,None]).sum(1)+((v[None,:]==v[:,None]).sum(1)-1)/2
    a,b=rank(x),rank(y)
    if a.std()==0 or b.std()==0:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])

correlations=[]
for split in ['train','validation','screen']:
    for snr in [-10,0,10,20,30]:
        part=[r for r in diagnostics if r['split']==split and float(r['snr_db'])==snr]
        score=np.array([float(r['score']) for r in part])
        for target in ['oracle_step_diagnostic','fixed_relative_gain','relative_correction_energy']:
            value=(np.array([float(r['correction'])/float(r['before']) for r in part])
                if target=='relative_correction_energy' else np.array([float(r[target]) for r in part]))
            estimate=spearman(score,value)
            if split=='screen':
                bootstrap=np.array([spearman(score[s],value[s]) for s in draws])
                low,high=np.nanquantile(bootstrap,[.025,.975])
            else:
                low=high=None
            correlations.append({'split':split,'snr_db':snr,'target':target,'spearman':estimate,
                'ci95_low':low,'ci95_high':high})
write_csv('summary.csv',summary);write_csv('paired_effects.csv',effects);write_csv('score_correlations.csv',correlations)
write_csv('solver_diagnostics.csv',solver_rows)
verification={'status':'passed','maximum_relative_score_error':maximum_score_error,
    'maximum_direct_free_posterior_relative_squared_difference':maximum_direct_free_difference,
    'maximum_policy_relative_squared_error':maximum_policy_error,'maximum_augmented_LS_coefficient_difference':maximum_fit_error,
    'train_only_feature_statistics':True,'frozen_inputs_verified':True,'freeze_before_new_observations':True,
    'frozen_split_ids_disjoint':True,'calibration_math_unchanged_across_resume':True,
    'validation_ridge_selection_replayed':True,
    'scope':'saved full-CSI rescoring and independent augmented least-squares/policy replay'}
(OUT/'independent_verification.json').write_text(json.dumps(verification,indent=2)+'\n')

fig,axes=plt.subplots(1,2,figsize=(12,4.7))
snrs=[-10,0,10,20,30]
for method,label in [('constant_calibrated','Constant step'),('score_calibrated','Score-dependent step')]:
    part=[next(r for r in effects if r['snr_db']==s and r['candidate']==method and r['reference']=='fixed_coherent') for s in snrs]
    y=np.array([r['improvement_db'] for r in part]);low=np.array([r['ci95_low_db'] for r in part]);high=np.array([r['ci95_high_db'] for r in part])
    axes[0].errorbar(snrs,y,yerr=np.stack([y-low,high-y]),fmt='o-',capsize=3,label=label)
axes[0].set_ylabel('Improvement vs fixed coherent update (dB)');axes[0].set_title('Frozen calibration on 32 new geometries')
for target,label in [('oracle_step_diagnostic','Best-step diagnostic'),('fixed_relative_gain','Fixed-update relative gain')]:
    part=[next(r for r in correlations if r['split']=='screen' and r['snr_db']==s and r['target']==target) for s in snrs]
    y=np.array([r['spearman'] for r in part]);low=np.array([r['ci95_low'] for r in part]);high=np.array([r['ci95_high'] for r in part])
    axes[1].errorbar(snrs,y,yerr=np.stack([y-low,high-y]),fmt='o-',capsize=3,label=label)
axes[1].set_ylabel('Score rank correlation');axes[1].set_title('Within-SNR prediction on new geometries')
for ax in axes:
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(snrs);ax.set_xlabel('Training-average received SNR (dB)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png','svg']:
    fig.savefig(OUT/f'calibration.{ext}',dpi=170)
plt.close(fig)
print(json.dumps(verification))
