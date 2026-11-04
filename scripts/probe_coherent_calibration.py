"""Train-only step calibration, validation ridge selection, then fresh geometries."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from analyze_coherent_mechanism import prepare_controls, control_posterior
from dma_ce.coherent import delay_model, coherent_posterior
from dma_ce.data import sync, tensor_hash
from dma_ce.experiment import file_hash, write_csv, write_json
from dma_ce.physical_study import archive, case_hardware, make_entries
from dma_ce.prior_probe import covariance_sbl
from dma_ce.sbl import posterior


def delay_score(gamma, model):
    selected = gamma.to(torch.float64).gather(1, model["indices"])
    score = (selected * model["weights"].sum(-1)).sum(-1) / gamma.to(torch.float64).sum(-1)
    if not torch.isfinite(score).all() or (score < 0).any() or (score > 1+1e-10).any():
        raise ValueError("finite fraction of retained coherent variance required")
    return score.clamp(1e-6, 1-1e-6)


def free_from_prepared(prepared, y):
    """Restore the removed per-tone variance to recover the full independent prior."""
    removed = prepared["weight_scale"].square().sum(-1)
    weighted = prepared["sensing"] * removed[:, None, None]
    covariance = prepared["factor"] @ prepared["factor"].mH + weighted @ prepared["sensing"].mH
    cross = prepared["cross"] + prepared["dictionary"] @ weighted.mH
    solved = torch.cholesky_solve(y.to(torch.complex128)[..., None], torch.linalg.cholesky(covariance))
    return (cross @ solved).squeeze(-1)


def terms(truth, free, coherent):
    h, f, d = truth.to(torch.complex128), free.to(torch.complex128), coherent.to(torch.complex128)-free.to(torch.complex128)
    energy = h.abs().square().sum((1, 2))
    if (energy <= 0).any():
        raise ValueError("positive full truth energy required")
    r = h-f
    return {"before": (r.abs().square().sum((1,2))/energy).cpu().numpy(),
        "c": (d.abs().square().sum((1,2))/energy).cpu().numpy(),
        "b": ((r.conj()*d).real.sum((1,2))/energy).cpu().numpy()}


def transform(score, mean, scale):
    return (np.log(score/(1-score))-mean)/scale


def fit_step(z, c, b, ridge, conditional):
    x = np.column_stack((np.ones(len(z)), z)) if conditional else np.ones((len(z),1))
    target = np.zeros(x.shape[1]); target[0] = 1.
    if (not all(np.isfinite(v).all() for v in (x,c,b)) or (c < 0).any() or c.sum() <= 0):
        raise ValueError("finite positive correction information required")
    penalty = ridge*c.sum()
    return np.linalg.solve(x.T@(c[:,None]*x)+penalty*np.eye(x.shape[1]),x.T@b+penalty*target)


def step(coef, z, bounds):
    return np.clip(coef[0]+(coef[1]*z if len(coef)>1 else np.zeros_like(z)),*bounds)


def self_check():
    z=np.array([-1.5,-.5,.5,1.5]); c=np.ones(4); b=1+.2*z
    np.testing.assert_allclose(fit_step(z,c,b,0,True),[1,.2],atol=1e-12)
    np.testing.assert_allclose(fit_step(z,c,b,0,False),[1],atol=1e-12)
    np.testing.assert_allclose(step(np.array([1,2]),z,[0,2]),[0,0,2,2])
    torch.manual_seed(20261028)
    dtype=torch.complex128
    a=torch.randn(4,2,5,dtype=dtype); d=torch.randn(4,5,7,dtype=dtype)
    y=torch.randn(2,4,2,dtype=dtype); gamma=torch.rand(2,7,dtype=torch.float64)+.2
    noise=torch.tensor([.1,.3],dtype=torch.float64)
    model={"indices":torch.tensor([[0,2],[1,4]]),"delays":torch.rand(2,2,2,dtype=torch.float64),
        "weights":torch.full((2,2,2),.2,dtype=torch.float64),"envelope":torch.ones(4,1,dtype=torch.float64)}
    prepared=prepare_controls(y,a,d,noise,gamma,model,.95)
    free=free_from_prepared(prepared,y)
    expected=coherent_posterior(y,a,d,noise,gamma,model,0)["mean"]
    torch.testing.assert_close(free,expected,rtol=1e-10,atol=1e-10)
    coherent=control_posterior(prepared,model["delays"])
    t=terms(expected,free,coherent)
    for value in [0,.5,1,2]:
        actual=terms(expected,free,free+value*(coherent-free))["c"]
        np.testing.assert_allclose(actual,value**2*t["c"],atol=1e-12)
    assert ((delay_score(gamma,model)>0)&(delay_score(gamma,model)<1)).all()


def run(resume=False):
    self_check()
    settings=json.loads((ROOT/'configs/coherent_calibration.json').read_text())
    if any(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',settings[k]) is None for k in ('run_id','parent_run')):
        raise ValueError('safe independent run IDs required')
    if (settings['train_geometries']!=32 or settings['validation_geometries']!=32 or settings['screen_geometries']!=32
            or settings['prior_strength']!=.95 or settings['ridge_grid']!=[0,.1,1,10]
            or settings['step_bounds']!=[0,2] or not 0<settings['timeout_seconds']<=1800):
        raise ValueError('use the declared bounded calibration design')
    parent=ROOT/'runs'/settings['parent_run']; path=ROOT/'runs'/settings['run_id']; out=ROOT/'results'/settings['run_id']
    if (path.exists() or out.exists()) and not resume:
        raise FileExistsError('unused run ID required')
    previous_frozen=None; prior_elapsed=0.
    if resume:
        old_status=json.loads((path/'status.json').read_text())
        if old_status['status'] not in ['stopped_checkpoint','failed']:
            raise ValueError('resume requires an explicitly stopped or failed incomplete run')
        previous_frozen=json.loads((path/'frozen.json').read_text())
        assert json.loads((path/'config_resolved.json').read_text())['calibration']==settings
        assert (path/'predictions/screen_observations.pt').is_file()
        for name,digest in previous_frozen['files_sha256'].items():
            assert file_hash(path/name)==digest
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        prior_elapsed=max(events[-1]['elapsed_seconds'],old_status.get('elapsed_seconds',0.))
        shutil.copyfile(Path(__file__),path/'resume_source.py')
        write_json(path/'resume_provenance.json',{'prior_elapsed_seconds':prior_elapsed,
            'resume_source_sha256':file_hash(path/'resume_source.py'),'started_at_unix':time.time(),
            'reason':'split unchanged frozen experiment to retain all 32 screening geometries within per-segment budget'})
    if shutil.disk_usage(ROOT).free<2_000_000_000:
        raise OSError('2 GB storage required')
    expected=json.loads((parent/'delivery_manifest.json').read_text())
    names=['config_resolved.json','source.zip','parent_provenance.json','frozen.json','inputs/physical_parent.pt','inputs/absorption_table.txt',
        *[f'cache/validation_{i}.pt' for i in range(5)]]
    digests={name:file_hash(parent/name) for name in names}
    assert all(digests[n]==expected[n] for n in names)
    if not resume:
        for folder in [path,out,path/'inputs',path/'cache',path/'predictions',path/'export']:
            folder.mkdir()
    config=json.loads((parent/'config_resolved.json').read_text())
    config['device']=settings['device']; config['calibration']=settings
    config['channel']['absorption_table_path']=str(parent/'inputs/absorption_table.txt')
    config['data'].update(screen=settings['screen_geometries'],screen_seed=settings['screen_seed'])
    if not resume:
        archive(ROOT,path,path/'export',config)
    write_json(path/'input_provenance.json',{'parent_run':settings['parent_run'],'files_sha256':digests})
    start=time.monotonic()
    def status(value):
        for folder in [path,out]:
            write_json(folder/'status.json',value)
    def log(event,**fields):
        data={'event':event,'elapsed_seconds':prior_elapsed+time.monotonic()-start,**fields}
        with (path/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(data)+'\n')
        print(json.dumps(data),flush=True)
        if time.monotonic()-start>settings['timeout_seconds']:
            raise TimeoutError('bounded calibration budget reached')
    status({'status':'running','stage':'training_labels'})
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        device=torch.device(settings['device'])
        cache=torch.load(parent/'inputs/physical_parent.pt',map_location='cpu',weights_only=True)
        selected,bank,hardware=case_hardware(config,config['cases'][0]); hw=hardware[0]
        assert tensor_hash(torch.tensor(bank))==tensor_hash(cache['state_bank'])
        a=cache['contexts'][0]['matrix'].to(device); ad=a.to(torch.complex128)
        assert tensor_hash(hw['matrix'])==tensor_hash(a)
        dictionary=hw['dictionary'].to(device)
        frequencies=torch.tensor(hw['frequencies'],device=device,dtype=torch.float64)
        indices=json.loads((parent/'parent_provenance.json').read_text())['original_context_indices']
        contexts=[cache['contexts'][i] for i in indices]
        rng=np.random.default_rng(settings['training_noise_seed'])
        shape=(settings['train_geometries'],a.shape[0],a.shape[1])
        standard=torch.tensor((rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2),dtype=torch.complex64)

        def infer(entry,ids,split,context):
            collected={key:[] for key in ['free','coherent','score','gamma','optimality_residual']}; models=[]
            for first in range(0,len(ids),settings['batch_size']):
                last=min(first+settings['batch_size'],len(ids))
                y=entry['observation'][first:last].to(device); noise=entry['variance'][first:last].to(device)
                base=covariance_sbl(y,a,dictionary,noise,dictionary[...,:1],0.,settings['sbl_iterations'])
                coef,var,_,_=posterior(y,a@dictionary,dictionary,noise,base['gamma'],channel_moments=False)
                model=delay_model(coef,var,base['gamma'],frequencies,config['probe'])
                prepared=prepare_controls(y,ad,dictionary,noise,base['gamma'],model,settings['prior_strength'])
                free=free_from_prepared(prepared,y); coherent=control_posterior(prepared,model['delays'])
                for key,value in {'free':free.to(torch.complex64),'coherent':coherent.to(torch.complex64),
                        'score':delay_score(base['gamma'],model),'gamma':base['gamma'],
                        'optimality_residual':base['optimality_residual']}.items():
                    collected[key].append(value.cpu())
                models.append({key:value.cpu() for key,value in model.items()})
                sync(device); log('inference_batch',split=split,context=context,completed=last,count=len(ids))
            return {**{key:torch.cat(value) for key,value in collected.items()},'model':{
                key:models[0][key] if key=='envelope' else torch.cat([m[key] for m in models]) for key in models[0]},
                **entry,'geometry_ids':ids}

        training=[]; validation=[]; choices={}; candidates=[]
        train_ids=cache['geometry_ids']['train'][:settings['train_geometries']]
        for context,physical in enumerate(contexts):
            truth=cache['training'][0][:len(train_ids)]
            variance=torch.full((len(truth),),physical['noise_complex_variance'])
            observed=torch.einsum('kmn,bkn->bkm',a.cpu(),truth)+variance.sqrt()[:,None,None]*standard
            if resume:
                entry=torch.load(path/f'cache/train_{context}.pt',map_location='cpu',weights_only=True)
            else:
                entry=infer({'truth':truth,'observation':observed,'variance':variance},train_ids,'train',context)
                torch.save(entry,path/f'cache/train_{context}.pt')
            training.append(entry)
            old=torch.load(parent/f'cache/validation_{context}.pt',map_location='cpu',weights_only=True)
            val={key:old[key] for key in ['truth','observation','variance','geometry_ids']}
            val.update(free=old['outputs']['rho_0'],coherent=old['outputs']['rho_0.95'],score=delay_score(old['gamma'],old['model']))
            validation.append(val)
            score=entry['score'].numpy(); logits=np.log(score/(1-score)); mean=float(logits.mean()); scale=max(float(logits.std()),1e-8)
            z=transform(score,mean,scale); zv=transform(val['score'].numpy(),mean,scale)
            t=terms(entry['truth'],entry['free'],entry['coherent']); tv=terms(val['truth'],val['free'],val['coherent'])
            choices[str(context)]={'score_logit_mean':mean,'score_logit_std':scale}
            for name,conditional in [('constant',False),('score',True)]:
                options=[]
                for ridge in settings['ridge_grid']:
                    coef=fit_step(z,t['c'],t['b'],ridge,conditional)
                    lam=step(coef,zv,settings['step_bounds'])
                    risk=float(np.mean(tv['before']+lam**2*tv['c']-2*lam*tv['b']))
                    options.append((risk,-ridge,coef)); candidates.append({'context':context,'snr_db':physical['snr_db'],
                        'calibrator':name,'ridge':ridge,'validation_nmse_linear':risk,'coefficients':json.dumps(coef.tolist())})
                best=min(options,key=lambda x:(x[0],x[1]))
                choices[str(context)][name]={'ridge':-best[1],'coefficients':best[2].tolist(),'validation_nmse_linear':best[0]}
            log('validation_selection',context=context,choices=choices[str(context)])
        write_csv(out/'validation.csv',candidates)
        frozen={'frozen_at_unix':time.time(),'selection_split':'validation','screening_used':False,'choices':choices,
            'training_geometry_ids':train_ids,'validation_geometry_ids':validation[0]['geometry_ids'],
            'files_sha256':{p.relative_to(path).as_posix():file_hash(p) for p in [path/'source.zip',*sorted((path/'cache').glob('train_*.pt'))]}}
        if resume:
            for context in choices:
                for name in ['constant','score']:
                    assert choices[context][name]['ridge']==previous_frozen['choices'][context][name]['ridge']
                    np.testing.assert_allclose(choices[context][name]['coefficients'],
                        previous_frozen['choices'][context][name]['coefficients'],rtol=1e-12,atol=1e-12)
            choices=previous_frozen['choices']
        else:
            write_json(path/'frozen.json',frozen); write_json(out/'frozen.json',frozen)
        status({'status':'running','stage':'new_screen'})
        reduced={**cache,'contexts':contexts}
        if resume:
            old_observations=torch.load(path/'predictions/screen_observations.pt',map_location='cpu',weights_only=True)
            assert old_observations['frozen_sha256']==file_hash(path/'frozen.json')
            screen_entries,ids=old_observations['entries'],old_observations['geometry_ids']
        else:
            screen_entries,ids=make_entries(config,selected,hardware,reduced,'screen')
        known=set(cache['geometry_ids']['train']+cache['geometry_ids']['validation'])
        for filename in (ROOT/'results').glob('*/metrics.csv'):
            if filename.parent==out:
                continue
            with filename.open() as stream:
                known.update(r['geometry_id'] for r in csv.DictReader(stream) if r.get('geometry_id'))
        assert len(set(ids))==len(ids) and not set(ids)&known
        if not resume:
            torch.save({'entries':screen_entries,'geometry_ids':ids,'created_at_unix':time.time(),
                'frozen_sha256':file_hash(path/'frozen.json')},path/'predictions/screen_observations.pt')
        metrics=[]; diagnostics=[]
        def evaluate(split,context,entry):
            choice=choices[str(context)]; t=terms(entry['truth'],entry['free'],entry['coherent'])
            score=entry['score'].numpy(); z=transform(score,choice['score_logit_mean'],choice['score_logit_std'])
            lambdas={'free':np.zeros(len(score)),'fixed_coherent':np.ones(len(score)),
                **{name+'_calibrated':step(np.array(choice[name]['coefficients']),z,settings['step_bounds']) for name in ['constant','score']}}
            outputs={}
            for method,lam in lambdas.items():
                prediction=entry['free'].to(torch.complex128)+torch.tensor(lam)[:,None,None]*(entry['coherent'].to(torch.complex128)-entry['free'].to(torch.complex128))
                actual=((prediction-entry['truth'].to(torch.complex128)).abs().square().sum((1,2))/entry['truth'].to(torch.complex128).abs().square().sum((1,2))).numpy()
                np.testing.assert_allclose(actual,t['before']+lam**2*t['c']-2*lam*t['b'],rtol=1e-9,atol=1e-12)
                outputs[method]=prediction
                for scene,gid in enumerate(entry['geometry_ids']):
                    metrics.append({'split':split,'context':context,'snr_db':contexts[context]['snr_db'],'method':method,
                        'scene':scene,'geometry_id':gid,'nmse_linear':float(actual[scene]),'step':float(lam[scene]),'score':float(score[scene])})
            for scene,gid in enumerate(entry['geometry_ids']):
                diagnostics.append({'split':split,'snr_db':contexts[context]['snr_db'],'scene':scene,'geometry_id':gid,
                    'score':float(score[scene]),'before':float(t['before'][scene]),'correction':float(t['c'][scene]),'inner_product':float(t['b'][scene]),
                    'oracle_step_diagnostic':float(t['b'][scene]/max(t['c'][scene],1e-30)),
                    'fixed_relative_gain':float((2*t['b'][scene]-t['c'][scene])/t['before'][scene])})
            return outputs
        for context in range(5):
            evaluate('train',context,training[context]); evaluate('validation',context,validation[context])
            saved_path=path/f'predictions/screen_{context}.pt'
            if resume and saved_path.is_file():
                entry=torch.load(saved_path,map_location='cpu',weights_only=True)
                assert entry['frozen_sha256']==file_hash(path/'frozen.json')
            else:
                entry=infer(screen_entries[context],ids,'screen',context)
            outputs=evaluate('screen',context,entry)
            torch.save({**entry,'outputs':outputs,'frozen_sha256':file_hash(path/'frozen.json')},path/f'predictions/screen_{context}.pt')
            write_csv(out/'metrics.csv',metrics); write_csv(out/'diagnostics.csv',diagnostics)
        assert all(file_hash(parent/n)==digest for n,digest in digests.items())
        assert file_hash(path/'frozen.json')==file_hash(out/'frozen.json')
        for name in ['config_resolved.json','environment.json','source_manifest.json','input_provenance.json']:
            shutil.copyfile(path/name,out/name)
        for name in ['metrics.csv','diagnostics.csv','validation.csv','frozen.json']:
            shutil.copyfile(out/name,path/'export'/name)
        write_json(out/'verification.json',{'status':'passed','parent_inputs_unchanged':True,'new_geometries_disjoint':True,
            'freeze_before_screen_generation':True,'quadratic_scores_match_full_CSI':True,'train_only_feature_statistics':True,
            'training_geometries':len(train_ids),'validation_geometries':len(validation[0]['geometry_ids']),'screen_geometries':len(ids)})
        status({'status':'complete','elapsed_seconds':prior_elapsed+time.monotonic()-start,
            'segment_seconds':time.monotonic()-start,'resumed_checkpoint':resume,'metric_rows':len(metrics)})
        log('complete')
    except BaseException as error:
        status({'status':'failed','error':repr(error),'elapsed_seconds':prior_elapsed+time.monotonic()-start})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--resume',action='store_true')
    arguments=parser.parse_args()
    if arguments.self_check:
        self_check(); print('calibration and independent-posterior checks passed')
    else:
        run(resume=arguments.resume)
