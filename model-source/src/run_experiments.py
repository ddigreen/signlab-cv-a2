"""Run measured experiments. No reported score is hardcoded.

Protocol: A1 backbone is frozen; every new arm starts from the same A1 head.
Original split is retained. Select an epoch by validation macro-F1 (loss tie).
Seed42 is the predefined demo model; seeds43/44 measure head-training variation.
All augmentation and synthetic imagery are train-only.
"""
import argparse,copy,csv,json,time,platform,hashlib,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from core import *

CACHE=ROOT/'artifacts/cache'
RESULTS=ROOT/'artifacts/a2'

def metrics(y,p):
    precision,recall,f1,_=precision_recall_fscore_support(y,p,labels=range(4),average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y,p)),'macro_precision':float(precision),'macro_recall':float(recall),'macro_f1':float(f1),'n':len(y)}

def extract(model,images,dev,batch=8):
    outputs=[];started=time.perf_counter()
    for start in range(0,len(images),batch):
        batch_tensor=torch.stack([preprocess(im) for im in images[start:start+batch]]).to(dev)
        with torch.inference_mode():f=model.features(batch_tensor).mean((2,3)).cpu()
        outputs.append(f)
        if start%160==0:print(f'features {start}/{len(images)} [{dev}]',flush=True)
    return torch.cat(outputs),time.perf_counter()-started

def cache_real(refresh=False):
    CACHE.mkdir(parents=True,exist_ok=True);seed_all();dev=device();model=load_model().to(dev)
    rows=manifest();images=[Image.open(ROOT/r['path']).convert('RGB') for r in rows]
    if (CACHE/'real.pt').exists() and not refresh:
        cached=torch.load(CACHE/'real.pt',weights_only=False)
        assert cached['rows']==rows
        features=cached['features'];seconds=None
        print('Reusing completed original-image feature cache',flush=True)
    else:
        features,seconds=extract(model,images,dev)
    y=torch.tensor([CLASSES.index(r['label']) for r in rows]);splits=[r['split'] for r in rows]
    torch.save({'features':features,'y':y,'splits':splits,'rows':rows},CACHE/'real.pt')
    # Fresh conventional augmentation, two deterministic views per original train image.
    augment=transforms.Compose([transforms.RandomRotation(12,fill=(124,116,104)),transforms.ColorJitter(brightness=.35,contrast=.3,saturation=.2),transforms.RandomApply([transforms.GaussianBlur(5,sigma=(.1,1.5))],p=.35)])
    augmented=[];labels=[]
    for i,(r,im) in enumerate(zip(rows,images)):
        if r['split']!='train':continue
        for view in range(2):
            seed_all(42+i*2+view);augmented.append(augment(im));labels.append(CLASSES.index(r['label']))
    aug_features,aug_sec=extract(model,augmented,dev)
    torch.save({'features':aug_features,'y':torch.tensor(labels)},CACHE/'conventional.pt')
    stress={};test_indices=[i for i,s in enumerate(splits) if s=='test']
    for condition in CONDITIONS:
        if condition=='clean':stress[condition]=features[test_indices];continue
        transformed=[corruption(images[i],condition,42+i) for i in test_indices]
        stress[condition],_=extract(model,transformed,dev)
    torch.save({'features':stress,'y':y[test_indices],'rows':[rows[i] for i in test_indices]},CACHE/'test_conditions.pt')
    with torch.inference_mode():logits=model.classifier(features.to(dev)).cpu()
    check={s:metrics(y[[i for i,v in enumerate(splits) if v==s]],logits[[i for i,v in enumerate(splits) if v==s]].argmax(1)) for s in ['train','val','test']}
    env={'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'device':str(dev),'feature_seconds':seconds,'augmentation_feature_seconds':aug_sec,'a1_reproduced':check,'seed':42,'source_checkpoint_sha256':hashlib.sha256((ROOT/'artifacts/a1/best_deep_model.pt').read_bytes()).hexdigest()}
    write_json(RESULTS/'environment.json',env);print(json.dumps(check,indent=2),flush=True)

def cache_synthetic():
    rows=list(csv.DictReader((ROOT/'data/synthetic/manifest.csv').open()))
    dev=device();model=load_model().to(dev)
    for arm in ['flat_control','maskgen']:
        selected=[r for r in rows if r['method']==arm]
        images=[Image.open(ROOT/r['path']).convert('RGB') for r in selected]
        f,sec=extract(model,images,dev)
        torch.save({'features':f,'y':torch.tensor([CLASSES.index(r['label']) for r in selected]),'rows':selected},CACHE/f'{arm}.pt')
    print('Synthetic feature caching complete',flush=True)

def train(epochs=24):
    RESULTS.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    real=torch.load(CACHE/'real.pt',weights_only=False);conv=torch.load(CACHE/'conventional.pt',weights_only=False)
    train_idx=[i for i,s in enumerate(real['splits']) if s=='train'];val_idx=[i for i,s in enumerate(real['splits']) if s=='val']
    base_x=torch.cat([real['features'][train_idx],conv['features']]);base_y=torch.cat([real['y'][train_idx],conv['y']])
    val_x=real['features'][val_idx];val_y=real['y'][val_idx]
    initial=load_model().classifier.state_dict();summary=[];history=[]
    config={'epochs':epochs,'seeds':[42,43,44],'demo_seed':42,'batch_size':32,'learning_rate':.00002,'weight_decay':.0001,'label_smoothing':.05,'backbone':'frozen A1 MobileNetV3-Large','selection':'validation macro-F1; validation cross-entropy tie break','conventional_views_per_train_image':2,'arms':['real_control','flat_control','maskgen'],'test_used_for_selection':False}
    write_json(RESULTS/'experiment_config.json',config)
    for arm in config['arms']:
        x,y=base_x,base_y
        if arm!='real_control':
            synth=torch.load(CACHE/f'{arm}.pt',weights_only=False);x=torch.cat([x,synth['features']]);y=torch.cat([y,synth['y']])
        for seed in config['seeds']:
            seed_all(seed);head=load_model().classifier;head.load_state_dict(copy.deepcopy(initial))
            optimizer=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            loss_fn=nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
            dl=DataLoader(TensorDataset(x,y),batch_size=config['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
            best=(-1,float('-inf'));best_state=None;started=time.perf_counter()
            for epoch in range(epochs+1):
                if epoch:
                    head.train();loss_sum=0;correct=0
                    for bx,by in dl:
                        optimizer.zero_grad();out=head(bx);loss=loss_fn(out,by);loss.backward();optimizer.step()
                        loss_sum+=float(loss.detach())*len(by);correct+=int((out.argmax(1)==by).sum())
                    train_loss=loss_sum/len(y);train_acc=correct/len(y)
                else: train_loss=None;train_acc=None
                head.eval()
                with torch.inference_mode():v=head(val_x);val_loss=float(nn.functional.cross_entropy(v,val_y));vm=metrics(val_y,v.argmax(1))
                row={'arm':arm,'seed':seed,'epoch':epoch,'train_loss':train_loss,'train_accuracy':train_acc,'val_loss':val_loss,'val_accuracy':vm['accuracy'],'val_macro_f1':vm['macro_f1']};history.append(row)
                # Epoch0 is included: returning the A1 checkpoint is a valid honest outcome.
                key=(vm['macro_f1'],-val_loss)
                if key>best:best=key;best_state=copy.deepcopy(head.state_dict());best_epoch=epoch
            checkpoint={'classifier':best_state,'arm':arm,'seed':seed,'best_epoch':best_epoch,'classes':CLASSES}
            torch.save(checkpoint,RESULTS/f'head_{arm}_{seed}.pt')
            summary.append({'arm':arm,'seed':seed,'selected_epoch':best_epoch,'validation_macro_f1':best[0],'training_seconds':time.perf_counter()-started,'train_feature_views':len(y)})
            print(json.dumps(summary[-1]),flush=True)
    write_csv(RESULTS/'training_history.csv',history);write_csv(RESULTS/'selected_epochs.csv',summary)

def model_with_head(arm,seed=42):
    model=load_model()
    if arm!='baseline':model.classifier.load_state_dict(torch.load(RESULTS/f'head_{arm}_{seed}.pt',weights_only=True)['classifier'])
    return model.eval()

def evaluate():
    test=torch.load(CACHE/'test_conditions.pt',weights_only=False);rows=[];predictions=[]
    for arm in ['baseline','real_control','flat_control','maskgen']:
        for seed in ([42] if arm=='baseline' else [42,43,44]):
            model=model_with_head(arm,seed)
            with torch.inference_mode():
                for condition,f in test['features'].items():
                    logits=model.classifier(f);prob=logits.softmax(1);pred=logits.argmax(1)
                    row={'arm':arm,'seed':seed,'condition':condition,**metrics(test['y'],pred)};rows.append(row)
                    for i,meta in enumerate(test['rows']):predictions.append({'arm':arm,'seed':seed,'condition':condition,'path':meta['path'],'clip':meta['clip'],'true':CLASSES[int(test['y'][i])],'predicted':CLASSES[int(pred[i])],'confidence':float(prob[i].max())})
    write_csv(RESULTS/'metrics.csv',rows);write_csv(RESULTS/'predictions.csv',predictions)
    # End-to-end CPU latency, model execution only; preprocessing / Grad-CAM separate.
    model=model_with_head('maskgen');torch.set_num_threads(4);x=preprocess(Image.open(ROOT/test['rows'][0]['path'])).unsqueeze(0)
    with torch.inference_mode():
        for _ in range(5):model(x)
        times=[]
        for _ in range(30):t=time.perf_counter();model(x);times.append((time.perf_counter()-t)*1000)
        wrapped=WithGradCAM(model);cam_times=[]
        for _ in range(30):t=time.perf_counter();wrapped(x);cam_times.append((time.perf_counter()-t)*1000)
    latency={'device':'Apple M1 CPU, 4 PyTorch threads','batch_size':1,'warmups':5,'repetitions':30,'prediction_mean_ms':float(np.mean(times)),'prediction_std_ms':float(np.std(times)),'prediction_plus_all_class_cam_mean_ms':float(np.mean(cam_times)),'scope':'model execution; excludes decode/preprocessing/UI','browser_latency':'measured live in app; not interchangeable with Python CPU timing'}
    write_json(RESULTS/'latency.json',latency)
    aggregate=[]
    for arm in ['baseline','real_control','flat_control','maskgen']:
        for condition in CONDITIONS:
            selected=[r for r in rows if r['arm']==arm and r['condition']==condition]
            aggregate.append({'model':arm,'condition':condition,'n':96,'seeds':len(selected),'accuracy':float(np.mean([r['accuracy'] for r in selected])),'accuracy_std':float(np.std([r['accuracy'] for r in selected])),'macro_f1':float(np.mean([r['macro_f1'] for r in selected])),'macro_f1_std':float(np.std([r['macro_f1'] for r in selected]))})
    selections=list(csv.DictReader((RESULTS/'selected_epochs.csv').open()))
    write_json(RESULTS/'summary.json',{'comparison':aggregate,'latency':latency,'demo_seed':42,'dataset':{'real_images':569,'train_images':353,'validation_images':120,'test_images':96,'synthetic_images':128},'training_selection':selections,'notes':['All scores measured on the original fixed 96-image test set.','All 9 new arm/seed combinations retained epoch0: no classifier validation improvement; A2 selected classifier retains A1 weights.','A2 means/std reflect 3 classifier-head training seeds only.','Historical A1 reported95/96; this rerun yields 94/96. Comparisons consistently use the measured rerun, without claiming exact reproduction.','Original main_road videos cross splits; physical signs repeat.','MaskGen synthesises backgrounds; real TRAIN sign cutouts preserve labels.','Flat-background copy-paste control isolates the generated-background contribution.','Six deterministic conditions reuse the same96 originals and are correlated.']})
    dest=ROOT/'web/results';dest.mkdir(parents=True,exist_ok=True);(dest/'summary.json').write_bytes((RESULTS/'summary.json').read_bytes())
    print(json.dumps(aggregate,indent=2),flush=True)

def export():
    import onnx
    path=ROOT/'web/models';path.mkdir(parents=True,exist_ok=True)
    parity=[]
    original_manifest = ROOT / 'data/original/manifest.csv'
    example = ROOT / manifest()[0]['path'] if original_manifest.exists() else ROOT / 'examples/give_way.jpg'
    x = preprocess(Image.open(example)).unsqueeze(0)
    for name,arm in [('baseline','baseline'),('enhanced','maskgen')]:
        model=model_with_head(arm);wrapped=WithGradCAM(model).eval()
        # Compare analytical Grad-CAM against ordinary PyTorch autograd.
        feat=model.features(x).detach().requires_grad_(True);logits=model.classifier(feat.mean((2,3)))
        with torch.no_grad():ref_logits,ref_cams=wrapped(x)
        for cls in range(4):
            grad=torch.autograd.grad(logits[0,cls],feat,retain_graph=True)[0]
            cam=torch.relu((grad.mean((2,3),keepdim=True)*feat).sum(1))
            err=float((cam-ref_cams[:,cls]).abs().max())
            assert err<1e-5,err;parity.append({'model':name,'class':cls,'max_cam_error':err})
        torch.onnx.export(wrapped,x,path/f'{name}.onnx',input_names=['input'],output_names=['logits','cams'],opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(path/f'{name}.onnx'))
    write_json(RESULTS/'gradcam_validation.json',parity)
    full=model_with_head('maskgen');torch.save({'model':full.state_dict(),'classes':CLASSES,'arch':'MobileNetV3 (transfer)','a2_arm':'maskgen','seed':42},RESULTS/'enhanced_model.pt')
    print('Exported both ONNX models; exact Grad-CAM checked against autograd',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cache-real','cache-synthetic','train','evaluate','export','all']);parser.add_argument('--epochs',type=int,default=24);parser.add_argument('--refresh-real',action='store_true');args=parser.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True)
    if args.stage in ['cache-real','all']:cache_real(args.refresh_real)
    if args.stage in ['cache-synthetic','all']:cache_synthetic()
    if args.stage in ['train','all']:train(args.epochs)
    if args.stage in ['evaluate','all']:evaluate()
    if args.stage in ['export','all']:export()
