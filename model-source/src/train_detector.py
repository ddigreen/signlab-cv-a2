"""Train and evaluate real-only and MaskGen-composite YOLO11n detectors."""
from pathlib import Path
import os,argparse,csv,json,shutil,time,hashlib
from core import ROOT,CLASSES,write_json,corruption,CONDITIONS

os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'artifacts/ultralytics_config'))
os.environ.setdefault('WANDB_MODE','disabled')

def prepare():
    import yaml
    data=ROOT/'data/detection'
    if not (data/'train.txt').exists():raise RuntimeError('Reviewed real bounding boxes are required before training')
    reviewed=json.loads((data/'annotation_provenance.json').read_text())
    if reviewed.get('reviewed_frame_count')!=569:raise RuntimeError('Complete visual review of the569 real boxes is required before this fixed-dataset experiment')
    synthetic=list(csv.DictReader((ROOT/'data/synthetic/manifest.csv').open()))
    original=(data/'train.txt').read_text().splitlines()
    lines=[]
    for r in synthetic:
        if r['method']!='maskgen':continue
        image=ROOT/r['path'];x1,y1,x2,y2=[float(r[k]) for k in ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]
        label=f'{CLASSES.index(r["label"])} {(x1+x2)/512:.8f} {(y1+y2)/512:.8f} {(x2-x1)/256:.8f} {(y2-y1)/256:.8f}\n'
        image.with_suffix('.txt').write_text(label)
        lines.append('./../synthetic/maskgen/'+str(image.relative_to(ROOT/'data/synthetic/maskgen')))
    (data/'train_maskgen.txt').write_text('\n'.join(original+lines)+'\n')
    for arm,trainlist in [('real','train.txt'),('maskgen','train_maskgen.txt')]:
        config={'path':str(data.resolve()),'train':trainlist,'val':'val.txt','test':'test.txt','names':dict(enumerate(CLASSES))}
        (data/f'runtime_{arm}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))

def train(arm,epochs=20,dev='mps'):
    from ultralytics import YOLO
    data=ROOT/'data/detection';model_path=ROOT/'artifacts/pretrained/yolo11n.pt'
    if not model_path.exists():raise RuntimeError('Download official yolo11n.pt into artifacts/pretrained first')
    out=ROOT/'artifacts/detection'
    model=YOLO(str(model_path))
    model.train(data=str(data/f'runtime_{arm}.yaml'),epochs=epochs,imgsz=320,batch=8,device=dev,workers=0,seed=42,deterministic=True,
                project=str(out),name=arm,exist_ok=True,pretrained=True,freeze=10,optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,
                patience=8,fliplr=0.,flipud=0.,degrees=10.,hsv_h=.01,hsv_s=.3,hsv_v=.3,mosaic=.5,close_mosaic=5,mixup=0.,copy_paste=0.,
                amp=False,cache=False,plots=True,save=True,verbose=True)
    config={'architecture':'YOLO11n','pretrained_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'arm':arm,'seed':42,'imgsz':320,'epochs_max':epochs,'batch':8,'freeze_first_modules':10,'optimizer':'AdamW','lr0':.001,'selection':'Ultralytics validation fitness (mAP50-95)','best_validation_fitness':float(model.trainer.best_fitness),'test_used_for_selection':False,'annotation_scope':'dominant target sign per image','device':dev,'determinism_limit':'MPS scatter_reduce and index_put accumulation warn of nondeterministic implementations despite fixed seeds.'}
    write_json(out/arm/'experiment_config.json',config)

def evaluate(dev='cpu'):
    from ultralytics import YOLO
    from PIL import Image
    import yaml
    data=ROOT/'data/detection';out=ROOT/'artifacts/detection';comparison=[]
    validation={}
    for arm in ['real','maskgen']:
        config=json.loads((out/arm/'experiment_config.json').read_text())
        history=list(csv.DictReader((out/arm/'results.csv').open()))
        validation[arm]=float(config.get('best_validation_fitness',max(float(r['metrics/mAP50-95(B)']) for r in history)))
    selected_arm=max(['real','maskgen'],key=lambda arm:validation[arm])
    selection={'selected_arm':selected_arm,'criterion':'highest best validation mAP50-95 across the two arms; real-only tie break','validation_fitness':validation,'test_used':False}
    write_json(out/'deployment_selection.json',selection)
    for arm in ['real','maskgen']:
        checkpoint=out/arm/'weights/best.pt'
        model=YOLO(str(checkpoint))
        # Full confidence sweep for AP, not the camera display threshold.
        result=model.val(data=str(data/f'runtime_{arm}.yaml'),split='test',imgsz=320,batch=8,device=dev,workers=0,conf=.001,iou=.7,rect=False,
                         plots=True,save_json=True,project=str(out),name=f'test_{arm}',exist_ok=True,verbose=True)
        comparison.append({'arm':arm,'model':f'YOLO11n {arm}','condition':'clean','n':96,'map50':float(result.box.map50),'map50_95':float(result.box.map),'precision':float(result.box.mp),'recall':float(result.box.mr),'mean_ms':float(result.speed['inference'])})
        write_json(out/f'per_class_{arm}.json',{'classes':CLASSES,'map50_95':result.box.maps.tolist(),'speed_ms':result.speed,'results_dict':{k:float(v) for k,v in result.results_dict.items()}})
    # Stress evaluation uses the same reviewed real boxes and exact corruption definitions.
    test_list=(data/'test.txt').read_text().splitlines()
    for condition in ['dark','blur','low_resolution']:
        condition_root=out/'stress_data'/condition
        for rel in test_list:
            original=(data/rel).resolve();relative=original.relative_to(data/'images')
            image_path=condition_root/'images'/relative;image_path.parent.mkdir(parents=True,exist_ok=True)
            corruption(Image.open(original),condition,42).save(image_path,quality=95)
            label=data/'labels'/relative.with_suffix('.txt');dest=condition_root/'labels'/relative.with_suffix('.txt');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(label,dest)
        config={'path':str(condition_root.resolve()),'train':'images','val':'images','test':'images','names':dict(enumerate(CLASSES))}
        configpath=condition_root/'data.yaml';configpath.write_text(yaml.safe_dump(config,sort_keys=False))
        for arm in ['real','maskgen']:
            model=YOLO(str(out/arm/'weights/best.pt'))
            r=model.val(data=str(configpath),split='test',imgsz=320,batch=8,device=dev,workers=0,conf=.001,iou=.7,rect=False,plots=False,project=str(out),name=f'test_{arm}_{condition}',exist_ok=True,verbose=False)
            comparison.append({'arm':arm,'model':f'YOLO11n {arm}','condition':condition,'n':96,'map50':float(r.box.map50),'map50_95':float(r.box.map),'precision':float(r.box.mp),'recall':float(r.box.mr),'mean_ms':float(r.speed['inference'])})
    best=next(r for r in comparison if r['arm']==selected_arm and r['condition']=='clean')
    summary={'map50':best['map50'],'map50_95':best['map50_95'],'selected_arm':selected_arm,'selection':selection,'comparison':comparison,'environment':dev,
             'protocol_description':'Two YOLO11n models start from the same official COCO-pretrained weights: real-only and real plus 128 MaskGen-background composites. Each uses seed 42, 320 px inputs, batch 8, AdamW (lr0=0.001), 20 maximum epochs, 8-epoch patience and the first 10 modules frozen. Horizontal/vertical flips are disabled. Validation fitness selects the checkpoint; the 96-image test is evaluated after training. AP uses confidence threshold 0.001 and NMS IoU 0.7; the deployed display threshold is 0.25 and NMS IoU is 0.45 differ intentionally.',
             'notes':['Dominant target-sign boxes; not exhaustive dense street-scene annotation.','Codex visual overlay review; no independent human annotation audit.','Same 96 original test images for all conditions; inherited source dependence remains.','Synthetic composite boxes are exact visible alpha bounds; real boxes have annotation uncertainty.','Detector comparison tests the entire augmentation pipeline: 128 extra images, copy-paste, and extra optimizer steps are confounded with generated backgrounds.','Detector mAP and classifier accuracy measure different tasks.','One detector training seed per arm; no estimate of detector seed variation.']}
    write_json(ROOT/'artifacts/a2/detection.json',summary);write_json(ROOT/'web/results/detection.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

def export():
    from ultralytics import YOLO
    import onnx
    selection=json.loads((ROOT/'artifacts/detection/deployment_selection.json').read_text())
    selected_arm=selection['selected_arm']
    model=YOLO(str(ROOT/'artifacts/detection'/selected_arm/'weights/best.pt'))
    path=model.export(format='onnx',imgsz=320,opset=17,simplify=False,dynamic=False,nms=False,device='cpu',batch=1)
    destination=ROOT/'web/models/detector.onnx';destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    graph=onnx.load(destination);onnx.checker.check_model(graph)
    write_json(ROOT/'artifacts/a2/detector_export.json',{'selected_arm':selected_arm,'source':str(Path(path).relative_to(ROOT)),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'inputs':[{'name':v.name,'shape':[d.dim_value for d in v.type.tensor_type.shape.dim]} for v in graph.graph.input],'outputs':[{'name':v.name,'shape':[d.dim_value for d in v.type.tensor_type.shape.dim]} for v in graph.graph.output]})
    print(destination,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train-real','train-maskgen','evaluate','export']);p.add_argument('--epochs',type=int,default=20);p.add_argument('--device',default='mps');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage.startswith('train-'):train(a.stage.removeprefix('train-'),a.epochs,a.device)
    elif a.stage=='evaluate':evaluate(a.device)
    else:export()
