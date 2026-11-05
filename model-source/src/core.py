"""Shared, auditable A1-compatible preprocessing and MobileNetV3 inference."""
from pathlib import Path
import csv, json, random, time
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['give_way', 'main_road', 'parking', 'pedestrian_crossing']
MEAN = [.485, .456, .406]
STD = [.229, .224, .225]

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def device():
    return torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

def manifest():
    return list(csv.DictReader((ROOT/'data/original/manifest.csv').open()))

def letterbox(image):
    image=ImageOps.exif_transpose(image).convert('RGB')
    scale=256/max(image.size)
    size=tuple(max(1,round(v*scale)) for v in image.size)
    resized=image.resize(size,Image.Resampling.BICUBIC)
    square=Image.new('RGB',(256,256),(124,116,104))
    square.paste(resized,((256-size[0])//2,(256-size[1])//2))
    return square

def preprocess(image):
    image=letterbox(image).resize((224,224),Image.Resampling.BILINEAR)
    return transforms.functional.normalize(transforms.functional.to_tensor(image),MEAN,STD)

def load_model(checkpoint=None):
    model=models.mobilenet_v3_large(weights=None)
    model.classifier[-1]=nn.Linear(1280,4)
    checkpoint=checkpoint or ROOT/'artifacts/a1/best_deep_model.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state.get('model',state))
    return model.eval()

def corruption(image, condition, seed=42):
    """Deterministic stress tests; these are NOT generative-model outputs."""
    image=ImageOps.exif_transpose(image).convert('RGB')
    rng=random.Random(seed)
    if condition=='clean': return image
    if condition=='dark': return ImageEnhance.Brightness(image).enhance(.35)
    if condition=='blur': return image.filter(ImageFilter.GaussianBlur(radius=4))
    if condition=='low_resolution': return image.resize((40,40),Image.Resampling.BILINEAR).resize(image.size,Image.Resampling.BILINEAR)
    if condition=='jpeg':
        import io
        buf=io.BytesIO(); image.save(buf,format='JPEG',quality=15);buf.seek(0)
        return Image.open(buf).convert('RGB')
    if condition=='occlusion':
        image=image.copy();draw=ImageDraw.Draw(image)
        w,h=image.size;bw,bh=round(w*.18),round(h*.18)
        x=rng.randrange(max(1,w-bw));y=rng.randrange(max(1,h-bh))
        draw.rectangle((x,y,x+bw,y+bh),fill=(50,50,50));return image
    raise ValueError(condition)

CONDITIONS=['clean','dark','blur','low_resolution','jpeg','occlusion']

class WithGradCAM(nn.Module):
    """Exact Grad-CAM for this GAP->Linear->Hardswish->Linear classifier.

    Compute all four class gradients analytically so ONNX needs no autograd.
    Dropout is disabled in eval mode. CAMs are unnormalised, rectified 7x7 maps.
    """
    def __init__(self, model):
        super().__init__();self.features=model.features;self.classifier=model.classifier
    def forward(self,input):
        f=self.features(input)
        pooled=f.mean(dim=(2,3))
        z=self.classifier[0](pooled)
        logits=self.classifier[3](torch.nn.functional.hardswish(z))
        slope=torch.where(z<=-3,torch.zeros_like(z),torch.where(z>=3,torch.ones_like(z),z/3+.5))
        d=(slope[:,None,:]*self.classifier[3].weight[None,:,:])@self.classifier[0].weight
        cams=torch.relu(torch.einsum('bkc,bchw->bkhw',d,f)/(f.shape[2]*f.shape[3]))
        return logits,cams

def gradcam_overlay(model,image):
    x=preprocess(image).unsqueeze(0)
    with torch.no_grad(): logits,cams=WithGradCAM(model)(x)
    pred=int(logits.argmax());cam=cams[0,pred].numpy();cam/=max(float(cam.max()),1e-8)
    heat=Image.fromarray(np.uint8(cam*255)).resize((256,256),Image.Resampling.BILINEAR)
    v=np.asarray(heat)/255
    rgb=np.stack([np.minimum(1,2*v),np.maximum(0,1-abs(v-.5)*2),np.maximum(0,1-2*v)],axis=2)
    overlay=Image.blend(letterbox(image),Image.fromarray(np.uint8(rgb*255)),.42)
    return overlay,torch.softmax(logits,1)[0].numpy(),cam

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
