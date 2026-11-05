"""Class-preserving composites on real MaskGen-generated backgrounds.

Only manually outlined TRAIN signs are reused. One background is used for all
four classes, avoiding a one-background-per-class shortcut. No held-out crop is
used. A flat-background control isolates the background-generator contribution.
"""
from pathlib import Path
import argparse, json, hashlib, random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from core import ROOT,CLASSES,manifest,write_csv,write_json

CUTOUT_CONFIG = ROOT / 'data/private/cutouts.json'

def cutouts(specs):
    train={Path(r['path']).name for r in manifest() if r['split']=='train'}
    result={}
    for cls,spec in specs.items():
        assert spec['file'] in train,'Training-only provenance violated'
        image=Image.open(ROOT/'data/original'/cls/spec['file']).convert('RGBA')
        mask=Image.new('L',image.size,0);ImageDraw.Draw(mask).polygon(spec['polygon'],fill=255)
        mask=mask.filter(ImageFilter.GaussianBlur(.4));image.putalpha(mask)
        result[cls]=image.crop(mask.getbbox())
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--background-dir',type=Path,default=ROOT/'data/maskgen_backgrounds');args=parser.parse_args()
    backgrounds=sorted(args.background_dir.glob('*.png'))
    if not backgrounds: raise RuntimeError('No genuine MaskGen backgrounds; run generate_maskgen.py first')
    provenance=args.background_dir/'provenance.json'
    if not provenance.exists():raise RuntimeError('Missing generation provenance; refusing untraceable inputs')
    if not CUTOUT_CONFIG.is_file():
        raise RuntimeError('Supply private training cutout definitions at data/private/cutouts.json')
    cutout_specs = json.loads(CUTOUT_CONFIG.read_text())
    if set(cutout_specs) != set(CLASSES):
        raise ValueError('A cutout definition is required for each sign class')
    cuts = cutouts(cutout_specs)
    rows = []
    for bi,bgpath in enumerate(backgrounds):
        for ci,cls in enumerate(CLASSES):
            for variant in range(4):
                local_seed=42000+bi*100+variant # SAME geometry for every class
                rng=random.Random(local_seed)
                cut=cuts[cls].copy();target=round(rng.uniform(.25,.57)*256)
                scale=target/max(cut.size);cut=cut.resize(tuple(max(1,round(v*scale)) for v in cut.size),Image.Resampling.LANCZOS)
                cut=cut.rotate(rng.uniform(-8,8),resample=Image.Resampling.BICUBIC,expand=True)
                x=rng.randint(12,max(12,244-cut.width));y=rng.randint(15,max(15,230-cut.height))
                for mode in ['maskgen','flat_control']:
                    bg=Image.open(bgpath).convert('RGB').resize((256,256)) if mode=='maskgen' else Image.new('RGB',(256,256),(72,77,83))
                    bg.paste(cut,(x,y),cut)
                    # Modest partial occlusion in one quarter, identical across modes.
                    if variant==3:
                        ImageDraw.Draw(bg).rectangle((x+cut.width*.45,y+cut.height*.43,x+cut.width*.60,y+cut.height*.65),fill=(60,62,65))
                    if variant==1: bg=ImageEnhance.Brightness(bg).enhance(.65)
                    if variant==2: bg=bg.filter(ImageFilter.GaussianBlur(.7))
                    path=ROOT/'data/synthetic'/mode/cls/f'bg{bi:02d}_v{variant}.jpg';path.parent.mkdir(parents=True,exist_ok=True);bg.save(path,quality=95)
                    alpha_box=cut.getchannel('A').getbbox()
                    x1,y1,x2,y2=x+alpha_box[0],y+alpha_box[1],x+alpha_box[2],y+alpha_box[3]
                    rows.append({'path':str(path.relative_to(ROOT)),'label':cls,'split':'train','method':mode,'background':bgpath.name,'source_train_crop':cutout_specs[cls]['file'],'seed':local_seed,'occluded':variant==3,'bbox_x1':x1,'bbox_y1':y1,'bbox_x2':x2,'bbox_y2':y2,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    write_csv(ROOT/'data/synthetic/manifest.csv',rows)
    write_json(ROOT/'data/synthetic/cutout_annotations.json',cutout_specs)
    write_json(ROOT/'data/synthetic/README.json',{'method':'MaskGen-VQ-L + TA-TiTok128 background synthesis; manually masked real TRAIN sign copy-paste','no_pure_sign_generation_claim':True,'backgrounds':len(backgrounds),'per_mode':len(rows)//2,'cutout_sources':4,'synthetic_bbox_coordinates_available':True,'manual_cutout_polygons_are_not_full_dataset_groundtruth':True,'generator_provenance':str(provenance.relative_to(ROOT)),'synthetic_train_only':True,'geometry_shared_across_classes':True})
    print(f'Created {len(rows)} train-only images ({len(rows)//2} per mode)',flush=True)
if __name__=='__main__':main()
