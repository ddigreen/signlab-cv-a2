#!/usr/bin/env python3
"""Color/geometry-assisted dominant-target-sign annotations with explicit QA records.

This utility proposes boxes from image colors and connected components, applies
stored human/visual-review corrections, and exports YOLO labels for the exact A1
split. No trained classifier supplies annotations. Review QA sheets before using
the labels. Only the dominant sign matching the source folder is annotated;
small additional signs are outside this annotation protocol.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['give_way', 'main_road', 'parking', 'pedestrian_crossing']
RULE_CONFIG = ROOT / 'data/private/annotation_clip_rules.json'
CLIP_RULES = json.loads(RULE_CONFIG.read_text()) if RULE_CONFIG.is_file() else {}


def components_pil(mask, size=3):
    """Run-length 8-connected components, for PIL/NumPy-only environments."""
    clean=np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size)))>0
    parent=[];runs=[];previous=[]
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for y,row in enumerate(clean):
        transitions=np.diff(np.pad(row.astype(np.int8),(1,1)))
        starts=np.flatnonzero(transitions==1);ends=np.flatnonzero(transitions==-1)
        current=[];j=0
        for x1,x2 in zip(starts,ends):
            idx=len(parent);parent.append(idx);runs.append((int(x1),int(x2),y))
            while j<len(previous) and previous[j][1]<x1-1:j+=1
            k=j
            while k<len(previous) and previous[k][0]<=x2:
                if previous[k][1]>=x1:
                    a,b=root(idx),root(previous[k][2]);parent[a]=b
                k+=1
            current.append((x1,x2,idx))
        previous=current
    groups={}
    for idx,(x1,x2,y) in enumerate(runs):
        key=root(idx);area=x2-x1
        if key not in groups:groups[key]=[x1,y,x2,y+1,0,0.,0.]
        g=groups[key];g[0]=min(g[0],x1);g[1]=min(g[1],y);g[2]=max(g[2],x2);g[3]=max(g[3],y+1)
        g[4]+=area;g[5]+=(x1+x2-1)/2*area;g[6]+=y*area
    stats=[[0,0,0,0,0]];centers=[[0,0]]
    for x1,y1,x2,y2,area,sumx,sumy in groups.values():
        stats.append([x1,y1,x2-x1,y2-y1,area]);centers.append([sumx/area,sumy/area])
    return len(stats),None,np.array(stats),np.array(centers)


def propose(image, label, clip=None):
    rgb = np.asarray(image.convert('RGB'))
    if cv2 is not None and label == 'give_way' and clip in CLIP_RULES.get('bright_triangle_clips', []):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        bright=(gray>238).astype(np.uint8)*255
        bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(bright,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        triangles=[];rings=[]
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour);area=cv2.contourArea(contour)
            if area<500 or not .5<bw/bh<1.8 or bh>rgb.shape[0]*.57:continue
            poly=cv2.approxPolyDP(contour,.05*cv2.arcLength(contour,True),True)
            center_penalty=((x+bw/2)/rgb.shape[1]-.5)**2+((y+bh/2)/rgb.shape[0]-.43)**2
            score=area*math.exp(-12*center_penalty)
            rings.append([x,y,x+bw,y+bh])
            if len(poly)==3 and .28<area/(bw*bh)<.72:triangles.append((score,[x,y,x+bw,y+bh]))
        if triangles:
            triangles.sort(reverse=True);score,box=triangles[0]
            x1,y1,x2,y2=box;cx,cy=(x1+x2)/2,(y1+y2)/2;bw,bh=x2-x1,y2-y1
            expanded=[cx-bw*.53,cy-bh*.53,cx+bw*.53,cy+bh*.53]
            for ring in rings:
                rx1,ry1,rx2,ry2=ring
                if rx1<=cx<=rx2 and ry1<=cy<=ry2 and (rx2-rx1)<bw*1.8 and (ry2-ry1)<bh*1.8:
                    expanded=[min(expanded[0],rx1),min(expanded[1],ry1),max(expanded[2],rx2),max(expanded[3],ry2)]
            result=[max(0,round(expanded[0])),max(0,round(expanded[1])),min(rgb.shape[1],round(expanded[2])),min(rgb.shape[0],round(expanded[3]))]
            return result,{'score':round(score,3),'candidate_count':len(triangles),'proposal_type':'overexposed-triangle-brightness-contour'}
    if cv2 is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
    else:
        hsv=np.asarray(image.convert('HSV'))
        h=hsv[:,:,0].astype(np.float32)*180/255;s=hsv[:,:,1];v=hsv[:,:,2]
    height, width = h.shape
    if label == 'give_way':
        mask = (((h < 5) | (h > 166)) & (s > 55) & (v > 30)).astype(np.uint8)*255
    elif label == 'main_road':
        mask = ((h > 15) & (h < 43) & (s > 80) & (v > 100)).astype(np.uint8)*255
    elif label == 'parking':
        mask = ((h > 92) & (h < 137) & (s > 60) & (v > 35)).astype(np.uint8)*255
    else:
        mask = ((h > 89) & (h < 137) & (s > 95) & (v > 42)).astype(np.uint8)*255
    kernel_size=7 if label in ['parking','pedestrian_crossing'] else 3
    if cv2 is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size,kernel_size), np.uint8), iterations=1)
        count, components, stats, centers = cv2.connectedComponentsWithStats(mask)
    else:
        count, components, stats, centers=components_pil(mask,kernel_size)
    candidates = []
    for idx in range(1,count):
        x, y, bw, bh, area = stats[idx]
        cx, cy = centers[idx]
        if area < 40 or min(bw,bh) < 12 or not .35 < bw/bh < 2.2:
            continue
        if bw > width*.92 or bh > height*.85:
            continue
        if label == 'parking' and (cy/height > .62 or cy/height < .15 or bh > height*.48):
            continue
        if label == 'pedestrian_crossing' and bh > height*.50:
            continue
        center_penalty = ((cx/width-.5)**2 + (cy/height-.44)**2)
        # Targets are the visually dominant signs, approximately centered by capture.
        score = math.sqrt(bw*bh) * math.exp(-4*center_penalty)
        if label == 'give_way':
            score *= min(1., area/(bw*bh*.12))
        candidates.append((score, (int(x),int(y),int(x+bw),int(y+bh)),int(area)))
    if not candidates:
        raise ValueError(f'No color candidate for {label}')
    candidates.sort(reverse=True)
    score, (x1,y1,x2,y2), area = candidates[0]
    if label == 'parking' and (CLIP_RULES.get('parking_merge_exempt_clip') is None or clip != CLIP_RULES['parking_merge_exempt_clip']):
        # White P/text can disconnect navy components. Join nearby color pieces
        # with strong row/column overlap, without connecting the distant gate.
        for _ in range(4):
            changed=False
            for x,y,bw,bh,pixels in stats[1:]:
                if pixels < 50:continue
                xx,yy=x+bw,y+bh
                overlap_y=max(0,min(y2,yy)-max(y1,y));overlap_x=max(0,min(x2,xx)-max(x1,x))
                gapx=max(0,max(x1,x)-min(x2,xx));gapy=max(0,max(y1,y)-min(y2,yy))
                horizontal=overlap_y>.45*min(y2-y1,bh) and gapx<.85*max(x2-x1,bw)
                vertical=overlap_x>.45*min(x2-x1,bw) and gapy<.17*max(y2-y1,bh)
                if horizontal or vertical:
                    new=(min(x1,x),min(y1,y),max(x2,xx),max(y2,yy))
                    if not .58 < (new[2]-new[0])/(new[3]-new[1]) < 1.5:continue
                    if new!=(x1,y1,x2,y2):x1,y1,x2,y2=new;changed=True
            if not changed:break
    if label == 'main_road':
        # Yellow marks the inner diamond; estimate its surrounding white board.
        cx,cy=(x1+x2)/2,(y1+y2)/2
        bw,bh=(x2-x1)*1.65,(y2-y1)*1.65
        x1,y1,x2,y2=round(cx-bw/2),round(cy-bh/2),round(cx+bw/2),round(cy+bh/2)
    else:
        # Color mask excludes the slim white rim of the physical sign board.
        pad=max(2,round(max(x2-x1,y2-y1)*.025))
        x1,y1,x2,y2=x1-pad,y1-pad,x2+pad,y2+pad
    box = [int(max(0,x1)),int(max(0,y1)),int(min(width,x2)),int(min(height,y2))]
    return box, {'score':round(score,3),'candidate_count':len(candidates),'color_pixels':area}


def sheets(records, directory):
    directory.mkdir(parents=True,exist_ok=True)
    membership = {}
    for start in range(0,len(records),30):
        number=start//30+1
        sheet=Image.new('RGB',(1440,1300),'#f5f5f5');d=ImageDraw.Draw(sheet)
        group=records[start:start+30]
        for k,r in enumerate(group):
            cellx=(k%6)*240;celly=(k//6)*260
            im=Image.open(ROOT/r['path']).convert('RGB');box=r['bbox_xyxy']
            d.text((cellx+4,celly+3),f"{r['image_id']}",fill='black')
            d.text((cellx+4,celly+15),f"{r['label']}  {r['split']}",fill='black')
            thumb=im.copy();thumb.thumbnail((130,225))
            sx,sy=thumb.width/im.width,thumb.height/im.height
            td=ImageDraw.Draw(thumb);td.rectangle([box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy],outline='#00ff50',width=2)
            sheet.paste(thumb,(cellx+3,celly+30))
            x1,y1,x2,y2=box;bw,bh=x2-x1,y2-y1
            margin=round(max(bw,bh)*.18)
            cropbox=[max(0,x1-margin),max(0,y1-margin),min(im.width,x2+margin),min(im.height,y2+margin)]
            crop=im.crop(cropbox);scale=min(100/crop.width,180/crop.height)
            crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))))
            cd=ImageDraw.Draw(crop);cd.rectangle([(x1-cropbox[0])*scale,(y1-cropbox[1])*scale,(x2-cropbox[0])*scale,(y2-cropbox[1])*scale],outline='#00ff50',width=1)
            sheet.paste(crop,(cellx+137,celly+55))
            d.text((cellx+137,celly+240),f"#{start+k+1}",fill='#333333')
            membership[r['image_id']]=f'qa_{number:02d}.jpg'
        sheet.save(directory/f'qa_{number:02d}.jpg',quality=93)
    return membership


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-sheets',default='',help='Comma-separated sheet numbers actually visually inspected')
    parser.add_argument('--repropose',action='store_true',help='Replace stored proposals; all altered sheets require new review')
    args=parser.parse_args()
    destination=ROOT/'data/detection';destination.mkdir(parents=True,exist_ok=True)
    corrections_path=destination/'corrections.json'
    corrections=json.loads(corrections_path.read_text()) if corrections_path.exists() else {}
    previous_file=destination/'annotations.json'
    previous={r['image_id']:r for r in json.loads(previous_file.read_text())} if previous_file.exists() and not args.repropose else {}
    rows=list(csv.DictReader((ROOT/'data/original/manifest.csv').open()))
    rows.sort(key=lambda r: (r['clip'],r['path']))
    records=[]
    for r in rows:
        image=Image.open(ROOT/r['path']).convert('RGB')
        key=Path(r['path']).stem
        if key in corrections:
            box=list(corrections[key]);extra={'method':'visual-correction'}
        elif key in previous:
            box=previous[key]['bbox_xyxy'].copy()
            extra={k:previous[key][k] for k in ['method','score','candidate_count','proposal_type','needs_review'] if k in previous[key]}
        else:
            try:
                box,extra=propose(image,r['label'],r['clip']);extra['method']='HSV-connected-component-proposal'
            except ValueError:
                print('No color proposal:',r['path'],flush=True)
                # An explicit neighbor proposal is reviewable; it is never called
                # verified until its contact sheet has actually been inspected.
                neighbors=[p for p in records if p['clip']==r['clip']]
                if not neighbors:raise
                box=neighbors[-1]['bbox_xyxy'].copy();extra={'method':'previous-frame-box-proposal','needs_review':True}
        width,height=image.size
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError((key,box,image.size))
        records.append({**r,'image_id':key,'width':width,'height':height,'bbox_xyxy':box,'class_id':CLASSES.index(r['label']),**extra})
    qa_members=sheets(records,destination/'qa')
    explicit_reviewed={f'qa_{int(v):02d}.jpg' for v in args.reviewed_sheets.split(',') if v.strip()}
    reviewed=set(explicit_reviewed)
    old_provenance=destination/'annotation_provenance.json'
    if old_provenance.exists() and not args.repropose:
        reviewed.update(json.loads(old_provenance.read_text()).get('reviewed_sheets',[]))
    changed_sheets={qa_members[r['image_id']] for r in records if r['image_id'] in previous and r['bbox_xyxy']!=previous[r['image_id']]['bbox_xyxy']}
    reviewed.difference_update(changed_sheets-explicit_reviewed)
    for r in records:
        source=ROOT/r['path'];relative=Path(r['label'])/source.name
        target=destination/'images'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            try:os.link(source,target)
            except OSError:shutil.copy2(source,target)
        label_path=destination/'labels'/relative.with_suffix('.txt');label_path.parent.mkdir(parents=True,exist_ok=True)
        x1,y1,x2,y2=r['bbox_xyxy'];w,h=r['width'],r['height']
        label_path.write_text(f"{r['class_id']} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}\n")
        r['qa_sheet']=qa_members[r['image_id']]
        r['review_status']='visually-inspected-on-contact-sheet' if r['qa_sheet'] in reviewed else 'unreviewed-proposal'
        r['image_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    for split in ['train','val','test']:
        selected=[f"./images/{r['label']}/{Path(r['path']).name}" for r in records if r['split']==split]
        (destination/f'{split}.txt').write_text('\n'.join(selected)+'\n')
    (destination/'data.yaml').write_text('train: train.txt\nval: val.txt\ntest: test.txt\nnames:\n'+''.join(f'  {i}: {v}\n' for i,v in enumerate(CLASSES)))
    (destination/'annotations.json').write_text(json.dumps(records,indent=2)+'\n')
    provenance={
        'annotation_scope':'One dominant target sign matching the original image-folder class per frame; physical sign board boundary, excluding post.',
        'not_exhaustive_scene_annotations':True,
        'limitation':'Additional distant/small signs (including other modeled classes) are not exhaustively annotated. Detection mAP is for the dominant-target protocol, not general-scene detection.',
        'proposal_method':'HSV thresholding + connected-component color/geometry scoring; nearby navy fragments merged for wall parking signs; brightness/triangle contours for overexposed give-way clips. main_road inner-yellow component expanded to white board. Proposals inspected and explicit corrections retained.',
        'no_classifier_pseudolabels':True,
        'correction_file':'corrections.json',
        'visual_review_method':'Every frame is displayed both as full image with proposed box and padded target crop in contact sheets. Review status only marks sheets actually inspected.',
        'reviewer':'Codex multimodal visual inspection of all reviewed contact-sheet tiles, with full-resolution inspection for corrected ambiguous frames. No independent human annotation audit.',
        'image_count':len(records),'class_counts':dict(Counter(r['label'] for r in records)),
        'split_counts':dict(Counter(r['split'] for r in records)),
        'reviewed_sheets':sorted(reviewed),'reviewed_frame_count':sum(r['review_status']!='unreviewed-proposal' for r in records),
        'manual_correction_count':len(corrections),
        'source_manifest':'data/original/manifest.csv','split_changed':False,
        'train_only_synthetic_images_in_this_export':False,
        'yolo_format':'class_id x_center/width y_center/height box_width/width box_height/height',
        'class_order':CLASSES,
        'geometry_precision':'Axis-aligned boxes visually reviewed on contact sheets; not pixel-level polygon annotation. Small rim/blur boundary uncertainty remains.',
        'qa_records':'annotations.json and qa/qa_*.jpg',
    }
    (destination/'annotation_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:provenance[k] for k in ['image_count','reviewed_frame_count','manual_correction_count','split_counts']},indent=2))


if __name__=='__main__':main()
