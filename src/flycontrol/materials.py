"""Export an editable controller overview and the actual numerical configuration tables."""
import csv
import json
from pathlib import Path
import yaml
from .paths import CONFIGS,sha256,write_json,source_digest


def flatten(value,prefix=''):
    if isinstance(value,dict):
        for key,child in value.items():yield from flatten(child,prefix+'.'+key if prefix else key)
    else:yield prefix,json.dumps(value,ensure_ascii=False)


def export(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    families={'S1_PPO':['figure2/protocol.json','figure3/protocol.json','figure4/protocol.json','figure6/experiments/*.json','maze/*base*.yaml','maze/*prerequisite*.yaml'],
              'S2_architectures':['maze/*base*.yaml'],
              'S3_maze_adaptation':['maze/*umaze*.yaml','maze/*medium*.yaml','maze/*large*.yaml','maze/prerequisites.json'],
              'S4_body_interfaces':['figure6/experiments/*.json']}
    sources={}
    for table,patterns in families.items():
        paths=sorted({p for pattern in patterns for p in (CONFIGS/'internal').glob(pattern)})
        if not paths:raise ValueError(f'No configuration sources for {table}')
        with (output/(table+'.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['configuration','setting','value','configuration_sha256'])
            for p in paths:
                name=str(p.relative_to(CONFIGS));sources[name]=sha256(p)
                for key,value in flatten(yaml.safe_load(p.read_text())):w.writerow([name,key,value,sources[name]])
    # Diagram is editable vector source; it depicts architecture, not measured data.
    boxes=[(20,'Scaled observations'),(260,'Learned input interface'),(500,'Fixed connectome'),(740,'Learned output interface'),(980,'Body / motor interface')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="220" viewBox="0 0 1220 220">','<rect width="1220" height="220" fill="white"/>','<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#31465a"/></marker></defs>']
    for i,(x,label) in enumerate(boxes):
        svg.append(f'<rect x="{x}" y="55" width="210" height="65" rx="6" fill="'+('#dce9df' if i==2 else '#e7edf4')+'" stroke="#31465a"/>')
        svg.append(f'<text x="{x+105}" y="93" text-anchor="middle" font-family="sans-serif" font-size="15">{label}</text>')
        if i<4:svg.append(f'<path d="M{x+210},87 H{x+237}" stroke="#31465a" marker-end="url(#arrow)"/>')
    svg.append('<path d="M1085,120 V170 H125 V120" fill="none" stroke="#31465a" marker-end="url(#arrow)"/><text x="610" y="195" text-anchor="middle" font-family="sans-serif" font-size="15">Closed-loop physical feedback · recurrent state evolves; anatomical graph stays fixed</text></svg>')
    (output/'F1_controller.svg').write_text('\n'.join(svg))
    write_json(output/'manifest.json',dict(source_sha256=source_digest(),configurations=sources,role='Configuration tables and architecture diagram; not experimental results',files={p.name:sha256(p) for p in output.iterdir() if p.suffix in ('.csv','.svg')}))
