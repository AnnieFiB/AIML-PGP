"""Commit-triggered batch scoring for ChurnGuard. Uses notebook-exported model/schema."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import joblib
import pandas as pd

BOUNDS = {'tenure':(0,99),'city_tier':(1,3),'cc_contacted_ly':(4,132),'service_score':(0,5),'account_user_count':(1,6),'cc_agent_score':(1,5),'rev_per_month':(1,140),'complain_ly':(0,1),'rev_growth_yoy':(4,28),'coupon_used_for_payment':(0,16),'day_since_cc_connect':(0,47),'cashback':(0,1997)}
CATEGORIES = {'payment':['cash on delivery','credit card','debit card','e wallet','upi'],'gender':['female','male'],'account_segment':['hni','regular','regular plus','super','super plus'],'marital_status':['divorced','married','single'],'login_device':['computer','mobile','others']}
ACTIONS = {'Critical':'Immediate personalized outreach, complaint resolution and tailored incentives','High':'Proactive engagement, targeted retention offers and loyalty incentives','Moderate':'Preventive engagement and service improvement','Low':'Routine monitoring and customer engagement'}


def changed_files(before, after):
    if not before or set(before) == {'0'}:
        args = ['git','ls-tree','-r','--name-only',after,'--','data/']
    else:
        args = ['git','diff','--name-status','--diff-filter=AMR',before,after,'--','data/']
    result = subprocess.check_output(args,text=True)
    if not before or set(before)=={'0'}:
        paths = result.splitlines()
    else:
        paths = [line.split('\t')[-1] for line in result.splitlines()]
    return sorted({p for p in paths if p.lower().endswith(('.csv','.xlsx')) and Path(p).is_file() and Path(p).parts[0]=='data'})


def read_table(path):
    raw = path.read_bytes()
    if not raw:
        raise ValueError('Empty file')
    if raw.startswith(b'PK\x03\x04'):
        return pd.read_excel(io.BytesIO(raw),engine='openpyxl',dtype={'accountid':'string'})
    if raw.startswith(b'\xd0\xcf\x11\xe0'):
        raise ValueError('Legacy .xls not supported')
    for encoding in ('utf-8-sig','utf-16','cp1252','latin-1'):
        try:
            df = pd.read_csv(io.StringIO(raw.decode(encoding)),sep=None,engine='python',dtype={'accountid':'string'})
            if len(df.columns)>1:
                return df
        except (UnicodeError,pd.errors.ParserError,ValueError):
            pass
    raise ValueError('Unable to parse CSV/XLSX')


def score(df, model, schema):
    df=df.copy()
    df.columns=[re.sub(r'[\s-]+','_',str(c).strip().lower()) for c in df.columns]
    if df.columns.duplicated().any() or df.empty:
        raise ValueError('Duplicate columns or no rows')
    if 'accountid' not in df:
        raise ValueError('Missing accountid')
    ids=df['accountid'].astype('string').str.strip()
    if ids.isna().any() or ids.eq('').any() or ids.duplicated().any():
        raise ValueError('Blank or duplicate accountid')
    features=schema['features']
    missing=[f for f in features if f not in df.columns]
    if missing:
        raise ValueError('Missing features: '+', '.join(missing))
    x=df[features].copy()
    for f in schema['numeric']:
        x[f]=pd.to_numeric(x[f],errors='coerce')
        if f in BOUNDS:
            lo,hi=BOUNDS[f]
            if (~x[f].dropna().between(lo,hi)).any():
                raise ValueError(f'{f} must be between {lo} and {hi}')
    for f in schema['categories']:
        x[f]=x[f].astype('string').str.strip().str.lower().astype(object)
        if f in CATEGORIES:
            invalid=x[f].dropna()[~x[f].dropna().isin(CATEGORIES[f])].unique().tolist()
            if invalid:
                raise ValueError(f'Unsupported {f}: {invalid}')
    probability=model.predict_proba(x)[:,1]
    risk=pd.cut(probability,[-.001,.30,.60,.80,1],labels=['Low','Moderate','High','Critical']).astype(str)
    result=pd.DataFrame({'accountid':ids.to_numpy(),'Churn Probability':probability,'Risk Segment':risk})
    result['Revenue at Risk']=x['rev_per_month'].to_numpy()*probability
    result['Recommended Action']=[ACTIONS[r] for r in risk]
    return result.sort_values('Churn Probability',ascending=False).reset_index(drop=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',default=os.getenv('BEFORE_SHA',''))
    p.add_argument('--after',default=os.getenv('GITHUB_SHA','HEAD'))
    p.add_argument('--all',action='store_true',help='Process all CSV/XLSX under data/')
    args=p.parse_args()
    files=sorted(str(p) for p in Path('data').rglob('*') if p.is_file() and p.suffix.lower() in ('.csv','.xlsx')) if args.all else changed_files(args.before,args.after)
    output=Path('outputs/automation');output.mkdir(parents=True,exist_ok=True)
    if not files:
        print('No added/modified CSV/XLSX files; nothing to score.')
        return
    schema=json.loads(Path('models/feature_schema.json').read_text(encoding='utf-8'))
    model=joblib.load(Path('models')/schema['model_file'])
    report=[]; failures=[]
    for name in files:
        try:
            path=Path(name)
            result=score(read_table(path),model,schema)
            key=hashlib.sha256(name.encode()).hexdigest()[:10]
            destination=output/f'{path.stem}_{key}_predictions.csv'
            result.to_csv(destination,index=False)
            report.append({'source':name,'output':str(destination),'rows':len(result),'critical':int(result['Risk Segment'].eq('Critical').sum()),'high':int(result['Risk Segment'].eq('High').sum())})
            print(f'Scored {name}: {len(result)} accounts -> {destination}')
        except Exception as e:
            failures.append({'source':name,'error':str(e)})
            print(f'FAILED {name}: {e}',file=sys.stderr)
    (output/'run_summary.json').write_text(json.dumps({'processed':report,'failed':failures},indent=2),encoding='utf-8')
    if failures:
        raise SystemExit(f'{len(failures)} file(s) failed validation/scoring')

if __name__=='__main__':
    main()
