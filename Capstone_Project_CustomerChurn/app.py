import io,json,re,sqlite3,hashlib,smtplib,ssl
from email.message import EmailMessage
from pathlib import Path
import joblib,pandas as pd,streamlit as st

# App settings and branding
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='ChurnGuard',page_icon='🛡️',layout='wide')
st.sidebar.title('🛡️ ChurnGuard')
st.sidebar.caption('Customer Churn Prediction')

# Valid training ranges and categories
BOUNDS={'tenure':(0,99),'city_tier':(1,3),'cc_contacted_ly':(4,132),'service_score':(0,5),'account_user_count':(1,6),'cc_agent_score':(1,5),'rev_per_month':(1,140),'complain_ly':(0,1),'rev_growth_yoy':(4,28),'coupon_used_for_payment':(0,16),'day_since_cc_connect':(0,47),'cashback':(0,1997)}
CATEGORIES={'payment':['cash on delivery','credit card','debit card','e wallet','upi'],'gender':['female','male'],'account_segment':['hni','regular','regular plus','super','super plus'],'marital_status':['divorced','married','single'],'login_device':['computer','mobile','others']}
ACTIONS={'Critical':'Immediate personalized outreach, complaint resolution and tailored incentives','High':'Proactive engagement, targeted retention offers and loyalty incentives','Moderate':'Preventive engagement and service improvement','Low':'Routine monitoring and customer engagement'}

# Load model and feature schema once
@st.cache_resource
def load_assets():
    schema=json.loads((ROOT/'models/feature_schema.json').read_text(encoding='utf-8'))
    return joblib.load(ROOT/'models'/schema['model_file']),schema

try: model,schema=load_assets()
except Exception as e:
    st.error(f'Unable to load model/schema: {e}')
    st.stop()
features=schema['features']

# Normalize incoming column names
def normalize(df):
    df=df.copy()
    df.columns=[re.sub(r'[\s-]+','_',str(c).strip().lower()) for c in df.columns]
    if df.columns.duplicated().any(): raise ValueError('Duplicate column names after normalization.')
    return df

# Read real CSV or XLSX content, even with a misleading extension
def read_table(upload):
    raw=upload.getvalue()
    if not raw: raise ValueError('The uploaded file is empty.')
    if raw.startswith(b'PK\x03\x04'):
        return pd.read_excel(io.BytesIO(raw),engine='openpyxl',dtype={'accountid':'string'})
    if raw.startswith(b'\xd0\xcf\x11\xe0'):
        raise ValueError('Old Excel (.xls) format detected. Save as .xlsx and upload again.')
    for encoding in ('utf-8-sig','utf-16','cp1252','latin-1'):
        try:
            df=pd.read_csv(io.StringIO(raw.decode(encoding)),sep=None,engine='python',dtype={'accountid':'string'})
            if len(df.columns)>1: return df
        except (UnicodeError,pd.errors.ParserError,ValueError): pass
    raise ValueError('Cannot read the file. Upload a customer CSV or XLSX with the required features.')

# Validate features, calculate probabilities and assign retention actions
def score(df):
    df=normalize(df)
    if df.empty: raise ValueError('No customer records found.')
    # Account ID is required for reliable tracking and email deduplication
    if 'accountid' not in df: raise ValueError('Missing accountid column. Every customer needs an Account ID.')
    df['accountid']=df['accountid'].astype('string').str.strip()
    if df['accountid'].isna().any() or df['accountid'].eq('').any(): raise ValueError('Account ID cannot be blank.')
    if df['accountid'].duplicated().any(): raise ValueError('Duplicate account IDs found. Include one row per customer.')
    missing=[f for f in features if f not in df]
    if missing: raise ValueError('Missing model features: '+', '.join(missing))
    x=df[features].copy()
    for f in schema['numeric']:
        x[f]=pd.to_numeric(x[f],errors='coerce')
        if f in BOUNDS:
            lo,hi=BOUNDS[f]
            if (~x[f].dropna().between(lo,hi)).any(): raise ValueError(f'{f} must be between {lo} and {hi}.')
    for f in schema['categories']:
        x[f]=x[f].astype('string').str.strip().str.lower().astype(object)
        if f in CATEGORIES:
            invalid=x[f].dropna()[~x[f].dropna().isin(CATEGORIES[f])].unique().tolist()
            if invalid: raise ValueError(f'{f} has unsupported values: {invalid}')
    probability=model.predict_proba(x)[:,1]
    risk=pd.cut(probability,[-.001,.30,.60,.80,1],labels=['Low','Moderate','High','Critical']).astype(str)
    result=pd.DataFrame({'accountid':df['accountid'].to_numpy()})
    for f in ['account_segment','payment','tenure','marital_status','complain_ly','login_device']:
        if f in x: result[f]=x[f].to_numpy()
    result['Churn Probability']=probability
    result['Risk Segment']=risk
    if 'rev_per_month' in x: result['Revenue at Risk']=x['rev_per_month'].to_numpy()*probability
    result['Recommended Action']=[ACTIONS[r] for r in risk]
    return result.sort_values('Churn Probability',ascending=False).reset_index(drop=True)

# Show predictions with downloadable CSV and XLSX
def show_results(result,name):
    st.dataframe(result,hide_index=True,width="stretch")
    csv_col,xlsx_col=st.columns(2)
    csv_col.download_button('Download CSV',result.to_csv(index=False).encode('utf-8-sig'),f'{name}.csv','text/csv')
    output=io.BytesIO()
    with pd.ExcelWriter(output,engine='openpyxl') as writer: result.to_excel(writer,index=False,sheet_name='Predictions')
    xlsx_col.download_button('Download Excel',output.getvalue(),f'{name}.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Email high-risk alerts; SQLite prevents duplicate messages across app reruns
DB=ROOT/'churnguard_alerts.sqlite3'
def send_alerts(result):
    alerts=result[result['Risk Segment'].isin(['Critical','High'])].copy()
    if alerts.empty: return 'No Critical or High-risk accounts to notify.'
    try:
        cfg=st.secrets['email']
        host,port,user,password=cfg['host'],int(cfg.get('port',587)),cfg['username'],cfg['password']
        sender=cfg.get('sender',user)
        recipients=cfg['recipients']
        recipients=[v.strip() for v in recipients.split(',')] if isinstance(recipients,str) else list(recipients)
        if not recipients: raise ValueError('No email recipients configured.')
    except Exception as e: raise ValueError('Configure [email] in .streamlit/secrets.toml (host, port, username, password, recipients).') from e
    with sqlite3.connect(DB,timeout=15) as con:
        con.execute('CREATE TABLE IF NOT EXISTS sent (account TEXT PRIMARY KEY, risk TEXT, sent_at TEXT DEFAULT CURRENT_TIMESTAMP)')
        # Alert once per account and risk level; a change in level generates a new alert
        pending=[]
        for _,r in alerts.iterrows():
            account=str(r['accountid']).strip()
            if not account or account.lower() in ('n/a','nan','none'): continue
            key=hashlib.sha256((account+'|'+str(r['Risk Segment'])).encode()).hexdigest()
            if not con.execute('SELECT 1 FROM sent WHERE account=?',(key,)).fetchone(): pending.append((key,r))
        if not pending: return 'No new alerts; previously notified accounts were skipped.'
        lines=['ChurnGuard early warning — new high-risk accounts','','Account ID | Risk | Churn Probability | Recommended Action']
        for _,r in pending:
            lines.append(f"{r['accountid']} | {r['Risk Segment']} | {r['Churn Probability']:.1%} | {r['Recommended Action']}")
        msg=EmailMessage()
        msg['Subject']=f'ChurnGuard: {len(pending)} new high-risk account(s)'
        msg['From']=sender
        msg['To']=', '.join(recipients)
        msg.set_content('\n'.join(lines))
        if port==465:
            with smtplib.SMTP_SSL(host,port,context=ssl.create_default_context(),timeout=30) as smtp:
                smtp.login(user,password); smtp.send_message(msg)
        else:
            with smtplib.SMTP(host,port,timeout=30) as smtp:
                smtp.starttls(context=ssl.create_default_context())
                smtp.login(user,password); smtp.send_message(msg)
        con.executemany('INSERT INTO sent(account,risk) VALUES (?,?)',[(key,str(r['Risk Segment'])) for key,r in pending])
        con.commit()
    return f'Email sent for {len(pending)} new Critical/High-risk account(s).'

# Automatic email after scoring; suppress repeat attempts on Streamlit reruns
# Persistent SQLite history prevents resending already notified account/risk pairs.
def auto_alert(result,run_id):
    if not result['Risk Segment'].isin(['Critical','High']).any(): return
    if st.session_state.get('last_alert_run')==run_id: return
    st.session_state['last_alert_run']=run_id
    try:
        message=send_alerts(result)
        if message.startswith('Email sent'):
            st.success('Email alerts: '+message)
        else:
            st.info(message)
    except Exception as e: st.warning(f'Automatic email alert failed: {e}')

# Sidebar navigation
page=st.sidebar.radio('Navigation',['Single Prediction','Batch Prediction','Model Information'])
st.title('🛡️ ChurnGuard')
st.caption('Customer Churn Prediction and Retention Strategy')

if page=='Single Prediction':
    st.subheader('Single Account Prediction')
    with st.form('single'):
        accountid=st.text_input('Account ID',help='Required: unique identifier for the customer')
        values={}
        with st.expander('Numerical Features',expanded=True):
            cols=st.columns(3)
            for i,f in enumerate(schema['numeric']):
                with cols[i%3]:
                    default=float(schema.get('defaults',{}).get(f,0) or 0)
                    if f in BOUNDS:
                        lo,hi=BOUNDS[f]
                        values[f]=st.number_input(f,min_value=float(lo),max_value=float(hi),value=max(float(lo),min(default,float(hi))))
                    else: values[f]=st.number_input(f,value=default)
        with st.expander('Categorical Features',expanded=True):
            cols=st.columns(3)
            for i,f in enumerate(schema['categories']):
                with cols[i%3]:
                    options=CATEGORIES.get(f) or schema['categories'].get(f) or ['']
                    values[f]=st.selectbox(f,options)
        submitted=st.form_submit_button('Predict Churn',type='primary')
    if submitted:
        try:
            result=score(pd.DataFrame([{**values,'accountid':accountid}]))
            a,b=st.columns(2)
            a.metric('Churn Probability',f"{result['Churn Probability'].iloc[0]:.1%}")
            b.metric('Risk Segment',result['Risk Segment'].iloc[0])
            st.info(result['Recommended Action'].iloc[0])
            show_results(result,'single_prediction')
            auto_alert(result,'single:'+hashlib.sha256(json.dumps({'accountid':accountid,**values},sort_keys=True,default=str).encode()).hexdigest())
        except Exception as e: st.error(str(e))
elif page=='Batch Prediction':
    st.subheader('Batch Account Prediction')
    upload=st.file_uploader('Upload customer CSV or XLSX',type=['csv','xlsx'])
    st.caption('Include a unique, nonblank accountid for each customer and all required features: '+', '.join(features))
    st.caption('Upload customer records, not a feature-importance report.')
    if upload:
        try:
            df=read_table(upload)
            result=score(df)
            a,b,c=st.columns(3)
            a.metric('Accounts',len(result))
            b.metric('Critical Risk',int(result['Risk Segment'].eq('Critical').sum()))
            c.metric('High Risk',int(result['Risk Segment'].eq('High').sum()))
            show_results(result,'batch_predictions')
            auto_alert(result,'batch:'+hashlib.sha256(upload.getvalue()).hexdigest())
        except Exception as e: st.error(str(e))
else:
    st.subheader('Model Information')
    name=Path(schema['model_file']).stem.removeprefix('churnguard_').replace('_',' ').title()
    st.write('**Selected Model:**',name)
    st.write('**Model Type:** Supervised binary classification')
    st.write('**Target:** Churn')
    st.write('**Predictors:**',len(features))
    st.write('**Preprocessing:** Included in saved pipeline')
    st.table(pd.DataFrame({'Risk Segment':['Low','Moderate','High','Critical'],'Churn Probability':['0–30%','>30–60%','>60–80%','>80–100%']}))
    st.caption('Previously reported evaluation: CV PR-AUC 0.9481 | Test PR-AUC 0.96 | Accuracy 96%')
st.caption('Predictions are estimates; campaign ROI requires pilot validation.')
