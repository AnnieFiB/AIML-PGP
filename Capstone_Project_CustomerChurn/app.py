import io,json,re
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# App theme and branding
ROOT=Path(__file__).resolve().parent
LOGO=ROOT/'logo.png'
st.set_page_config(page_title='ChurnGuard',page_icon=str(LOGO) if LOGO.exists() else '📊',layout='wide')
st.markdown('''<style>.stApp{background:#101010;color:#e5e5e5}[data-testid="stSidebar"]{background:#1b1b1b}[data-testid="stMetric"]{background:#242424;padding:15px;border-radius:10px}.stButton>button,.stDownloadButton>button{background:#383838;color:white;border:1px solid #666}</style>''',unsafe_allow_html=True)

# Training-data bounds and categorical vocabulary
BOUNDS={'tenure':(0,99),'city_tier':(1,3),'cc_contacted_ly':(4,129),'service_score':(0,5),'account_user_count':(1,6),'cc_agent_score':(1,5),'rev_per_month':(1,140),'complain_ly':(0,1),'rev_growth_yoy':(4,28),'coupon_used_for_payment':(0,16),'day_since_cc_connect':(0,47),'cashback':(0,1991)}
CATEGORIES={'payment':['cash on delivery','credit card','debit card','e wallet','upi'],'gender':['female','male'],'account_segment':['hni','regular','regular plus','super','super plus'],'marital_status':['divorced','married','single'],'login_device':['computer','mobile','others']}
ACTIONS={'Critical':'Immediate personalized outreach, complaint resolution and tailored incentives','High':'Proactive engagement, targeted retention offers and loyalty incentives','Moderate':'Preventive engagement and service improvement','Low':'Routine monitoring and customer engagement'}

@st.cache_resource
def load_assets():
    """Load the fitted classification pipeline and training feature schema."""
    schema=json.loads((ROOT/'models/feature_schema.json').read_text())
    return joblib.load(ROOT/'models'/schema['model_file']),schema

try: model,schema=load_assets()
except Exception as e:
    st.error(f'Unable to load model/schema: {e}')
    st.stop()
features=schema['features']

def normalize(data):
    """Standardize column headings and reject ambiguous duplicates."""
    data=data.copy()
    data.columns=[re.sub(r'[\s-]+','_',str(c).strip().lower()) for c in data.columns]
    if data.columns.duplicated().any(): raise ValueError('Duplicate column names after normalization')
    return data

def score(data):
    """Validate inputs, score churn and return only account-level results."""
    data=normalize(data)
    missing=[c for c in features if c not in data.columns]
    if missing: raise ValueError('Missing columns: '+', '.join(missing)+' | Detected: '+', '.join(data.columns))
    if data.empty: raise ValueError('The uploaded file has no customer rows')
    x=data[features].copy()
    for c in schema['numeric']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
        if c in BOUNDS:
            lo,hi=BOUNDS[c]
            if x[c].notna().any() and (~x[c].dropna().between(lo,hi)).any(): raise ValueError(f'{c} must be between {lo} and {hi}')
    for c in schema['categories']:
        x[c]=x[c].astype('string').str.strip().str.lower().astype(object)
        if c in CATEGORIES:
            bad=x[c].dropna()[~x[c].dropna().isin(CATEGORIES[c])].unique().tolist()
            if bad: raise ValueError(f'{c} has unsupported values: {bad}')
    p=model.predict_proba(x)[:,1]
    risk=pd.Series(pd.cut(p,[-.001,.30,.60,.80,1],labels=['Low','Moderate','High','Critical']).astype(str))
    # Include the six most important predictors for interpretation and prioritization
    main=['account_segment','payment','tenure','marital_status','complain_ly','login_device']
    result=pd.DataFrame({'accountid':data['accountid'].to_numpy() if 'accountid' in data else ['N/A']*len(data)})
    for c in main: result[c]=x[c].to_numpy()
    result['Churn Probability']=p
    result['Risk Segment']=risk
    if 'rev_per_month' in data: result['Revenue at Risk']=pd.to_numeric(data['rev_per_month'],errors='coerce').to_numpy()*p
    result['Recommended Action']=risk.map(ACTIONS)
    return result.sort_values('Churn Probability',ascending=False).reset_index(drop=True)

def show_results(result,filename):
    """Display scored accounts and offer CSV and Excel exports."""
    st.dataframe(result,hide_index=True,width='stretch')
    a,b=st.columns(2)
    a.download_button('Download CSV',result.to_csv(index=False).encode(),filename+'.csv','text/csv')
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer: result.to_excel(writer,index=False,sheet_name='Predictions')
    b.download_button('Download Excel',buffer.getvalue(),filename+'.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Logo and navigation
if LOGO.exists(): st.sidebar.image(str(LOGO),width=125)
page=st.sidebar.radio('Navigation',['Single Prediction','Batch Prediction','Model Information'])
header,title=st.columns([1,6],vertical_alignment='center')
with header:
    if LOGO.exists(): st.image(str(LOGO),width=105)
with title:
    st.title('ChurnGuard')
    st.caption('Customer Churn Prediction and Retention Strategy')

if page=='Single Prediction':
    st.subheader('Single Account Prediction')
    with st.form('single'):
        accountid=st.text_input('Account ID')
        values={}
        with st.expander('Numerical Features',expanded=True):
            cols=st.columns(3)
            for i,f in enumerate(schema['numeric']):
                with cols[i%3]:
                    default=float(schema['defaults'].get(f,0) or 0)
                    if f in BOUNDS:
                        lo,hi=BOUNDS[f]
                        values[f]=st.number_input(f,min_value=float(lo),max_value=float(hi),value=max(float(lo),min(default,float(hi))))
                    else: values[f]=st.number_input(f,value=default)
        with st.expander('Categorical Features',expanded=True):
            cols=st.columns(3)
            for i,f in enumerate(schema['categories']):
                with cols[i%3]: values[f]=st.selectbox(f,CATEGORIES.get(f,schema['categories'][f]) or [''])
        submitted=st.form_submit_button('Predict Churn',type='primary')
    if submitted:
        try:
            result=score(pd.DataFrame([{**values,'accountid':accountid}]))
            a,b=st.columns(2)
            a.metric('Churn Probability',f"{result['Churn Probability'].iloc[0]:.1%}")
            b.metric('Risk Segment',result['Risk Segment'].iloc[0])
            st.info(result['Recommended Action'].iloc[0])
            show_results(result,'single_prediction')
        except Exception as e: st.error(str(e))
elif page=='Batch Prediction':
    st.subheader('Batch Account Prediction')
    upload=st.file_uploader('Upload CSV or XLSX',type=['csv','xlsx'])
    st.caption('Required model features: '+', '.join(features)+'. Include accountid for identification.')
    if upload:
        try:
            data=pd.read_csv(upload,dtype={'accountid':'string'}) if upload.name.lower().endswith('.csv') else pd.read_excel(upload,dtype={'accountid':'string'})
            st.caption('Detected columns: '+', '.join(map(str,data.columns)))
            result=score(data)
            a,b,c=st.columns(3)
            a.metric('Accounts',len(result))
            b.metric('Critical Risk',int((result['Risk Segment']=='Critical').sum()))
            c.metric('High Risk',int((result['Risk Segment']=='High').sum()))
            show_results(result,'batch_predictions')
        except Exception as e: st.error(str(e))
else:
    st.subheader('Model Information')
    name=schema['model_file'].removeprefix('churnguard_').removesuffix('.joblib').replace('_',' ').title()
    st.write('**Selected Model:**',name)
    st.write('**Model Type:** Supervised binary classification')
    st.write('**Target:** Churn')
    st.write('**Predictors:**',len(features))
    st.write('**Preprocessing:** Included in saved pipeline')
    st.table(pd.DataFrame({'Risk Segment':['Low','Moderate','High','Critical'],'Churn Probability':['0–30%','>30–60%','>60–80%','>80–100%']}))
    st.caption('Previously reported evaluation: CV PR-AUC 0.9481 | Test PR-AUC 0.96 | Accuracy 96%')
st.caption('Predictions are estimates; campaign ROI requires pilot validation.')
