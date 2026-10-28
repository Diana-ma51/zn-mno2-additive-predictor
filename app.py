"""Research interface for the verified 201-record, nine-feature model."""
from pathlib import Path
import io
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
FEATURES = ['1.分子量','2.溶解度','3.添加剂含量','4.Zn吸附能','5.MnO2吸附能','Chi1','NumValenceElectrons','NumHeavyAtoms','Chi0']
LABELS = ['Molecular weight (g/mol)','Solubility (g/L)','Additive content (wt.%)','Zn adsorption energy (eV)','MnO₂ adsorption energy (eV)','Chi1','NumValenceElectrons','NumHeavyAtoms','Chi0']
DEFAULTS = [200.,100.,1.,-1.,-1.,5.,50,20,10.]
st.set_page_config(page_title='Zn–MnO₂ | Additive screening',page_icon='🔬',layout='wide')
st.markdown('''<style>
.stApp{background:#f6f8fa;color:#243440}
.block-container{max-width:1160px;padding-top:2.5rem;padding-bottom:3rem}
h1{font-size:2rem!important;font-weight:650!important;letter-spacing:-.04em}
h2,h3{font-weight:600!important;letter-spacing:-.02em}
[data-testid="stMetric"]{background:white;border:1px solid #dfe6eb;border-radius:8px;padding:16px}
[data-testid="stMetricValue"]{font-size:1.75rem;color:#217fb8}
[data-testid="stForm"]{background:white;border-color:#dfe6eb;border-radius:10px}
div.stButton>button[kind="primary"],div.stFormSubmitButton>button[kind="primary"]{background:#217fb8;border:0}
[data-baseweb="tab"]{font-weight:500}
.eyebrow{color:#217fb8;font-size:12px;font-weight:650;letter-spacing:.16em;margin-bottom:6px}
.subtitle{color:#667681;font-size:15px;max-width:820px;margin-bottom:24px}
.research-nav{display:flex;justify-content:space-between;align-items:center;padding:4px 0 24px;font-size:12px;letter-spacing:.1em;color:#617283}.research-nav strong{color:#142d43;font-size:15px}.research-nav a{color:#617283;text-decoration:none;margin-left:24px}
.hero{background:radial-gradient(ellipse at 85% 30%,#16475b 0,transparent 55%),#0c2135;border-radius:16px;padding:54px 48px;color:#fff;display:grid;grid-template-columns:1.2fr 1fr;gap:48px;align-items:center;margin-bottom:30px;overflow:hidden}
.hero .kicker{color:#7dd3dd;font-size:11px;letter-spacing:.18em;font-weight:600}.hero h1{color:white!important;font-size:46px!important;line-height:1.12!important;margin:16px 0 20px!important;letter-spacing:-.045em!important}.hero p{color:#bdcdda;font-size:15px;line-height:1.8;max-width:480px}.hero a.cta{display:inline-block;background:#77d4df;color:#102b40;text-decoration:none;border-radius:5px;padding:12px 20px;margin-top:18px;font-size:13px;font-weight:600}.hero .secondary{color:#c8dce8;font-size:13px;text-decoration:none;margin-left:20px}
.cell{border:1px solid #466377;border-radius:12px;padding:26px 22px;background:#ffffff06}.cell-title{font-size:11px;letter-spacing:.14em;color:#b9d2de;margin-bottom:25px}.cell-stack{display:grid;grid-template-columns:1fr 1.6fr 1fr;gap:8px;height:190px}.electrode{display:flex;flex-direction:column;justify-content:center;align-items:center;border-radius:6px;font-size:24px;font-weight:600}.electrode small{font-size:10px;letter-spacing:.08em;margin-top:12px;font-weight:400}.zn{background:linear-gradient(120deg,#bdccd7,#667f91);color:#122a3d}.mn{background:linear-gradient(120deg,#55b5ce,#216986);color:white}.electrolyte{position:relative;border-block:1px solid #4c7d94;display:flex;flex-direction:column;justify-content:center;align-items:center;color:#90deea;font-size:12px;gap:17px;background:radial-gradient(circle,#77d4df55 1.5px,transparent 2px) 0 0/24px 24px}.ion{background:#173b52;border:1px solid #75becb;border-radius:30px;padding:5px 10px;font-size:12px}.cell-foot{color:#91adbd;font-size:10px;line-height:1.7;margin-top:19px}.overview{display:grid;grid-template-columns:repeat(3,1fr);gap:26px;margin:12px 0 34px}.overview article{border-top:1px solid #c9d6df;padding-top:18px}.overview .num{color:#278da8;font-size:11px;letter-spacing:.12em}.overview h3{font-size:17px!important;margin:10px 0!important}.overview p{font-size:13px;line-height:1.8;color:#617283}.section-intro{margin:30px 0 20px}.section-intro h2{font-size:25px!important;margin-bottom:5px!important}.section-intro p{font-size:14px;color:#617283}
@media(max-width:760px){.hero{grid-template-columns:1fr;padding:28px 24px;gap:28px}.hero h1{font-size:34px!important}.overview{grid-template-columns:1fr;gap:12px}.research-nav span{display:none}.block-container{padding-inline:18px}.cell-stack{height:150px}}
/* Airy editorial treatment: fine rules, translucent layers, restrained colour. */
.stApp{background:radial-gradient(ellipse at 85% 5%,#eaf6fa 0,transparent 42%),#fcfdfd;color:#243d4c}
.block-container{max-width:1220px;padding-top:2rem}
.research-nav{border-bottom:1px solid #dce7eb;padding-bottom:22px;margin-bottom:30px;font-size:10px;letter-spacing:.13em}
.research-nav strong{font-size:12px;font-weight:550;letter-spacing:.12em;color:#315362}
.hero{position:relative;background:linear-gradient(115deg,#ffffffb8,#eef8fa80);border:1px solid #dbe9ee;border-radius:3px;padding:64px 48px;gap:56px;color:#243d4c;box-shadow:0 20px 60px #254b6005}
.hero::before{content:'';position:absolute;left:0;top:0;width:70px;height:2px;background:#74a8ba}
.hero .kicker{color:#487b8b;font-size:10px;letter-spacing:.2em;font-weight:500}
.hero h1{color:#244354!important;font-size:45px!important;font-weight:400!important;line-height:1.16!important;letter-spacing:-.045em!important}
.hero p{color:#617c89;font-size:14px;line-height:1.9}
.hero a.cta{background:#ffffff80;border:1px solid #84aab9;color:#294f60;border-radius:2px;font-weight:500;padding:12px 18px;transition:background .2s,border-color .2s}
.hero a.cta:hover{background:#e5f2f7;border-color:#3b7c97}
.hero .secondary{color:#6a8794;font-size:12px}
.cell{position:relative;border:1px solid #cbdfe7;border-radius:3px;padding:30px 24px;background:linear-gradient(135deg,#ffffffb8,#ffffff35);box-shadow:12px 14px 0 -1px #f4fafb,12px 14px 0 0 #deebef}
.cell-title{color:#4d788b;font-size:10px;letter-spacing:.18em;border-bottom:1px solid #dce9ee;padding-bottom:16px;margin-bottom:24px}
.cell-stack{gap:12px;height:190px;padding:8px 0}
.electrode{border-radius:2px;font-weight:400;font-size:28px;position:relative}
.electrode::after{content:'';position:absolute;inset:5px;border:1px solid #ffffffc0;pointer-events:none}
.electrode small{font-size:9px;letter-spacing:.16em}
.zn{background:linear-gradient(110deg,#ffffff90,#deeaf080);border:1px solid #9fb8c6;color:#496777}
.mn{background:linear-gradient(110deg,#f4fcfe90,#ccebf180);border:1px solid #87b7c7;color:#347287}
.electrolyte{border-block:1px solid #c5dde5;background:radial-gradient(circle,#88b5c555 .7px,transparent 1px) 0 0/19px 19px;color:#5c8596;font-size:11px;gap:20px;text-align:center}
.ion{background:#ffffffa0;border:1px solid #a5c8d4;border-radius:30px;padding:4px 10px;color:#4d8093;font-size:11px;font-weight:400}
.cell-foot{color:#6c8997;font-size:10px;line-height:1.85;border-top:1px solid #dce9ee;padding-top:16px}
.section-intro{margin-top:44px}.section-intro h2{color:#2e4a59;font-size:25px!important;font-weight:450!important}.eyebrow{color:#638d9d;font-size:10px;font-weight:500;letter-spacing:.18em}
.overview{gap:34px;margin-bottom:42px}.overview article{border-color:#d5e5eb}.overview h3{color:#365362;font-weight:500!important;font-size:16px!important}.overview p{font-size:12px;color:#6a808b}.overview .num{font-size:10px;color:#7296a4}
[data-testid="stMetric"]{background:#ffffff70;border:0;border-top:1px solid #cbdfe7;border-radius:0;padding:16px 8px}
[data-testid="stMetricValue"]{font-size:1.8rem;color:#3b6c80;font-weight:400}
[data-testid="stForm"]{background:#ffffffa8;border:1px solid #dce8ed;border-radius:3px;box-shadow:0 12px 30px #30536903}
[data-baseweb="input"]{border-radius:3px;background:#f8fbfc;border-color:#dbe7ec}
div.stFormSubmitButton>button[kind="primary"]{background:#376b80;border:1px solid #376b80;border-radius:3px;font-weight:500}
[data-baseweb="tab-list"]{border-bottom:1px solid #deeaee;gap:28px}[data-baseweb="tab"]{font-weight:450;color:#607f8e}
@media(max-width:760px){.hero{padding:30px 23px;gap:32px}.hero h1{font-size:34px!important}.cell{margin-right:12px}.research-nav strong{font-size:10px}.cell-stack{height:160px}.hero .secondary{display:inline-block;margin:18px 0 0 15px}}
.stApp{background:radial-gradient(ellipse at 92% 8%,#c7e5f5 0,transparent 45%),linear-gradient(150deg,#f0f7fc,#e4eff8 65%,#f4f9fd);color:#203e5b}
.hero{background:radial-gradient(ellipse at 82% 28%,#8ec8e866,transparent 58%),linear-gradient(115deg,#e3f0fa 0%,#c1ddf0 60%,#a6cce6 100%);border:1px solid #9ebed5;padding:52px 42px;gap:20px;grid-template-columns:1fr 1.15fr;box-shadow:0 18px 55px #285a8710;isolation:isolate}
.hero::after{content:'';position:absolute;width:160px;height:160px;right:18%;top:-100px;border-radius:50%;border:1px solid #ffffffb8;background:radial-gradient(circle at 30% 25%,#ffffffb0,#76b8e012 50%,#397bb333);box-shadow:inset -6px -9px 22px #4f94c233;z-index:-1}
.hero::before{width:90px;background:#387eaf}.hero h1{color:#173f65!important;font-weight:450!important;font-size:43px!important}.hero p{color:#3c6280}.hero .kicker{color:#2d709c}.hero a.cta{background:#266390;color:white;border-color:#266390;box-shadow:0 5px 18px #25628d20}.hero a.cta:hover{background:#174f7b;color:white}.hero .secondary{color:#365f80}
.battery-art{position:relative;min-width:0}.battery-art img{width:100%;display:block;filter:drop-shadow(0 12px 15px #245d8818)}.art-caption{border-top:1px solid #6f9dbd66;padding:15px 8px 0;font-size:9px;letter-spacing:.15em;color:#335e80;display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px}.art-caption span{letter-spacing:0;color:#567894;font-size:10px}
.research-nav{border-color:#a9c5d9}.research-nav strong{color:#244f74}.section-intro h2{color:#234d70}.overview article{border-color:#9ebed5}.overview .num{color:#3d7ba7}.overview p{color:#496a83}
[data-testid="stMetric"]{background:#f7fbff99;border-top:2px solid #72a5ca;padding:18px 14px;box-shadow:0 6px 18px #2f608305}[data-testid="stMetricValue"]{color:#285f8e}
[data-testid="stForm"]{background:#f9fcffce;border-color:#b9cfdf;box-shadow:0 12px 32px #315f8510}[data-baseweb="input"]{background:#edf5fb;border-color:#b5ccde}div.stFormSubmitButton>button[kind="primary"]{background:#28638f;border-color:#28638f}
@media(max-width:760px){.hero{grid-template-columns:1fr;padding:30px 22px;gap:22px}.hero h1{font-size:34px!important}.battery-art{max-width:540px;margin:auto}.art-caption{font-size:8px}}
/* Higher-contrast blue technology finish. */
.stApp{background:radial-gradient(ellipse at 95% 0%,#b9d9ee 0,transparent 45%),#edf3f9;color:#183650}
.research-nav{border-bottom:1px solid #87aac7}.research-nav strong{color:#113d68;font-weight:650}.research-nav a{color:#315d82}
.hero{background:linear-gradient(110deg,#102e50 0%,#174773 48%,#83b8dd 100%);border:1px solid #527fa7;border-radius:5px;box-shadow:0 16px 44px #133b6225;position:relative}
.hero::before{height:3px;width:110px;background:#66ddff;box-shadow:0 0 15px #50c8ff80}
.hero::after{width:260px;height:260px;right:-80px;top:-150px;border-color:#b2eeff80;background:radial-gradient(circle at 28% 25%,#ffffff70,#7bd7fa08 50%,#3995c730);box-shadow:inset -10px -14px 30px #215b8a33}
.hero h1{color:#ffffff!important;font-weight:550!important;text-shadow:0 2px 15px #12324f22}.hero p{color:#d3e6f5}.hero .kicker{color:#86ddff;font-weight:600;letter-spacing:.2em}
.hero a.cta{background:#8de2ff;color:#103752;border:1px solid #b5eeff;box-shadow:0 4px 22px #47c7ff25;font-weight:650}.hero a.cta:hover{background:#c0f0ff;color:#0a2947;border-color:white}.hero .secondary{color:#d5ecfb}
.battery-art{background:linear-gradient(135deg,#d4efffdb,#add7f0ad);border:1px solid #bcE9ffb0;border-radius:4px;padding:12px 10px 15px;box-shadow:0 0 0 5px #a5deff0b,0 18px 38px #082b4f20,inset 0 1px 0 #ffffffb0}
.battery-art img{filter:drop-shadow(0 12px 12px #123d7128) contrast(1.12) saturate(1.12)}.art-caption{border-color:#467da56b;color:#174c77;font-weight:550}.art-caption span{color:#365f7f;font-weight:400}
.section-intro h2{color:#173e64;font-weight:550!important}.eyebrow{color:#216a9c;font-weight:650}.section-intro p{color:#496782}
.overview article{border-top:2px solid #73a3c7;padding:18px 15px 8px;background:linear-gradient(180deg,#ffffffa3,#ffffff00)}.overview h3{color:#1d4b73;font-weight:600!important}.overview .num{color:#247bab;font-weight:600}.overview p{color:#47637a}
[data-testid="stMetric"]{background:#173f66;border-top:2px solid #75d5ff;border-radius:3px;box-shadow:0 8px 18px #214c7415}
[data-testid="stMetricValue"]{color:#f2fbff;font-weight:500}[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p{color:#bcdbef!important}
[data-testid="stForm"]{border:1px solid #91b1cd;box-shadow:0 12px 30px #1c4f7910;background:#fbfdff}
[data-baseweb="input"]{background:#edf4fb;border-color:#a7bfd3;color:#163d60}
div.stFormSubmitButton>button[kind="primary"]{background:#185b8c;border-color:#185b8c;box-shadow:0 4px 12px #1d659724}
[data-baseweb="tab-list"]{border-color:#a5bfd4}[data-baseweb="tab"]{color:#345e80;font-weight:550}[data-baseweb="tab"][aria-selected="true"]{color:#0e639b!important}[data-baseweb="tab-highlight"]{background:#268fc8}
@media(max-width:760px){.hero{background:linear-gradient(150deg,#102e50,#225985 65%,#80b4d8)}.battery-art{padding:8px}.art-caption{padding-inline:4px}}
.hero h1{font-size:clamp(28px,3.25vw,42px)!important}.hero>div{min-width:0}.battery-art img{height:auto}.hero a:focus-visible,.research-nav a:focus-visible{outline:2px solid #66dfff;outline-offset:5px}
@media(max-width:640px){.hero h1{font-size:30px!important;line-height:1.2!important}.hero{padding:26px 18px}.hero .secondary{margin-left:0;display:block}.research-nav strong{letter-spacing:.06em}.art-caption{line-height:1.6}[data-baseweb="tab-list"]{gap:16px;overflow-x:auto}[data-baseweb="tab"]{white-space:nowrap}}
</style>''',unsafe_allow_html=True)
battery_uri = 'data:image/svg+xml;base64,' + base64.b64encode((ROOT/'battery_scene.svg').read_bytes()).decode()
st.markdown('''<nav class="research-nav"><strong>Zn–MnO₂ / MATERIALS INTELLIGENCE</strong><span><a href="#battery-system">Battery system</a><a href="#screening">Predict retention</a></span></nav>
<section class="hero"><div><div class="kicker">AQUEOUS BATTERIES · DATA-GUIDED DISCOVERY</div><h1>Electrolyte additive<br>screening for aqueous<br>Zn–MnO₂ batteries.</h1><p>Explore electrolyte additives for aqueous zinc–manganese dioxide batteries. Connect molecular descriptors and adsorption energies with predicted capacity retention to guide experimental screening.</p><a class="cta" href="#screening">Explore the predictor ↗</a><a class="secondary" href="#battery-system">The battery system ↓</a></div><div class="battery-art"><img src="BATTERY_SCENE_DATA" alt="Conceptual three-dimensional Zn anode, separator and MnO2 cathode in aqueous electrolyte"><div class="art-caption">Zn / AQUEOUS ELECTROLYTE / MnO₂<span>Conceptual illustration · not to scale</span></div></div></section>
<div id="battery-system" class="section-intro"><div class="eyebrow">01 / BATTERY CHEMISTRY</div><h2>Three components. One coupled system.</h2><p>A zinc metal anode, a manganese dioxide cathode and a water-based electrolyte.</p></div>
<section class="overview"><article><div class="num">01 · ZINC METAL</div><h3>The negative electrode</h3><p>Zinc undergoes dissolution and deposition during cycling. Electrolyte composition influences the electrode interface and the conditions under which these processes occur.</p></article><article><div class="num">02 · MANGANESE DIOXIDE</div><h3>The positive electrode</h3><p>MnO₂ participates in charge-storage reactions. Its electrochemical response depends on the material structure, electrolyte and cycling conditions; a single mechanism is not assumed here.</p></article><article><div class="num">03 · AQUEOUS ELECTROLYTE</div><h3>The additive environment</h3><p>The water-based electrolyte connects the two electrodes. This tool examines additive properties and adsorption descriptors as predictors of capacity retention, helping prioritise candidates for testing.</p></article></section>
<div id="screening" class="section-intro"><div class="eyebrow">02 / COMPUTATIONAL SCREENING</div><h2>From descriptors to a testable prediction.</h2><p>Enter nine features, estimate retention, and inspect how the model uses material descriptors.</p></div>'''.replace('BATTERY_SCENE_DATA',battery_uri),unsafe_allow_html=True)
cols=st.columns(4)
for col,label,value in zip(cols,['Model records','Input descriptors','Held-out R²','Held-out RMSE'],['201','9','0.801','8.78 pp']):col.metric(label,value)

@st.cache_resource
def load_model(stamp):
    model=joblib.load(ROOT/'xgb_model_201_9features.joblib')
    names=list(joblib.load(ROOT/'feature_names_201_9features.pkl'))
    if names!=FEATURES or model.n_features_in_!=9:raise ValueError('Model feature order does not match the nine inputs.')
    return model

try:
    path=ROOT/'xgb_model_201_9features.joblib'
    model=load_model((path.stat().st_mtime_ns,(ROOT/'feature_names_201_9features.pkl').stat().st_mtime_ns))
except Exception as exc:
    st.error(f'Model unavailable: {exc}')
    st.stop()

def validate(values):
    x=np.asarray(values,dtype=float).reshape(1,-1)
    if x.shape!=(1,9) or not np.isfinite(x).all():raise ValueError('Enter nine finite values.')
    if x[0,0]<=0:raise ValueError('Molecular weight must be greater than zero.')
    if x[0,1]<0 or not 0<=x[0,2]<=100:raise ValueError('Check solubility and additive content.')
    if x[0,5]<0 or x[0,8]<0:raise ValueError('Connectivity indices must be non-negative.')
    if any(x[0,i]<1 or x[0,i]!=int(x[0,i]) for i in [6,7]):raise ValueError('Atom and electron counts must be positive integers.')
    return x

prediction,explanation,methods=st.tabs(['Predict','Model interpretation','Methods & performance'])
with prediction:
    left,right=st.columns([1.6,1],gap='large')
    with left:
        st.subheader('Material descriptors')
        st.caption('Use the same descriptor definitions and adsorption-energy convention as the training data.')
        with st.form('inputs'):
            a,b=st.columns(2)
            values=[]
            for i,(label,default) in enumerate(zip(LABELS,DEFAULTS)):
                with a if i<5 else b:
                    options={'value':default,'key':f'feature_{i}'}
                    if i in [6,7]:options.update(min_value=1,step=1)
                    elif i not in [3,4]:options.update(min_value=0.,step=.1,format='%.4f')
                    else:options.update(step=.1,format='%.4f')
                    if i==2:options['max_value']=100.
                    values.append(st.number_input(label,**options))
            submitted=st.form_submit_button('Predict retention',type='primary',width='stretch')
        if submitted:
            st.session_state.pop('result',None)
            try:
                x=validate(values)
                value=float(model.predict(x)[0])
                if not np.isfinite(value):raise ValueError('The model returned a non-finite result.')
                st.session_state.result={'value':value,'inputs':list(values)}
            except Exception as exc:st.error(str(exc))
    with right:
        st.subheader('Prediction')
        result=st.session_state.get('result')
        if result:
            st.metric('Predicted capacity retention',f'{result["value"]:.2f}%')
            st.caption('Result for the last submitted inputs. Submit again after editing parameters.')
            frame=pd.DataFrame([result['inputs']],columns=LABELS)
            frame['Predicted retention (%)']=result['value']
            st.download_button('Download result · CSV',frame.to_csv(index=False).encode('utf-8-sig'),'retention_prediction.csv','text/csv',width='stretch')
            with st.expander('Submitted inputs'):st.dataframe(frame.T.rename(columns={0:'Value'}),width='stretch')
        else:st.info('Enter the material descriptors and select Predict retention.')
        st.caption('A model estimate for screening; experimental validation is required. Test RMSE describes aggregate error and is not a sample-specific confidence interval.')

with explanation:
    st.subheader('Feature contributions')
    st.caption('Mean absolute SHAP contributions: MnO₂ adsorption energy 7.970 pp · Zn adsorption energy 6.854 pp · molecular weight 3.130 pp.')
    st.image(str(ROOT/'SHAP_Importance_201_9features.png'),width='stretch')
    st.caption('Blue → white → red denotes low → high feature values. Ranking-plot colours use the original 5th–95th percentile scale. SHAP describes model associations, not causality.')
    for ext in ['svg','pdf']:
        f=ROOT/f'SHAP_Importance_201_9features.{ext}'
        if f.exists():st.download_button(f'Download SHAP · {ext.upper()}',f.read_bytes(),f.name,'image/svg+xml' if ext=='svg' else 'application/pdf')

with methods:
    st.subheader('Evaluation summary')
    st.dataframe(pd.DataFrame({'Evaluation':['Training','Held-out test','5-fold CV (mean ± SD)','14-additive evaluation'],'Records':['160','41','201','14'],'R²':['0.949','0.801','0.729 ± 0.124','0.496'],'RMSE (pp)':['4.37','8.78','—','6.99'],'MAE (pp)':['3.52','7.34','—','5.31']}),hide_index=True,width='stretch')
    st.caption('Model version: 201 records / 9 features · verified against the corresponding analysis workbook. pp = percentage points.')
st.divider()
st.caption('Zn–MnO₂ additive screening · Research interface · XGBoost + SHAP')
