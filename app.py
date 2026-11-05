"""RetenIA - app de retencion y churn. Ejecutar: streamlit run app.py"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap
from explain_utils import lime_for_encoded

BASE = Path(__file__).parent
ART = BASE/'artifacts'
DATA = BASE/'WA_Fn-UseC_-Telco-Customer-Churn.csv'
st.set_page_config(page_title='RetenIA | Retencion explicable',page_icon='📶',layout='wide')
st.markdown('''<style>
.block-container{padding-top:1.35rem;max-width:1400px}
[data-testid="stMetric"]{background:#f4f8fd;padding:16px;border-radius:13px;border:1px solid #e2e9f1}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {color:#173352!important}
.small-note{color:#576a80;font-size:.89rem}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return joblib.load(ART/'model.joblib'), joblib.load(ART/'rf_reference.joblib')

@st.cache_data
def load_data():
    data=pd.read_csv(DATA)
    data['TotalCharges']=pd.to_numeric(data['TotalCharges'],errors='coerce')
    summary=json.loads((ART/'summary.json').read_text(encoding='utf-8'))
    return data,summary,pd.read_csv(ART/'model_comparison.csv'),pd.read_csv(ART/'search_comparison.csv'),pd.read_csv(ART/'pareto_front.csv'),pd.read_csv(ART/'ranking_demo.csv')

try:
    model,rf_model=load_models()
    df,report,baselines,searches,pareto,ranking=load_data()
except Exception as exc:
    st.error(f'Faltan artefactos del entrenamiento: {exc}. Ejecute RetenIA_Notebook_Completo.ipynb primero.')
    st.stop()

st.title('RetenIA  |  Retencion inteligente y explicable')
st.caption('Demostrador académico basado en Telco Customer Churn · Modelo entrenado fuera de Streamlit · Probabilidades, no causalidad')
tab1,tab2,tab3,tab4=st.tabs(['01 · Exploracion','02 · Modelos y Pareto','03 · Prediccion y XAI','04 · Negocio y ROI'])

with tab1:
    st.subheader('¿Dónde se observa el abandono de clientes?')
    f1,f2=st.columns(2)
    with f1:
        contract=st.multiselect('Contrato',sorted(df['Contract'].dropna().unique().tolist()),default=sorted(df['Contract'].dropna().unique().tolist()))
    with f2:
        service=st.multiselect('Servicio de internet',sorted(df['InternetService'].dropna().unique().tolist()),default=sorted(df['InternetService'].dropna().unique().tolist()))
    d=df[df['Contract'].isin(contract)&df['InternetService'].isin(service)]
    c1,c2,c3=st.columns(3)
    c1.metric('Clientes filtrados',f'{len(d):,}')
    c2.metric('Abandono observado',f"{(d['Churn'].eq('Yes').mean()*100 if len(d) else 0):.1f}%")
    c3.metric('Pago mensual promedio',f"${d['MonthlyCharges'].mean() if len(d) else 0:.2f}")
    left,right=st.columns(2)
    with left:
        rates=d.groupby('Contract',as_index=False).agg(tasa_abandono=('Churn',lambda s:100*s.eq('Yes').mean()),clientes=('Churn','size'))
        fig=px.bar(rates,x='Contract',y='tasa_abandono',color='Contract',hover_data=['clientes'],title='Tasa de abandono por tipo de contrato')
        fig.update_layout(yaxis_title='Abandono (%)',showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=px.histogram(d,x='tenure',color='Churn',barmode='overlay',nbins=32,opacity=.7,title='Antigüedad de clientes y abandono')
        fig.update_layout(xaxis_title='Antigüedad (meses)',yaxis_title='Clientes')
        st.plotly_chart(fig,use_container_width=True)
    fig=px.box(d,x='Churn',y='MonthlyCharges',color='Churn',points=False,title='Pagos mensuales por estado de abandono')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Los gráficos son descriptivos, no relaciones causales. Dataset público histórico; no son clientes reales de un comprador.')

with tab2:
    st.subheader('Evidencia técnica verificable')
    cols=st.columns(5)
    for col,(label,key) in zip(cols,[('Accuracy','accuracy'),('Precision','precision'),('Recall','recall'),('F1','f1'),('ROC-AUC','roc_auc')]):
        col.metric(label,f"{report['test'][key]:.3f}")
    st.caption('Métricas calculadas sobre test aislado (1.409 clientes), usando umbral 0,50. Selección mediante validación cruzada, antes de observar test.')
    cal=report['calibration']
    st.warning(f"Calibración pendiente: score medio en test {cal['mean_predicted_score_test']:.1%} frente a abandono observado {cal['observed_churn_rate_test']:.1%} (Brier={cal['brier_score_test']:.3f}). El ROC-AUC mide ranking, no calibración de probabilidades.")
    l,r=st.columns(2)
    with l:
        st.markdown('**Comparación de dos modelos (CV, train)**')
        st.dataframe(baselines,hide_index=True,use_container_width=True)
    with r:
        st.markdown('**Optimización (24 ensayos / 3 folds por estrategia)**')
        st.dataframe(searches[['estrategia','cv_f1','tiempo_s','configuraciones','folds']],hide_index=True,use_container_width=True)
    a,b=st.columns(2)
    with a:
        cm=np.array(report['test']['confusion_matrix'])
        fig=px.imshow(cm,text_auto=True,x=['Pred. permanece','Pred. abandona'],y=['Real permanece','Real abandona'],color_continuous_scale='Blues',title='Matriz de confusión · Test')
        st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.scatter(pareto,x='contact_rate',y='cv_f1',hover_data=['trial','C','threshold'],title='Frente de Pareto (NSGA-II, CV)')
        fig.update_layout(xaxis_title='Proporción de clientes contactados (minimizar)',yaxis_title='F1 de abandono (maximizar)')
        st.plotly_chart(fig,use_container_width=True)
    st.caption(f"Estrategia ganadora: {report['selected_strategy']}. El frente tiene {report['pareto_front_size']} soluciones no dominadas y muestra el compromiso F1-carga de contactos.")

with tab3:
    st.subheader('Simula un cliente y explica su riesgo')
    st.write('Introduzca valores del cliente. La aplicación carga el pipeline guardado: **no se reentrena**.')
    X=df.drop(columns=['Churn','customerID'])
    numeric=['tenure','MonthlyCharges','TotalCharges']
    fields={}
    with st.form('prediction_form'):
        cols=st.columns(3)
        for i,name in enumerate(X.columns):
            with cols[i%3]:
                if name=='tenure':
                    fields[name]=st.number_input('tenure · meses',min_value=0,max_value=200,value=6,step=1)
                elif name=='MonthlyCharges':
                    fields[name]=st.number_input('MonthlyCharges · USD',min_value=0.0,max_value=1000.0,value=80.0,step=5.0)
                elif name=='TotalCharges':
                    fields[name]=st.number_input('TotalCharges · USD',min_value=0.0,max_value=100000.0,value=480.0,step=50.0)
                else:
                    values=sorted(X[name].dropna().unique().tolist(),key=str)
                    preferred={'Contract':'Month-to-month','InternetService':'Fiber optic','PaymentMethod':'Electronic check','OnlineSecurity':'No','TechSupport':'No'}
                    default=values.index(preferred[name]) if name in preferred and preferred[name] in values else 0
                    fields[name]=st.selectbox(name,values,index=default)
        submitted=st.form_submit_button('Calcular riesgo y explicar',type='primary')
    if submitted:
        row=pd.DataFrame([fields],columns=X.columns)
        prob=float(model.predict_proba(row)[0,1])
        m1,m2=st.columns(2)
        m1.metric('Riesgo estimado de abandono',f'{prob:.1%}')
        m2.metric('Clasificacion al umbral 50%','Riesgo alto' if prob>=.5 else 'Riesgo bajo')
        if prob<.5: st.info('Riesgo más bajo respecto al umbral de 50%. No significa riesgo cero.')
        else: st.warning('Requiere consideración para el equipo de retención; no autoriza descuentos automáticos.')
        p=model.named_steps['preprocessor']; cls=model.named_steps['model']
        vec=np.asarray(p.transform(row),dtype=float)
        bg=np.load(ART/'background_encoded.npy')
        names=p.get_feature_names_out()
        explainer=shap.LinearExplainer(cls,bg)
        sv=explainer(vec).values[0]
        shap_s=pd.Series(sv,index=names).reindex(pd.Series(abs(sv),index=names).nlargest(9).index).sort_values()
        lime_pairs,lime_method,fidelity=lime_for_encoded(vec[0],bg,cls.predict_proba,names,seed=42)
        c1,c2=st.columns(2)
        with c1:
            fig=px.bar(x=shap_s.values,y=[n.replace('cat__','').replace('num__','') for n in shap_s.index],orientation='h',title='SHAP: contribución al log-odds de abandono')
            fig.update_layout(xaxis_title='Contribución (log-odds)',yaxis_title='')
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            names_l=[x[0] for x in lime_pairs][::-1];scores_l=[x[1] for x in lime_pairs][::-1]
            fig=px.bar(x=scores_l,y=names_l,orientation='h',title=f'{lime_method}: sustituto local')
            fig.update_layout(xaxis_title='Efecto local sobre la predicción',yaxis_title='')
            st.plotly_chart(fig,use_container_width=True)
        st.caption(f'Fidelidad aproximada del sustituto LIME: {fidelity:.2f}. SHAP y LIME usan escalas distintas; los coeficientes no son causalidad. En LIME, un one-hot perturbado puede no respetar combinaciones reales.')
        st.info('Interpretación para negocio: las variables ayudan a explicar el score, pero deben validarse las posibles acciones mediante un piloto o experimento A/B.')

with tab4:
    st.subheader('¿Puede ayudar a priorizar una campaña?')
    st.write('Simulación en el conjunto de prueba anonimizado (1.409 clientes). **Todos los supuestos económicos son hipotéticos**.')
    c1,c2,c3,c4=st.columns(4)
    with c1: frac=st.slider('Porcentaje a contactar',1,30,10,1)/100
    with c2: cost=st.number_input('Costo por contacto (USD)',0.0,500.0,4.0,1.0)
    with c3: margin=st.number_input('Contribución anual por cliente retenido (USD)',0.0,10000.0,360.0,20.0)
    with c4: efficacy=st.slider('Eficacia supuesta de intervención',0,100,20,5)/100
    fee=st.number_input('Licencia ilustrativa por campaña (USD)',0.0,5000.0,249.0,10.0)
    n=max(1,int(np.ceil(len(ranking)*frac)))
    top=ranking.sort_values('p_churn',ascending=False).head(n)
    expected=sum(top.p_churn)*efficacy
    gross=expected*margin
    net=gross-n*cost-fee
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Clientes a contactar',f'{n}')
    c2.metric('Suma de scores NO calibrados',f'{top.p_churn.sum():.1f}')
    c3.metric('Retenciones hipoteticas',f'{expected:.1f}')
    c4.metric('Resultado neto simulado',f'${net:,.0f}')
    frames=[]
    for pct in range(1,31):
        n2=int(np.ceil(len(ranking)*pct/100));v=ranking.sort_values('p_churn',ascending=False).head(n2)
        frames.append({'contacto_pct':pct,'beneficio_neto_hipotetico':v.p_churn.sum()*efficacy*margin-n2*cost-fee})
    fig=px.line(pd.DataFrame(frames),x='contacto_pct',y='beneficio_neto_hipotetico',markers=True,
         title='Sensibilidad: resultado hipotético vs. capacidad de contacto')
    fig.update_layout(xaxis_title='Porcentaje contactado',yaxis_title='USD por campaña (escenario, no observado)')
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('''**Producto comercial:** diagnóstico, ranking de riesgo, explicación caso por caso y simulador de presupuesto. **Comprador:** responsable de CRM o retención. **Propuesta de piloto:** validar métricas de campaña con grupo de control, revisar privacidad, sesgos, calibración y cambios del mercado. **Precio de USD 249:** ilustrativo, no cotización real.''')
    st.error('Los scores NO están calibrados: el promedio de probabilidad predicha en test fue 41,6% vs. 26,5% de abandono observado. Su suma no es un pronóstico fiable. Además, el score no mide el efecto de la intervención; eficacia, margen, ahorro y ROI son SUPUESTOS. Calibrar en TRAIN y evaluar un piloto controlado antes de usarlo para presupuestos.')

st.divider()
st.caption('RetenIA · Proyecto académico individual · Fuente de datos: Telco Customer Churn (dataset de demostración) · Sin almacenamiento de formularios')
