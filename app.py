import streamlit as st
import joblib

file = open('tree.pkl','rb')
model = joblib.load(file)

st.title('Wine Quality Prediction')
st.text("Put in your wine's features")

# df_filtered=df.query("species==@quality")
# st.dataframe(df_filtered.sample(10))
fa=st.slider('fixed acidity',4.6,16.0,12.6,step=0.1)
va=st.slider('volatile acidity',0.12,1.60,1.04,step=0.01)
ca=st.slider('citric acid',0.00,1.00,0.95,step=0.01)
rs=st.slider('residual sugar',0.9,15.5,12.6,step=0.1)
c=st.slider('chlorides',0.012,0.600,0.567,step=0.001)
fsd=st.slider('free sulfur dioxide',1,70,55,step=1)
tsd=st.slider('total sulfur dioxide',6,250,200,step=1)
d=st.slider('density',0.9901,1.0100,1.0000,step=0.0001)
p=st.slider('pH',2.70,4.00,2.00,step=0.01)
s=st.slider('sulphates',0.00,1.00,0.90,step=0.01)
a=st.slider('alcohol',8.0,15.0,10.0,step=0.1)
button=st.button('predict')


if button:
    y_pred=model.predict([[fa,va,ca,rs,c,fsd,tsd,d,p,s,a]])
    st.subheader(y_pred)