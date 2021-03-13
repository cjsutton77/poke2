#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
from fastbook import *;
from fastai.vision.widgets import *;
from fastai.vision.all import *;
from PIL import Image

# In[12]:


path = Path()
learn_inf = load_learner(path/'export_031221.pkl', cpu = True)
learn_inf2 = load_learner(path/'export_031021.pkl', cpu = True)

# In[13]:


#img = PILImage.create('/Users/christiansutton/Downloads/this-footage-of-six-magikarp-flopping-around-a-pokemon-sword-and-shield-camp-is-cursed-but-beautiful.jpeg')


# In[16]:


#display(img.to_thumb(244))


# In[17]:


#pred,pred_idx,probs = learn_inf.predict(img)
#print(f'Poke-Prediction: {pred}; Poke-Probability: {probs[pred_idx]:.04f}')


# In[ ]:


st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    #red,pred_idx,probs = learn_inf.predict(img)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    #st.write("")
    st.markdown("**Classifying...**.")
    pred,pred_idx,probs = learn_inf.predict(np.array(img))
    pred2,pred_idx2,probs2 = learn_inf2.predict(np.array(img))
    add_selectbox = st.sidebar.selectbox(
    "Do you want current or old model",
    ("Current", "old"))
    if add_selectbox == "Current":
        st.markdown(f'*Current model*:')
        st.write(f'This is a poke-prediction of a {pred} with a poke-probability of {100*probs[pred_idx]:.03f}%.')
    else:
        st.markdown(f'*Old model*:')
        st.write(f'This is a poke-prediction of a {pred2} with a poke-probability of {100*probs2[pred_idx2]:.03f}%.')
    #label = predict(uploaded_file)
    #st.write('%s (%.2f%%)' % (label[1], label[2]*100))

