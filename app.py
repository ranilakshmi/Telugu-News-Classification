import streamlit as st
import tensorflow as tf
st.title("Telugu News Classification")
txt = st.text_area("News article")
model = tf.keras.models.load_model('model')
pred_probs = model.predict([txt])
y_preds = tf.argmax(pred_probs, axis=1)
class_names = ['business', 'editorial', 'entertainment', 'nation', 'sports']
if st.button('Predict'):
    st.write(f"Predicted label: {class_names[y_preds[[0]]]}\n")