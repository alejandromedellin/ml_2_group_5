import streamlit as st
from PIL import Image

st.image('assets/logoparamonegro.png', width=200)
st.markdown("---")
st.title(" 👩‍💻👨‍💻 Development Team")

st.markdown("""This is the team of developers who made PáramoGPT possible. 
			If there are any doubts about the development of the app, 
			the deployment of the app to the cloud, or even the app itself, please feel free to reach out to any of them. 
			They will be happy to answer your questions.""")

st.write("")

#Fixing the images
## Samir
samir_image = Image.open("assets/samir_image.jpg")
samir_image = samir_image.resize((300, 300))

## Alejandro
alejandro_image = Image.open("assets/alejandro_image.jpg")
alejandro_image = alejandro_image.resize((300, 300))

## Thomas
thomas_image = Image.open("assets/thomas_image.jpg")
thomas_image = thomas_image.resize((300, 300))

## Nour
nour_image = Image.open("assets/nour_image.jpg")
nour_image = nour_image.resize((300, 300))

## Joy
joy_image = Image.open("assets/joy_image.jpg")
joy_image = joy_image.resize((300, 300))

#continue here to add pics
col1,col2,col3,col4,col5=st.columns(5)
with col1:
    st.image(samir_image)
    st.markdown('*Samir Barakat*')
    
with col2:
	st.image(alejandro_image)
	st.markdown('*Alejandro Medellín*')
	
	
with col3:
	st.image(thomas_image)
	st.markdown('*Thomas Renwick*')
	
	
with col4:
	st.image(nour_image)
	st.markdown('*Nour Sewilam*')
	
	
with col5:
	st.image(joy_image)
	st.markdown('*Joy Zhong*')
	
st.write("")

st.subheader("GitHub Repo")

st.markdown("""This is an OpenSource project. Therefore, the GitHub repository which the includes the code 
			used to carry out this project can be found following this link: 
			https://github.com/alejandromedellin/ml_2_group_5""")
	





