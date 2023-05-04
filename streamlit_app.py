import numpy as np
import pickle
import streamlit as st
import sklearn
import time

# loading the saved model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    # giving a title
    st.title('Diabetes Prediction Web App')

    var = st.sidebar.radio("Choose the options", ["HOME", "DIABETIC PREDICTION", "CONTACT US"])
    if(var=="HOME"):


        image1_path = "C:\\Users\\KIIT\\PycharmProjects\\minorproject\\images\\diabetic_NEW.jpg"
        image2_path = "C:\\Users\\KIIT\\PycharmProjects\\minorproject\\images\\dia.jpg"
        image3_path = "C:\\Users\\KIIT\\PycharmProjects\\minorproject\\images\\diab4.jpg"
        image4_path = "C:\\Users\\KIIT\\PycharmProjects\\minorproject\\images\\Diabete3.jpg"
        image5_path = "C:\\Users\\KIIT\\PycharmProjects\\minorproject\\images\\diabetes-symptoms-infographic-free-vector.jpg"
        image6_path = "C:\\Users\\KIIT\\PycharmProjects\\minorproject\\images\\diabetics2.jpg"
        
        im1 = st.image(image1_path, caption="Image 1", width=800)
        im2 = st.image(image2_path, caption="Image 2", width=800)
        im3 = st.image(image3_path, caption="Image 3", width=800)
        im4 = st.image(image4_path, caption="Image 4", width=800)
        im5 = st.image(image5_path, caption="Image 5", width=800)
        im6 = st.image(image6_path, caption="Image 6", width=800)




        images = [im1,im2,im3,im4,im5,im6]
        time_interval = 8

        # loop through the list of images
        for image_url in images:
            # display the image
            st.image(image_url,)

            # wait for the specified time interval
            time.sleep(time_interval)

    if(var=="CONTACT US"):
        print("CONTACT US")
        name, phone = st.columns(2)
        with name:
            a = st.text("NISHANT")
        with phone:
            b = st.text("7255925971")
        name2, phone2 = st.columns(2)
        with name2:
            a = st.text("SAGNIK GHOSH")
        with phone2:
            b = st.text("6290923702")
        name3, phone3 = st.columns(2)
        with name3:
            a = st.text("SANCHIT KUMAR")
        with phone3:
            b = st.text("9861954223")
        st.write(
            "for more details click on these link https://en.wikipedia.org/wiki/Diabetes")

    if(var=="DIABETIC PREDICTION"):
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure value')
        SkinThickness = st.text_input('Skin Thickness value')
        Insulin = st.text_input('Insulin Level')
        BMI = st.text_input('BMI value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        Age = st.text_input('Age of the Person')

        # code for Prediction
        diagnosis = ''

        # creating a button for Prediction

        if st.button('Diabetes Test Result'):
            diagnosis = diabetes_prediction(
                [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])


            prog = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                prog.progress(i + 1)
            st.balloons()
        st.success(diagnosis)


    # getting the input data from the user



if __name__ == '__main__':
    main()
