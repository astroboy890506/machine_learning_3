import streamlit as st
import nltk
nltk.download('names')
names = [(name, 'male') for name in nltk.corpus.names.words("male.txt")]
names += [(name, 'female') for name in nltk.corpus.names.words("female.txt")]

def extract_gender_features(name):  #extract the feature
    name = name.lower()
    features = {}
    features["suffix"] = name[-1:]
    features["suffix2"] = name[-2:] if len(name) > 1 else name[0]
    features["suffix3"] = name[-3:] if len(name) > 2 else name[0]
    features["suffix4"] = name[-4:] if len(name) > 3 else name[0]
    features["suffix5"] = name[-5:] if len(name) > 4 else name[0]
    features["suffix6"] = name[-6:] if len(name) > 5 else name[0]
    features["prefix"] = name[:1]
    features["prefix2"] = name[:2] if len(name) > 1 else name[0]
    features["prefix3"] = name[:3] if len(name) > 2 else name[0]
    features["prefix4"] = name[:4] if len(name) > 3 else name[0]
    features["prefix5"] = name[:5] if len(name) > 4 else name[0]
    #features["wordLen"] = len(name)
    #print (features)
    #for letter in "abcdefghijklmnopqrstuvwyxz":
    #    features[letter + "-count"] = name.count(letter)
   
    return features

data = [(extract_gender_features(name), gender) for (name,gender) in names]
print (data)  #categarize features with corresponding genders

import random
random.shuffle(data)

dataCount = len(data)
trainCount = int(.8*dataCount) #in ML, we have to split data into 30-70 or 20-80

trainData = data[:trainCount]
testData = data[trainCount:]
bayes = nltk.NaiveBayesClassifier.train(trainData)

def main():
    st.set_page_config(
        page_title="Gender Prediction App",
        page_icon=":boy:",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("Gender Prediction from Names")
    
    # Multiline text input for names
    names_input = st.text_area("Enter names (one per line):", "", height=150)
    
    # Prediction button
    if st.button("Predict Gender"):
        names_list = names_input.strip().split("\n")
        
        if names_list:
            st.write("Predicted Genders:")
            for name in names_list:
                gender = bayes.classify(extract_gender_features(name))
                st.write(f"Name: {name}, Predicted Gender: {gender.capitalize()}")
        else:
            st.warning("Please enter at least one name for prediction.")
    
if __name__ == "__main__":
    main()


