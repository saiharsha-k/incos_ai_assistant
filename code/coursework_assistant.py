import streamlit as st
import json

# Custom CSS to set the color scheme
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #d35400;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    .sidebar .sidebar-content {
        background-color: #2c2c2c;
    }
    h1, h2, h3 {
        color: #e67e22;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Deep Learning Coursework Assistant")
st.write("Welcome to your coursework assistant. Ask questions or navigate through the sections below.")

# Sidebar for quick navigation
st.sidebar.header("Quick Links")
section = st.sidebar.radio("Go to:", ["Overview", "Deadlines", "Guidelines", "Resources"])

# Main content area
if section == "Overview":
    st.header("Coursework Overview")
    # Load and display overview information from JSON
    with open('E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Coursework\coursework_data\Coursework_Overview.json', 'r') as file:
        overview = json.load(file)
    st.write(overview['content'])
    for component in overview['components']:
        st.write(f"- {component['name']}: {component['weight']}%")

elif section == "Deadlines":
    st.header("Deadlines and Milestones")
    # Load and display deadline information from JSON
    with open('E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Coursework\coursework_data\Deadlines_and_Milestones.json', 'r') as file:
        deadlines = json.load(file)
    for milestone in deadlines['milestones']:
        st.write(f"**{milestone['name']}:** {milestone['due_date']}")

elif section == "Guidelines":
    st.header("Coursework Guidelines")
    guideline_type = st.selectbox("Select guideline:", ["Project Proposal", "Technical Report", "Source Code", "Presentation"])
    # Load and display selected guideline from JSON
    filename = f"{guideline_type.replace(' ', '_')}_Guidelines.json"
    with open(filename, 'r') as file:
        guidelines = json.load(file)
    st.write(guidelines['content'])
    for section in guidelines['sections']:
        st.write(f"**{section['name']}:** {section['description']}")

elif section == "Resources":
    st.header("Additional Resources")
    # Load and display resource information from JSON
    with open('E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Coursework\coursework_data\General_Guidance_and_Best_Practices.json', 'r') as file:
        resources = json.load(file)
    for resource in resources['resources']:
        st.write(f"- [{resource['name']}]({resource['url']})")

# Chatbot interface
st.header("Ask a Question")
user_question = st.text_input("Type your question here:")
if st.button("Send"):
    # Here you would typically send the user's question to your backend for processing
    # and then display the response from the chatbot
    st.write("Chatbot: I'm sorry, I'm still learning how to answer questions. Please refer to the sections above for information.")

# Display ethical considerations
st.header("Ethical Considerations")
with open('E:\Sai Harsha\From_23\MSc Data Science\DeepLearning_Coursework\coursework_data\Ethical_Considerations.json', 'r') as file:
    ethics = json.load(file)
st.write(ethics['content'])
for consideration in ethics['considerations']:
    st.write(f"- {consideration}")