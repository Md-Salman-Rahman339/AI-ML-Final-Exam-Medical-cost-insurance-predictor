import gradio as gr
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "insurance_model.pkl"

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)


def predict_charges(age, sex, bmi, children, smoker, region):
    
    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region
    ]], columns=[
        'age', 'sex', 'bmi', 'children', 'smoker', 'region'
    ])

    input_df['bmi_category'] = pd.cut(
        input_df['bmi'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=['underweight', 'normal', 'overweight', 'obese']
    )

    input_df['age_group'] = pd.cut(
        input_df['age'],
        bins=[0, 18, 35, 50, 100],
        labels=['teen', 'young', 'adult', 'senior']
    )

    input_df['family_size'] = input_df['children'] + 1
    
 
    prediction = model.predict(input_df)[0]
    
    return f"Estimated Insurance Cost: ${prediction:.2f}"


inputs = [
    gr.Number(label="Age", value=25),
    
    gr.Radio(
        choices=["male", "female"], 
        label="Sex"
    ),
    
    gr.Number(label="BMI", value=28.0),
    
    gr.Slider(
        minimum=0, maximum=5, step=1, 
        label="Number of Children"
    ),
    
    gr.Radio(
        choices=["yes", "no"], 
        label="Smoker"
    ),
    
    gr.Dropdown(
        choices=["northeast", "northwest", "southeast", "southwest"],
        label="Region"
    )
]

app = gr.Interface(
    fn=predict_charges,
    inputs=inputs,
    outputs="text",
    title="Insurance Cost Predictor",
    description="Predict medical insurance charges based on personal details"
)


app.launch()