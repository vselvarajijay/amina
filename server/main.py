from typing import Union
import boto3
from fastapi import FastAPI
from pydantic import BaseModel



class Payload(BaseModel):
    text: str

# Initialize the AWS Comprehend Medical client
session = boto3.Session(profile_name='Dev')
comprehend_medical = session.client(service_name='comprehendmedical', region_name='us-east-1')

app = FastAPI()

def get_icd10_codes_and_symptoms(text):
    try:
        # Call the detect_entities_v2 API
        result = comprehend_medical.infer_icd10_cm(Text=text)

        # Extract the ICD-10-CM codes and symptoms from the response
        icd10_codes = []
        symptoms = []

        for entity in result['Entities']:
            # Extract ICD-10-CM codes if available
            if 'ICD10CMConcepts' in entity and entity['ICD10CMConcepts']:
                for concept in entity['ICD10CMConcepts']:
                    icd10_codes.append({
                        'Code': concept['Code'],
                        'Description': concept['Description'],
                        'Score': concept['Score']
                    })
            # Extract symptoms if the entity is categorized as a symptom
            if 'Traits' in entity:
                for trait in entity['Traits']:
                    if trait['Name'] == 'SYMPTOM':
                        symptoms.append({
                            'Symptom': entity['Text'],
                            'Score': trait['Score']
                        })

        return {
            'ICD10Codes': icd10_codes,
            'Symptoms': symptoms
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            'ICD10Codes': [],
            'Symptoms': []
        }



@app.post("/text-chunk")
def text_chunk(payload: Payload):
    # Sample text for lookup
    #text = "The patient was diagnosed with diabetes mellitus and hypertension."
    text = payload.text

    # Get the ICD-10-CM codes for the text
    icd10_codes = get_icd10_codes(text)

    # Print the ICD-10-CM codes
    for code in icd10_codes:
        print(f"Code: {code['Code']}, Description: {code['Description']}, Score: {code['Score']}")
