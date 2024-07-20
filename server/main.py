from typing import Union
import boto3
from fastapi import FastAPI
from pydantic import BaseModel



class Payload(BaseModel):
    text: str

# Initialize the AWS Comprehend Medical client
comprehend_medical = boto3.client(service_name='comprehendmedical', region_name='us-east-1')

app = FastAPI()

def get_icd10_codes(text):
    # Call the detect_entities_v2 API
    result = comprehend_medical.detect_entities_v2(Text=text)

    # Extract the ICD-10-CM codes from the response
    icd10_codes = []
    for entity in result['Entities']:
        if 'ICD10CMConcepts' in entity:
            for concept in entity['ICD10CMConcepts']:
                icd10_codes.append({
                    'Code': concept['Code'],
                    'Description': concept['Description'],
                    'Score': concept['Score']
                })
    return icd10_codes


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






