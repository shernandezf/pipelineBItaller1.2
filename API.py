from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
from joblib import load

app = FastAPI()
class DataModelapp(BaseModel):
    Limpiostudycondition: str
@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.post("/Data/predict")
async def make_prediction(dataModel: DataModelapp):
    
    df = pd.DataFrame(dataModel.dict(),index=[0])
    model = load("modelRandomForest.joblib")
    result = model.predict(df)
    resultado=result[0]
    return {"Resultado": resultado}
    