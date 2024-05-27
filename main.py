from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import xgboost as xgb



app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

xgboost = xgb.XGBClassifier()
xgboost.load_model('xgb.model')
df = pd.read_csv("hiv_dataset_deployment.csv")

def predict_values(input_features):
    try:
        # Convert the list to a NumPy array with shape (1, -1)
        input_array = np.array(input_features).reshape(1, -1)
        
        # Make prediction using the loaded XGBoost model
        prediction = xgboost.predict(input_array)
        print(prediction)
        # You can format the prediction as needed
        if prediction[0] == 0:
            formatted_prediction = 'the protein is HIV Negative'
        elif prediction[0] == 1:
            formatted_prediction = 'the protein is HIV positive'

        return formatted_prediction

    except Exception as e:
        # Handle any errors that may occur during prediction
        error_message = f'Error during prediction: {str(e)}'
        return error_message

@app.get("/contributors", response_class=HTMLResponse)
async def contributors_page():
    # Your logic to fetch contributors and render HTML
    return {"message": "List of contributors"}


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("ss.html", {"request": request})

@app.post("/predict2/")
async def predict2(request: Request, protein_selection: str = Form("...")):
    df_new = df[df['Entry'] == protein_selection]
    X =df_new['LABEL'].iloc[-1]
    ##print(df_new)
    df_new = df_new.drop(['Entry', 'LABEL'], axis=1)
    result_text = predict_values(df_new)
    #print(protein_selection,df_new.shape,X)
    return templates.TemplateResponse("ss.html", {"request": request, "selected_protein": result_text,"selected_proteinA":X})

@app.post("/predict/")
async def predict(
    request: Request,
    subgraph: float = Form(0.00),
    degree: float = Form(0.00),
    eigenvector: float = Form(0.00),
    information: float = Form(0.00),
    lac: float = Form(0.00),
    betweenness: float = Form(0.00),
    closeness: float = Form(0.00),
    network: float = Form(0.00),
    pcp_pc: float = Form(0.00),
    pcp_nc: float = Form(0.00),
    pcp_ne: float = Form(0.00),
    pcp_po: float = Form(0.00),
    pcp_np: float = Form(0.00),
    pcp_al: float = Form(0.00),
    pcp_cy: float = Form(0.00),
    pcp_ar: float = Form(0.00),
    pcp_ac: float = Form(0.00),
    pcp_bs: float = Form(0.00),
    pcp_ne_ph: float = Form(0.00),
    pcp_hb: float = Form(0.00),
    pcp_hl: float = Form(0.00),
    pcp_nt: float = Form(0.00),
    pcp_hx: float = Form(0.00),
    pcp_sc: float = Form(0.00),
    pcp_tn: float = Form(0.00),
    pcp_sl: float = Form(0.00),
    pcp_lr: float = Form(0.00),
    pcp_z1: float = Form(0.00),
    pcp_z2: float = Form(0.00),
    pcp_z3: float = Form(0.00),
    pcp_z4: float = Form(0.00),
    pcp_z5: float = Form(0.00),
    secondary_structure_helix: float = Form(0.00),
    secondary_structure_strands: float = Form(0.00),
    secondary_structure_coil: float = Form(0.00),
    solvent_accessibility_buried: float = Form(0.00),
    solvent_accessibility_exposed: float = Form(0.00),
    solvent_accessibility_intermediate: float = Form(0.00),
    ser_a: float = Form(0.00),
    ser_c: float = Form(0.00),
    ser_d: float = Form(0.00),
    ser_e: float = Form(0.00),
    ser_f: float = Form(0.00),
    ser_g: float = Form(0.00),
    ser_h: float = Form(0.00),
    ser_i: float = Form(0.00),
    ser_k: float = Form(0.00),
    ser_l: float = Form(0.00),
    ser_m: float = Form(0.00),
    ser_n: float = Form(0.00),
    ser_p: float = Form(0.00),
    ser_q: float = Form(0.00),
    ser_r: float = Form(0.00),
    ser_s: float = Form(0.00),
    ser_t: float = Form(0.00),
    ser_v: float = Form(0.00),
    ser_w: float = Form(0.00),
    ser_y: float = Form(0.00),
    sep_pp: float = Form(0.00),
):
    # Calculate the sum of input values
    input_features = [
    subgraph, degree, eigenvector, information, lac, betweenness, closeness, network,
    pcp_pc, pcp_nc, pcp_ne, pcp_po, pcp_np, pcp_al, pcp_cy, pcp_ar, pcp_ac, pcp_bs,
    pcp_ne_ph, pcp_hb, pcp_hl, pcp_nt, pcp_hx, pcp_sc, pcp_tn, pcp_sl, pcp_lr,
    pcp_z1, pcp_z2, pcp_z3, pcp_z4, pcp_z5,
    secondary_structure_helix, secondary_structure_strands, secondary_structure_coil,
    solvent_accessibility_buried, solvent_accessibility_exposed, solvent_accessibility_intermediate,
    ser_a, ser_c, ser_d, ser_e, ser_f, ser_g, ser_h, ser_i, ser_k, ser_l, ser_m, ser_n,
    ser_p, ser_q, ser_r, ser_s, ser_t, ser_v, ser_w, ser_y, sep_pp
        ]
    prediction = predict_values(input_features)
   
    return templates.TemplateResponse("ss.html", {"request": request, "value": prediction})
