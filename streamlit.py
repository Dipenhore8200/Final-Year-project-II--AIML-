import streamlit as st
import pickle
import numpy as np
import os
import xgboost as xgb
# Load the pre-trained machine learning model
xgboost =  xgb.XGBClassifier()
xgboost.load_model('xgb.model')
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
def predict_values(input_features):
    try:
        # Convert the list to a NumPy array with shape (1, -1)
        input_array = np.array(input_features).reshape(1, -1)
        
        # Make prediction using the loaded XGBoost model
        prediction = xgboost.predict(input_array)

        # You can format the prediction as needed
        if prediction[0] == 0:
            formatted_prediction = 'the prtein is HIV positive'
        elif prediction[0] ==1 :
            formatted_prediction = 'the prtein is HIV Negative'

        return formatted_prediction

    except Exception as e:
        # Handle any errors that may occur during prediction
        error_message = f'Error during prediction: {str(e)}'
        return error_message

def main():
    st.markdown(
        """
        <style>
            body {
                background-image: url('img.jpg');
                background-size: cover;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title('Your Machine Learning Web App')

    # Fieldset 1: Network Properties
    st.header('Network Properties')
    subgraph = st.number_input('Subgraph', value=0.0,format="%.8f")
    degree = st.number_input('Degree', value=0.0)
    eigenvector = st.number_input('Eigenvector', value=0.0,format="%.8f")
    information = st.number_input('Information', value=0.0,format="%.8f")
    lac = st.number_input('LAC', value=0.0)
    betweenness = st.number_input('Betweenness', value=0.0,format="%.8f")
    closeness = st.number_input('Closeness', value=0.0,format="%.8f")
    network = st.number_input('Network', value=0.0,format="%.8f")

    # Fieldset 2: Standard Physico-chemical Properties
    st.header('Standard Physico-chemical Properties')
    pcp_pc = st.number_input('pcp_pc(Positively charged)', value=0.0,format="%.8f")
    pcp_nc = st.number_input('pcp_nc(Negatively charged)', value=0.0,format="%.8f")
    pcp_ne = st.number_input('PCP_NE(Neutral charged)', value=0.0,format="%.8f")
    pcp_po = st.number_input('pcp_po(Polarity)', value=0.0,format="%.8f")
    pcp_np = st.number_input('PCP_NP(Non-polarity)', value=0.0,format="%.8f")
    pcp_al = st.number_input('PCP_AL(ALIPHATICITY)', value=0.0,format="%.8f")
    pcp_cy = st.number_input('pcp_CY(CYCLIC)', value=0.0,format="%.8f")
    pcp_ar = st.number_input('pcp_AR(AROMATICITY)', value=0.0,format="%.8f")
    pcp_ac = st.number_input('pcp_AC(ACIDITY)', value=0.0,format="%.8f")
    pcp_bs = st.number_input('PCP_BS(BASICITY)', value=0.0,format="%.8f")
    pcp_ne_ph = st.number_input('PCP_NE(NEUTRAL(PH))', value=0.0,format="%.8f")
    pcp_hb = st.number_input('PCP_HB(Hydrophobicity)', value=0.0,format="%.8f")
    pcp_hl = st.number_input('PCP_HL(Hydrophilicity)', value=0.0,format="%.8f")
    pcp_nt = st.number_input('PCP_NT(NEWTRAL)', value=0.0,format="%.8f")
    pcp_hx = st.number_input('PCP_HX(Hydroxylic)', value=0.0,format="%.8f")
    pcp_sc = st.number_input('PCP_SC(Sulphur content)', value=0.0,format="%.8f")
    pcp_tn = st.number_input('PCP_tn(Tiny)', value=0.0,format="%.8f")
    pcp_sl = st.number_input('PCP_sl(Small)', value=0.0,format="%.8f")
    pcp_lr = st.number_input('PCP_lr(Large)', value=0.0,format="%.8f")

    # Fieldset 3: Advance Physico-chemical Properties
    st.header('Advance Physico-chemical Properties')
    pcp_z1 = st.number_input('pcp_z1', value=0.0,format="%.8f")
    pcp_z2 = st.number_input('pcp_z2', value=0.0,format="%.8f")
    pcp_z3 = st.number_input('pcp_z3', value=0.0,format="%.8f")
    pcp_z4 = st.number_input('pcp_z4', value=0.0,format="%.8f")
    pcp_z5 = st.number_input('pcp_z5', value=0.0,format="%.8f")

    # Fieldset 4: Structural Physico-chemical Properties
    st.header('Structural Physico-chemical Properties')
    secondary_structure_helix = st.number_input('Secondary Structure(Helix)', value=0.0,format="%.8f")
    secondary_structure_strands = st.number_input('Secondary Structure(Strands)', value=0.0,format="%.8f")
    secondary_structure_coil = st.number_input('Secondary Structure(Coil)', value=0.0,format="%.8f")
    solvent_accessibility_buried = st.number_input('Solvent Accessibilty(Buried)', value=0.0,format="%.8f")
    solvent_accessibility_exposed = st.number_input('Solvent Accesibilty(Exposed)', value=0.0,format="%.8f")
    solvent_accessibility_intermediate = st.number_input('Solvent Accesibilty(Intermediate)', value=0.0,format="%.8f")

    # Fieldset 5: Shannon Entropy of Residues
    st.header('Shannon Entropy of Residues')
    ser_a = st.number_input('SER-A', value=0.0,format="%.8f")
    ser_c = st.number_input('SER-C', value=0.0,format="%.8f")
    ser_d = st.number_input('SER-D', value=0.0,format="%.8f")
    ser_e = st.number_input('SER-E', value=0.0,format="%.8f")
    ser_f = st.number_input('SER-F', value=0.0,format="%.8f")
    ser_g = st.number_input('SER-G', value=0.0,format="%.8f")
    ser_h = st.number_input('SER-H', value=0.0,format="%.8f")
    ser_i = st.number_input('SER-I', value=0.0,format="%.8f")
    ser_k = st.number_input('SER-K', value=0.0,format="%.8f")
    ser_l = st.number_input('SER-L', value=0.0,format="%.8f")
    ser_m = st.number_input('SER-M', value=0.0,format="%.8f")
    ser_n = st.number_input('SER-N', value=0.0,format="%.8f")
    ser_p = st.number_input('SER-P', value=0.0,format="%.8f")
    ser_q = st.number_input('SER-Q', value=0.0,format="%.8f")
    ser_r = st.number_input('SER-R', value=0.0,format="%.8f")
    ser_s = st.number_input('SER-S', value=0.0,format="%.8f")
    ser_t = st.number_input('SER-T', value=0.0,format="%.8f")
    ser_v = st.number_input('SER-V', value=0.0,format="%.8f")
    ser_w = st.number_input('SER-W', value=0.0,format="%.8f")
    ser_y = st.number_input('SER-Y', value=0.0,format="%.8f")
    st.header('Shannon Entropy of protien')
    sep_pp = st.number_input('SEp-pp', value=0.0,format="%.8f")

    # Create a list of input features
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

    # Make prediction when the button is clicked
    if st.button('Predict'):
        prediction = predict_values(input_features)
        st.success(prediction)

if __name__ == '__main__':
    main()
