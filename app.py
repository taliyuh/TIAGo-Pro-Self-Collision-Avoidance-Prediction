import streamlit as st
import json
import numpy as np
import pickle
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

# ----------------------------------------------------
# 1. Page Configuration & Custom Light Theme Injection
# ----------------------------------------------------
st.set_page_config(
    page_title="Tiago Pro Robot - Collision Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom light CSS overrides
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

/* Force clean light theme */
.stApp {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

[data-testid="stSidebar"] * {
    color: #334155 !important;
}

/* Sleek Typography */
h1, h2, h3, h4, h5, h6 {
    color: #0F172A !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}

p, span, label, ul, li {
    color: #334155 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Style native Streamlit containers with borders as custom cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
    margin-bottom: 16px !important;
}

/* Clean title for section cards */
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
    margin-top: 0px;
    margin-bottom: 16px;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 8px;
}

/* Mini stat badges */
.stat-badge {
    background-color: #F1F5F9;
    color: #1E293B;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-right: 6px;
    margin-bottom: 6px;
    border: 1px solid #E2E8F0;
}

/* Metric Display values */
.metric-large {
    font-family: 'Outfit', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #2563EB;
}

/* Custom Table Layout Adjustments */
table {
    border-spacing: 8px !important;
    border-collapse: separate !important;
}

td, th {
    transition: all 0.2s ease-in-out;
}

/* Hover effects for grid table cells */
.grid-cell-link {
    text-decoration: none !important;
    display: block;
    width: 100%;
    height: 100%;
}

/* Custom Safe and Collision Alert badges without emojis */
.alert-safe {
    background-color: #DCFCE7;
    border: 1px solid #86EFAC;
    color: #14532D;
    padding: 16px;
    border-radius: 8px;
    font-weight: 700;
    text-align: center;
    font-size: 18px;
}

.alert-collision {
    background-color: #FEE2E2;
    border: 1px solid #FCA5A5;
    color: #7F1D1D;
    padding: 16px;
    border-radius: 8px;
    font-weight: 700;
    text-align: center;
    font-size: 18px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. PyTorch Network Class & Model Caching
# ----------------------------------------------------
class DynamicCollisionPredictor(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.num_layers = len(layer_sizes)
        in_size = 17
        for i, out_size in enumerate(layer_sizes):
            setattr(self, f'fc{i+1}', nn.Linear(in_size, out_size))
            in_size = out_size
        self.out = nn.Linear(in_size, 2)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        for i in range(self.num_layers):
            fc = getattr(self, f'fc{i+1}')
            x = F.relu(fc(x))
            if i < self.num_layers - 1:
                x = self.drop(x)
        return self.out(x)

@st.cache_resource
def load_pytorch_model(weights_file, layers):
    model = DynamicCollisionPredictor(layers)
    model.load_state_dict(torch.load(f"{weights_file}.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_xgboost_model():
    model = xgb.XGBClassifier()
    model.load_model("xgboost_model.json")
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_performance_data():
    with open("model_performance.json", "r") as f:
        return json.load(f)

# Load resources
try:
    performance_data = load_performance_data()
    scaler = load_scaler()
    xgb_model = load_xgboost_model()
except Exception as e:
    st.error(f"Error loading base models or performance logs: {e}")
    st.stop()

# ----------------------------------------------------
# 4. XGBoost Reference Performance & Joint Limits
# ----------------------------------------------------
XGB_PERFORMANCE = {
    "accuracy": 0.8734,
    "precision": 0.8801,
    "recall": 0.7904,
    "f1_score": 0.8329,
    "auc_roc": 0.9448,
    "avg_time_per_sample_ms": 0.000214,
    "samples_per_second": 4667757.10,
    "confusion_matrix": [[111588, 8592], [16731, 63089]]
}

left_limits_deg = [(-280.0, 150.0), (-140.0, 0.0), (-160.0, 160.0), (-140.0, 0.0), (-100.0, 100.0), (-108.0, 108.0), (-150.0, 150.0)]
right_limits_deg = [(-40.0, 270.0), (-140.0, 0.0), (-160.0, 160.0), (-140.0, 0.0), (-220.0, -20.0), (-108.0, 108.0), (-150.0, 150.0)]
extra_limits_rad = [(0.0, 0.35), (-1.309, 1.309), (-1.0472, 0.34907)]

def get_limits_rad():
    limits = []
    for lo, hi in left_limits_deg:
        limits.append((np.radians(lo), np.radians(hi)))
    for lo, hi in right_limits_deg:
        limits.append((np.radians(lo), np.radians(hi)))
    limits.append(extra_limits_rad[0])
    limits.append(extra_limits_rad[1])
    limits.append(extra_limits_rad[2])
    return limits

joint_limits_rad = get_limits_rad()

pose_presets = {
    "Default Home Pose": [0.0] * 17,
    "Safe Configuration": [-0.46453, -0.46564, -2.75623, -0.47447, 0.69163, -0.60224, -1.80390, 4.48088, -1.62100, -2.27453, -2.20713, -0.88141, 0.39103, 1.60811, 0.25540, 0.09484, 0.31153],
    "Collision Configuration": [-0.08807, -2.38234, -1.25647, -1.89805, 0.82544, 0.66614, 2.05344, -0.22774, -1.41251, -2.62610, -1.90922, -2.07570, -1.78491, -1.57688, 0.22745, 0.11765, -0.73940]
}

# ----------------------------------------------------
# 5. State Initialization
# ----------------------------------------------------
if "prev_unit" not in st.session_state:
    st.session_state.prev_unit = "Radians"

# Initialize slider positions in session state if not already set
for idx in range(17):
    key = f"s_{idx}"
    if key not in st.session_state:
        st.session_state[key] = 0.0

# ----------------------------------------------------
# 6. Callbacks for Sliders & State Modifications
# ----------------------------------------------------
def convert_units():
    new_unit = st.session_state.units_radio
    prev_unit = st.session_state.prev_unit
    
    if prev_unit == new_unit:
        return
        
    for idx in range(17):
        val = st.session_state[f"s_{idx}"]
        if idx == 14: # Torso is always in meters
            continue
            
        if new_unit == "Degrees":
            # Convert Radians to Degrees
            lo, hi = joint_limits_rad[idx]
            val = np.clip(val, lo, hi)
            st.session_state[f"s_{idx}"] = float(np.degrees(val))
        else:
            # Convert Degrees to Radians
            if idx in [15, 16]:
                lo = float(np.degrees(joint_limits_rad[idx][0]))
                hi = float(np.degrees(joint_limits_rad[idx][1]))
            elif idx < 7:
                lo, hi = left_limits_deg[idx]
            else:
                lo, hi = right_limits_deg[idx - 7]
            val = np.clip(val, lo, hi)
            st.session_state[f"s_{idx}"] = float(np.radians(val))
            
    st.session_state.prev_unit = new_unit

def load_preset_pose(pose_name):
    current_unit = st.session_state.get("units_radio", "Radians")
    rad_vals = pose_presets[pose_name]
    
    for idx, rad_val in enumerate(rad_vals):
        lo, hi = joint_limits_rad[idx]
        rad_val = np.clip(rad_val, lo, hi)
        
        if current_unit == "Degrees" and idx != 14:
            st.session_state[f"s_{idx}"] = float(np.degrees(rad_val))
        else:
            st.session_state[f"s_{idx}"] = float(rad_val)

def randomize_pose():
    current_unit = st.session_state.get("units_radio", "Radians")
    for idx in range(17):
        lo, hi = joint_limits_rad[idx]
        rad_val = random.uniform(lo, hi)
        
        if current_unit == "Degrees" and idx != 14:
            st.session_state[f"s_{idx}"] = float(np.degrees(rad_val))
        else:
            st.session_state[f"s_{idx}"] = float(rad_val)

# ----------------------------------------------------
# 7. State Synchronization
# ----------------------------------------------------
query_params = st.query_params

selected_model = query_params.get("model", st.session_state.get("selected_model", "model_6"))
selected_width = query_params.get("width", st.session_state.get("selected_width", None))
selected_layers = query_params.get("layers", st.session_state.get("selected_layers", None))

if selected_width is not None:
    selected_width = int(selected_width)
if selected_layers is not None:
    selected_layers = int(selected_layers)

st.session_state.selected_model = selected_model
st.session_state.selected_width = selected_width
st.session_state.selected_layers = selected_layers

def clear_filters():
    st.query_params.clear()
    st.session_state.selected_model = "model_6"
    st.session_state.selected_width = None
    st.session_state.selected_layers = None
    st.rerun()

# ----------------------------------------------------
# 8. Render App Layout
# ----------------------------------------------------
st.markdown("""
<div style="background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; padding: 24px; margin-bottom: 24px;">
    <h1 style="margin: 0; font-size: 32px; color: #0F172A;">Tiago Pro Robot: Self-Collision Analysis</h1>
    <p style="margin: 6px 0 0 0; color: #475569; font-size: 15px;">
        Compare Neural Network architectures, evaluate model features, and perform live self-collision predictions in real-time.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.52, 0.48])

# ----------------------------------------------------
# COLUMN 1: NN Model Performance Grid & Details
# ----------------------------------------------------
with col1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Model Exploration Grid</div>', unsafe_allow_html=True)
        
        param_options = {
            "accuracy": "Accuracy Score",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "auc_roc": "AUC-ROC",
            "avg_time_per_sample_ms": "Average Latency (ms / sample)",
            "train_loss": "Final Training Loss",
            "valid_loss": "Final Validation Loss"
        }
        
        selected_param = st.selectbox(
            "Select performance metric to display in the grid:",
            options=list(param_options.keys()),
            format_func=lambda x: param_options[x],
            index=0
        )
        
        # Render dynamic HSL table
        def generate_grid_html(models_data, param_key, selected_model, selected_width, selected_layers):
            vals = [m.get(param_key) for m in models_data.values() if m.get(param_key) is not None]
            min_val, max_val = (min(vals), max(vals)) if vals else (0, 1)
            
            grid_structure = {
                3: {128: "model_1", 256: "model_2", 512: "model_3", 1024: "model_4"},
                4: {128: "model_5", 256: "model_6", 512: "model_7", 1024: "model_8"},
                5: {128: "model_9", 256: "model_10", 512: "model_11", 1024: "model_12"}
            }
            
            html = []
            html.append('<table style="width: 100%; border-collapse: separate; border-spacing: 8px;">')
            html.append('  <thead>')
            html.append('    <tr>')
            html.append('      <th style="border: none; background: transparent; width: 60px;"></th>')
            html.append('      <th style="border: none; background: transparent; width: 100px;"></th>')
            html.append('      <th colspan="4" style="text-align: center; border: none; font-size: 13px; color: #64748B; padding-bottom: 4px; font-weight: 600;">')
            html.append('        Width of Neural Network (Neurons in Hidden Layers) ➔')
            html.append('      </th>')
            html.append('    </tr>')
            html.append('    <tr>')
            html.append('      <th style="border: none; background: transparent;"></th>')
            html.append('      <th style="border: none; background: transparent;"></th>')
            
            for w in [128, 256, 512, 1024]:
                is_w_sel = (selected_width == w)
                hdr_style = (
                    "background-color: #2563EB; color: #FFFFFF; border: 1px solid #1D4ED8; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);"
                    if is_w_sel else
                    "background-color: #F1F5F9; color: #334155; border: 1px solid #E2E8F0;"
                )
                html.append(f'      <th style="{hdr_style} border-radius: 6px; padding: 10px; font-weight: 700; text-align: center; width: 22%; font-size: 13px;">')
                html.append(f'        <a href="?width={w}" target="_self" style="text-decoration: none; color: inherit; display: block; width: 100%; height: 100%;">{w} Units</a>')
                html.append(f'      </th>')
            html.append('    </tr>')
            html.append('  </thead>')
            html.append('  <tbody>')
            
            first_row = True
            for l in [3, 4, 5]:
                html.append('    <tr>')
                if first_row:
                    html.append('      <td rowspan="3" style="border: none; background: transparent; text-align: center; font-weight: 700; color: #475569; font-size: 13px; padding: 0 12px; line-height: 1.6; vertical-align: middle; width: 60px;">')
                    html.append('        L<br>a<br>y<br>e<br>r<br>s<br>▼')
                    html.append('      </td>')
                    first_row = False
                    
                # Compute active row selection check
                is_l_sel = (selected_layers == l)
                row_style = (
                    "background-color: #2563EB; color: #FFFFFF; border: 1px solid #1D4ED8; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);"
                    if is_l_sel else
                    "background-color: #F1F5F9; color: #334155; border: 1px solid #E2E8F0;"
                )
                html.append(f'      <td style="{row_style} border-radius: 6px; padding: 10px; font-weight: 700; text-align: center; font-size: 13px;">')
                html.append(f'        <a href="?layers={l}" target="_self" style="text-decoration: none; color: inherit; display: block; width: 100%; height: 100%;">{l} Layers</a>')
                html.append(f'      </td>')
                
                for w in [128, 256, 512, 1024]:
                    m_name = grid_structure[l][w]
                    m_data = models_data[m_name]
                    val = m_data.get(param_key)
                    
                    if val is None:
                        val_str = "N/A"
                        bg_color = "#F8FAFC"
                        border_color = "#E2E8F0"
                        text_color = "#94A3B8"
                    else:
                        if param_key in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
                            val_str = f"{val:.4f}"
                        elif param_key in ["avg_time_per_sample_ms"]:
                            val_str = f"{val:.5f} ms"
                        elif param_key in ["train_loss", "valid_loss"]:
                            val_str = f"{val:.4f}"
                        else:
                            val_str = str(val)
                        
                        ratio = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                        if param_key in ["avg_time_per_sample_ms", "train_loss", "valid_loss"]:
                            ratio = 1.0 - ratio
                        
                        hue = ratio * 120
                        bg_color = f"hsl({hue:.1f}, 90%, 93%)"
                        border_color = f"hsl({hue:.1f}, 60%, 78%)"
                        text_color = f"hsl({hue:.1f}, 100%, 20%)"
                    
                    is_m_sel = (selected_model == m_name)
                    cell_border = "3px solid #2563EB" if is_m_sel else f"1px solid {border_color}"
                    cell_shadow = "box-shadow: 0 4px 10px rgba(37,99,235,0.25);" if is_m_sel else ""
                    
                    html.append(f'      <td style="background-color: {bg_color}; border: {cell_border}; {cell_shadow} border-radius: 8px; padding: 12px; text-align: center; cursor: pointer;">')
                    html.append(f'        <a href="?model={m_name}" target="_self" style="text-decoration: none; display: block; width: 100%; height: 100%;">')
                    html.append(f'          <div style="font-size: 10px; font-weight: 600; color: #64748B; margin-bottom: 2px;">{m_name.upper()}</div>')
                    html.append(f'          <div style="font-size: 14px; font-weight: 700; color: {text_color};">{val_str}</div>')
                    html.append(f'        </a>')
                    html.append(f'      </td>')
                html.append('    </tr>')
            html.append('  </tbody>')
            html.append('</table>')
            return "\n".join(html)

        st.write(generate_grid_html(performance_data, selected_param, selected_model, selected_width, selected_layers), unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 12px; display: flex; justify-content: space-between; align-items: center;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 12px; color: #64748B;'>Lower score &nbsp;&nbsp; ➔ &nbsp;&nbsp; Higher score</div>", unsafe_allow_html=True)
        if selected_width or selected_layers or selected_model != "model_6":
            st.button("Clear Grid Selections", on_click=clear_filters)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container(border=True):
        if selected_width is not None:
            st.markdown(f'<div class="section-title">Column View: Models with Width {selected_width}</div>', unsafe_allow_html=True)
            col_models = [m for m, d in performance_data.items() if d["width"] == selected_width]
            col_models = sorted(col_models, key=lambda x: performance_data[x]["layers"])
            
            cols = st.columns(len(col_models))
            for idx, m_name in enumerate(col_models):
                m_data = performance_data[m_name]
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; text-align: center;">
                        <div style="font-size: 13px; font-weight: 700; color: #0F172A;">{m_name.upper()}</div>
                        <div style="font-size: 11px; color: #64748B; margin-bottom: 6px;">({m_data['layers']} Layers)</div>
                        <div style="font-size: 12px; font-weight: 600; color: #2563EB; margin-bottom: 4px;">Acc: {m_data['accuracy']:.4f}</div>
                        <div style="font-size: 12px; color: #475569;">F1: {m_data['f1_score']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Analyze {m_name.upper()}", key=f"btn_col_{m_name}"):
                        st.query_params.clear()
                        st.query_params["model"] = m_name
                        st.rerun()

        elif selected_layers is not None:
            st.markdown(f'<div class="section-title">Row View: Models with {selected_layers} Layers</div>', unsafe_allow_html=True)
            row_models = [m for m, d in performance_data.items() if d["layers"] == selected_layers]
            row_models = sorted(row_models, key=lambda x: performance_data[x]["width"])
            
            cols = st.columns(len(row_models))
            for idx, m_name in enumerate(row_models):
                m_data = performance_data[m_name]
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; text-align: center;">
                        <div style="font-size: 13px; font-weight: 700; color: #0F172A;">{m_name.upper()}</div>
                        <div style="font-size: 11px; color: #64748B; margin-bottom: 6px;">({m_data['width']} Width)</div>
                        <div style="font-size: 12px; font-weight: 600; color: #2563EB; margin-bottom: 4px;">Acc: {m_data['accuracy']:.4f}</div>
                        <div style="font-size: 12px; color: #475569;">F1: {m_data['f1_score']:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Analyze {m_name.upper()}", key=f"btn_row_{m_name}"):
                        st.query_params.clear()
                        st.query_params["model"] = m_name
                        st.rerun()

        else:
            m_data = performance_data[selected_model]
            st.markdown(f'<div class="section-title">Model Profile: {selected_model.upper()}</div>', unsafe_allow_html=True)
            
            def calc_total_params(layer_sizes):
                in_s = 17
                total_params = 0
                for out_s in layer_sizes:
                    total_params += (in_s * out_s) + out_s
                    in_s = out_s
                total_params += (in_s * 2) + 2
                return total_params

            tot_params = calc_total_params(m_data['layer_sizes'])

            st.markdown(f"""
            <div style="margin-bottom: 16px;">
                <span class="stat-badge">Layers: {m_data['layers']}</span>
                <span class="stat-badge">Latent Width: {m_data['width']} Neurons</span>
                <span class="stat-badge">Trainable Parameters: {tot_params:,}</span>
                <span class="stat-badge">Epochs Trained: {m_data['epoch'] or 'N/A'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            det_cols = st.columns(4)
            with det_cols[0]:
                st.metric("Test Accuracy", f"{m_data['accuracy']:.4%}" if m_data['accuracy'] else "N/A")
            with det_cols[1]:
                st.metric("F1-Score", f"{m_data['f1_score']:.4f}" if m_data['f1_score'] else "N/A")
            with det_cols[2]:
                st.metric("AUC-ROC", f"{m_data['auc_roc']:.4f}" if m_data['auc_roc'] else "N/A")
            with det_cols[3]:
                st.metric("Latency / Sample", f"{m_data['avg_time_per_sample_ms']:.5f} ms" if m_data['avg_time_per_sample_ms'] else "N/A")

            train_l = m_data.get("train_loss")
            valid_l = m_data.get("valid_loss")
            if train_l or valid_l:
                l_col1, l_col2 = st.columns(2)
                with l_col1:
                    st.markdown(f"**Final Training Loss**: `{train_l:.6f}`" if train_l else "")
                with l_col2:
                    st.markdown(f"**Final Validation Loss**: `{valid_l:.6f}`" if valid_l else "")
                    
            st.markdown("---")
            
            if selected_model == "model_8":
                tab_layers, tab_cm, tab_balancing = st.tabs([
                    "PyTorch Layer Configuration", 
                    "Confusion Matrix", 
                    "Class Balancing Analysis"
                ])
            else:
                tab_layers, tab_cm = st.tabs([
                    "PyTorch Layer Configuration", 
                    "Confusion Matrix"
                ])
                
            with tab_layers:
                def generate_layers_html(layer_sizes):
                    html = []
                    html.append('<table style="width:100%; border-collapse: collapse; font-family: \'Inter\'; font-size: 14px; text-align: left;">')
                    html.append('  <thead>')
                    html.append('    <tr style="border-bottom: 2px solid #E2E8F0; font-weight: 700; color: #0F172A; background-color: #F8FAFC;">')
                    html.append('      <th style="padding: 10px;">Layer Index</th>')
                    html.append('      <th style="padding: 10px;">Layer Type</th>')
                    html.append('      <th style="padding: 10px;">Input Features</th>')
                    html.append('      <th style="padding: 10px;">Output Neurons</th>')
                    html.append('      <th style="padding: 10px;">Activation & Regularization</th>')
                    html.append('    </tr>')
                    html.append('  </thead>')
                    html.append('  <tbody>')
                    
                    in_s = 17
                    for idx, out_s in enumerate(layer_sizes):
                        html.append('    <tr style="border-bottom: 1px solid #F1F5F9;">')
                        html.append(f'      <td style="padding: 10px; font-weight: 600;">Hidden Layer {idx+1}</td>')
                        html.append('      <td style="padding: 10px;">Fully Connected (Linear)</td>')
                        html.append(f'      <td style="padding: 10px;">{in_s}</td>')
                        html.append(f'      <td style="padding: 10px; font-weight: 700; color: #2563EB;">{out_s}</td>')
                        html.append('      <td style="padding: 10px;">ReLU, Dropout (p=0.2)</td>')
                        html.append('    </tr>')
                        in_s = out_s
                        
                    html.append('    <tr style="border-bottom: 1px solid #F1F5F9;">')
                    html.append('      <td style="padding: 10px; font-weight: 600;">Output Layer</td>')
                    html.append('      <td style="padding: 10px;">Fully Connected (Linear)</td>')
                    html.append(f'      <td style="padding: 10px;">{in_s}</td>')
                    html.append('      <td style="padding: 10px; font-weight: 700; color: #16A34A;">2</td>')
                    html.append('      <td style="padding: 10px;">Softmax probabilities (Safe vs Collision)</td>')
                    html.append('    </tr>')
                    html.append('  </tbody>')
                    html.append('</table>')
                    return "\n".join(html)
                
                st.write(generate_layers_html(m_data['layer_sizes']), unsafe_allow_html=True)
                
            with tab_cm:
                cm = m_data["confusion_matrix"]
                if cm:
                    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                    st.markdown(f"""
                    <table style="width:100%; text-align:center; font-family:'Inter'; border-collapse: collapse;">
                        <tr>
                            <td style="border:none;"></td>
                            <td colspan="2" style="font-weight:700; color:#0F172A; padding:6px; background-color:#F1F5F9; border-radius:6px 6px 0 0;">Predicted Class</td>
                        </tr>
                        <tr style="border-bottom:1px solid #E2E8F0;">
                            <td style="font-weight:700; color:#0F172A; padding:6px; background-color:#F1F5F9; border-radius:6px 0 0 6px;">Actual Class</td>
                            <td style="font-weight:600; padding:12px; color:#475569; background-color:#F8FAFC;">Safe (0)</td>
                            <td style="font-weight:600; padding:12px; color:#475569; background-color:#F8FAFC;">Collision (1)</td>
                        </tr>
                        <tr style="border-bottom:1px solid #E2E8F0;">
                            <td style="font-weight:600; padding:12px; color:#475569; background-color:#F8FAFC;">Safe (0)</td>
                            <td style="background-color:#DCFCE7; color:#14532D; font-weight:700; padding:16px;">True Neg: {tn:,}</td>
                            <td style="background-color:#FEE2E2; color:#7F1D1D; font-weight:700; padding:16px;">False Pos: {fp:,}</td>
                        </tr>
                        <tr>
                            <td style="font-weight:600; padding:12px; color:#475569; background-color:#F8FAFC; border-radius:0 0 0 6px;">Collision (1)</td>
                            <td style="background-color:#FEE2E2; color:#7F1D1D; font-weight:700; padding:16px;">False Neg: {fn:,}</td>
                            <td style="background-color:#DCFCE7; color:#14532D; font-weight:700; padding:16px;">True Pos: {tp:,}</td>
                        </tr>
                    </table>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Confusion matrix not evaluated for this model configuration.")

            if selected_model == "model_8":
                with tab_balancing:
                    st.markdown("##### Class Balancing Impact Study")
                    st.markdown(
                        "This analysis compares Model 8 trained under two configurations: "
                        "with class balancing (**[1.0, 1.5] weights**) and without class balancing (**[1.0, 1.0] weights**)."
                    )
                    
                    balancing_html = """
                    <table style="width:100%; border-collapse: collapse; font-family: 'Inter'; text-align: left; font-size: 14px; margin-top: 10px;">
                        <thead>
                            <tr style="border-bottom: 2px solid #CBD5E1; font-weight:700; color:#0F172A; background-color:#F8FAFC;">
                                <th style="padding: 10px;">Metric</th>
                                <th style="padding: 10px;">Class-Balanced (Model 8)<br><span style="font-size: 11px; font-weight: normal; color: #64748B;">weights = [1.0, 1.5]</span></th>
                                <th style="padding: 10px;">Unweighted (No Balancing)<br><span style="font-size: 11px; font-weight: normal; color: #64748B;">weights = [1.0, 1.0]</span></th>
                                <th style="padding: 10px;">Difference / Impact</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight:600;">Accuracy</td>
                                <td style="padding: 10px;">92.794%</td>
                                <td style="padding: 10px; font-weight: 600; color: #16A34A;">93.320%</td>
                                <td style="padding: 10px; font-weight:700; color:#16A34A;">+0.526% (Unweighted)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight:600;">Precision</td>
                                <td style="padding: 10px;">0.8946</td>
                                <td style="padding: 10px; font-weight: 600; color: #16A34A;">0.9249</td>
                                <td style="padding: 10px; font-weight:700; color:#16A34A;">+0.0303 (Unweighted)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight:600;">Recall (Sensitivity)</td>
                                <td style="padding: 10px; font-weight: 600; color: #16A34A;">0.9289</td>
                                <td style="padding: 10px;">0.9061</td>
                                <td style="padding: 10px; font-weight:700; color:#16A34A;">+0.0228 (Weighted)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight:600;">F1-Score</td>
                                <td style="padding: 10px;">0.9114</td>
                                <td style="padding: 10px; font-weight: 600; color: #16A34A;">0.9154</td>
                                <td style="padding: 10px; font-weight:700; color:#16A34A;">+0.0040 (Unweighted)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight:600;">AUC-ROC</td>
                                <td style="padding: 10px;">0.9836</td>
                                <td style="padding: 10px; font-weight: 600; color: #16A34A;">0.9843</td>
                                <td style="padding: 10px; font-weight:700; color:#16A34A;">+0.0007 (Unweighted)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight:600;">Final Losses (Train / Valid)</td>
                                <td style="padding: 10px;">0.1816 / 0.1663</td>
                                <td style="padding: 10px;">0.1720 / 0.1591</td>
                                <td style="padding: 10px; color: #475569;">Lower loss on Unweighted</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0;">
                                <td style="padding: 10px; font-weight:600;">Epochs Trained</td>
                                <td style="padding: 10px;">87</td>
                                <td style="padding: 10px;">122</td>
                                <td style="padding: 10px; color: #475569;">+35 Epochs</td>
                            </tr>
                        </tbody>
                    </table>
                    """
                    st.write(balancing_html, unsafe_allow_html=True)
                    
                    st.markdown(
                        "**Key Takeaways:**\\n"
                        "- **Recall Improvement**: Training with a positive class weight multiplier of `1.5` successfully biases the model towards catching self-collisions, increasing **Recall** from `90.61%` to `92.89%` (a **+2.28%** improvement).\\n"
                        "- **Precision Trade-off**: This safety-first bias causes a drop in **Precision** (from `92.49%` to `89.46%`) due to a higher rate of false-positive warnings. In robotic self-collision avoidance, this is highly desirable, as false negatives (undetected collisions) cause physical hardware damage."
                    )

# ----------------------------------------------------
# COLUMN 2: Live Collision Predictor & XGBoost comparison
# ----------------------------------------------------
with col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Live Collision Predictor</div>', unsafe_allow_html=True)
        
        available_models = list(performance_data.keys())
        default_idx = available_models.index(selected_model) if selected_model in available_models else 5
        
        predictor_model_name = st.selectbox(
            "Select PyTorch Model to predict with:",
            options=available_models,
            index=default_idx,
            format_func=lambda x: f"{x.upper()} ({' ➔ '.join(map(str, performance_data[x]['layer_sizes']))})"
        )
        
        # Preset Poses (Quick Loading buttons)
        st.markdown("**Load Preset Pose Configuration:**")
        preset_cols = st.columns(4)
        
        with preset_cols[0]:
            st.button("Default Pose", on_click=load_preset_pose, args=("Default Home Pose",))
        with preset_cols[1]:
            st.button("Verified Safe", on_click=load_preset_pose, args=("Safe Configuration",))
        with preset_cols[2]:
            st.button("Collision State", on_click=load_preset_pose, args=("Collision Configuration",))
        with preset_cols[3]:
            st.button("Randomize", on_click=randomize_pose)

        # Display units radio
        units = st.radio(
            "Display Sliders in:", 
            ["Radians", "Degrees"], 
            horizontal=True, 
            key="units_radio",
            on_change=convert_units
        )
        
        # Sync the display unit state tracker
        st.session_state.prev_unit = units
        
        joint_names = [
            *[f"Left Arm Joint {i}" for i in range(1, 8)],
            *[f"Right Arm Joint {i}" for i in range(1, 8)],
            "Torso Lift Joint (meters)",
            "Head Joint 1",
            "Head Joint 2"
        ]
        
        st.markdown("---")
        slider_inputs = []
        
        tab_left, tab_right, tab_other = st.tabs(["Left Arm Joints", "Right Arm Joints", "Torso and Head"])
        
        with tab_left:
            for idx in range(7):
                lo_rad, hi_rad = joint_limits_rad[idx]
                
                # Bounds dependent on current display unit
                if units == "Degrees":
                    lo_disp = float(np.degrees(lo_rad))
                    hi_disp = float(np.degrees(hi_rad))
                else:
                    lo_disp = float(lo_rad)
                    hi_disp = float(hi_rad)
                    
                # Bind directly to st.session_state key (e.g. s_0) to allow external modification
                st.session_state[f"s_{idx}"] = float(np.clip(st.session_state[f"s_{idx}"], lo_disp, hi_disp))
                slider_val = st.slider(joint_names[idx], min_value=lo_disp, max_value=hi_disp, key=f"s_{idx}")
                
        with tab_right:
            for idx in range(7, 14):
                lo_rad, hi_rad = joint_limits_rad[idx]
                
                if units == "Degrees":
                    lo_disp = float(np.degrees(lo_rad))
                    hi_disp = float(np.degrees(hi_rad))
                else:
                    lo_disp = float(lo_rad)
                    hi_disp = float(hi_rad)
                    
                st.session_state[f"s_{idx}"] = float(np.clip(st.session_state[f"s_{idx}"], lo_disp, hi_disp))
                slider_val = st.slider(joint_names[idx], min_value=lo_disp, max_value=hi_disp, key=f"s_{idx}")
                
        with tab_other:
            # Torso Lift (always meters)
            lo_rad, hi_rad = joint_limits_rad[14]
            st.session_state["s_14"] = float(np.clip(st.session_state["s_14"], lo_rad, hi_rad))
            slider_val = st.slider(joint_names[14], min_value=lo_rad, max_value=hi_rad, key="s_14")
            
            # Head joints
            for idx in [15, 16]:
                lo_rad, hi_rad = joint_limits_rad[idx]
                
                if units == "Degrees":
                    lo_disp = float(np.degrees(lo_rad))
                    hi_disp = float(np.degrees(hi_rad))
                else:
                    lo_disp = float(lo_rad)
                    hi_disp = float(hi_rad)
                    
                st.session_state[f"s_{idx}"] = float(np.clip(st.session_state[f"s_{idx}"], lo_disp, hi_disp))
                slider_val = st.slider(joint_names[idx], min_value=lo_disp, max_value=hi_disp, key=f"s_{idx}")

        # Construct raw joint positions in Radians from session state keys for inference
        raw_positions = []
        for idx in range(17):
            val = st.session_state[f"s_{idx}"]
            if idx == 14: # Torso is always meters
                raw_positions.append(val)
            else:
                if units == "Degrees":
                    raw_positions.append(float(np.radians(val)))
                else:
                    raw_positions.append(float(val))

        pos_array = np.array(raw_positions, dtype=np.float32).reshape(1, -1)
        scaled_pos = scaler.transform(pos_array)
        
        # 1. PyTorch NN Model Prediction
        m_info = performance_data[predictor_model_name]
        nn_model = load_pytorch_model(predictor_model_name, m_info["layer_sizes"])
        
        start_nn = time.perf_counter()
        with torch.no_grad():
            nn_out = nn_model(torch.tensor(scaled_pos, dtype=torch.float32))
            nn_prob = F.softmax(nn_out, dim=1).numpy()[0]
            nn_pred = np.argmax(nn_prob)
        t_nn_ms = (time.perf_counter() - start_nn) * 1000
        
        # 2. XGBoost Prediction
        start_xgb = time.perf_counter()
        xgb_prob = xgb_model.predict_proba(pos_array)[0]
        xgb_pred = np.argmax(xgb_prob)
        t_xgb_ms = (time.perf_counter() - start_xgb) * 1000
        
        st.markdown("---")
        st.markdown("#### Prediction Output")
        
        r_col1, r_col2 = st.columns(2)
        
        with r_col1:
            st.markdown(f"**PyTorch Model: {predictor_model_name.upper()}**")
            if nn_pred == 1:
                st.markdown('<div class="alert-collision">COLLISION DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-safe">SAFE (NO COLLISION)</div>', unsafe_allow_html=True)
            
            prob_percent = nn_prob[nn_pred] * 100
            st.write(f"Confidence: **{prob_percent:.2f}%**")
            st.write(f"Latency: **{t_nn_ms:.3f} ms**")
            
        with r_col2:
            st.markdown("**XGBoost Classifier Baseline**")
            if xgb_pred == 1:
                st.markdown('<div class="alert-collision">COLLISION DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-safe">SAFE (NO COLLISION)</div>', unsafe_allow_html=True)
                
            xgb_prob_percent = xgb_prob[xgb_pred] * 100
            st.write(f"Confidence: **{xgb_prob_percent:.2f}%**")
            st.write(f"Latency: **{t_xgb_ms:.3f} ms**")

    with st.container(border=True):
        st.markdown('<div class="section-title">Neural Network vs XGBoost Comparison</div>', unsafe_allow_html=True)
        
        comp_model_name = st.selectbox(
            "Select Model to compare against XGBoost metrics:",
            options=available_models,
            index=default_idx,
            key="compare_dropdown"
        )
        
        nn_comp_data = performance_data[comp_model_name]
        
        metrics_comparison_table = f"""
        <table style="width:100%; border-collapse: collapse; font-family: 'Inter'; text-align: left; font-size: 14px;">
            <thead>
                <tr style="border-bottom: 2px solid #CBD5E1; font-weight:700; color:#0F172A; background-color:#F8FAFC;">
                    <th style="padding: 10px;">Metric</th>
                    <th style="padding: 10px;">{comp_model_name.upper()} (NN)</th>
                    <th style="padding: 10px;">XGBoost Baseline</th>
                    <th style="padding: 10px;">Winner / Diff</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 10px; font-weight:600;">Accuracy</td>
                    <td style="padding: 10px;">{nn_comp_data['accuracy']:.4%}</td>
                    <td style="padding: 10px;">{XGB_PERFORMANCE['accuracy']:.4%}</td>
                    <td style="padding: 10px; font-weight:700; color:#16A34A;">+{(nn_comp_data['accuracy'] - XGB_PERFORMANCE['accuracy'])*100:.2f}% (NN)</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 10px; font-weight:600;">Precision</td>
                    <td style="padding: 10px;">{nn_comp_data['precision']:.4f}</td>
                    <td style="padding: 10px;">{XGB_PERFORMANCE['precision']:.4f}</td>
                    <td style="padding: 10px; font-weight:700; color:{'#16A34A' if nn_comp_data['precision'] > XGB_PERFORMANCE['precision'] else '#DC2626'}">
                        {'+' if nn_comp_data['precision'] > XGB_PERFORMANCE['precision'] else ''}{nn_comp_data['precision'] - XGB_PERFORMANCE['precision']:.4f} 
                        ({'NN' if nn_comp_data['precision'] > XGB_PERFORMANCE['precision'] else 'XGB'})
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 10px; font-weight:600;">Recall</td>
                    <td style="padding: 10px;">{nn_comp_data['recall']:.4f}</td>
                    <td style="padding: 10px;">{XGB_PERFORMANCE['recall']:.4f}</td>
                    <td style="padding: 10px; font-weight:700; color:#16A34A;">+{nn_comp_data['recall'] - XGB_PERFORMANCE['recall']:.4f} (NN)</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 10px; font-weight:600;">F1-Score</td>
                    <td style="padding: 10px;">{nn_comp_data['f1_score']:.4f}</td>
                    <td style="padding: 10px;">{XGB_PERFORMANCE['f1_score']:.4f}</td>
                    <td style="padding: 10px; font-weight:700; color:#16A34A;">+{nn_comp_data['f1_score'] - XGB_PERFORMANCE['f1_score']:.4f} (NN)</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 10px; font-weight:600;">AUC-ROC</td>
                    <td style="padding: 10px;">{nn_comp_data['auc_roc']:.4f}</td>
                    <td style="padding: 10px;">{XGB_PERFORMANCE['auc_roc']:.4f}</td>
                    <td style="padding: 10px; font-weight:700; color:#16A34A;">+{nn_comp_data['auc_roc'] - XGB_PERFORMANCE['auc_roc']:.4f} (NN)</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 10px; font-weight:600;">Inference Speed</td>
                    <td style="padding: 10px;">{nn_comp_data['samples_per_second']:,.0f} samples/s</td>
                    <td style="padding: 10px;">{XGB_PERFORMANCE['samples_per_second']:,.0f} samples/s</td>
                    <td style="padding: 10px; font-weight:700; color:#DC2626;">+{XGB_PERFORMANCE['samples_per_second']/nn_comp_data['samples_per_second']:.1f}x Faster (XGB)</td>
                </tr>
            </tbody>
        </table>
        """
        
        st.write(metrics_comparison_table, unsafe_allow_html=True)
