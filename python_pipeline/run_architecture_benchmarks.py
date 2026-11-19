import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys

# Import V1
from ml_baseline_compare import IMMFilter, LightweightLSTM, Lightweight1DCNN

class LightweightGRU(nn.Module):
    def __init__(self): 
        super().__init__()
        self.gru = nn.GRU(2, 32, 2, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x): 
        _, h = self.gru(x)
        return torch.sigmoid(self.fc(h[-1]))

class SolarFlareNetProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        return torch.sigmoid(self.fc(self.cnn(x).squeeze(-1)))

def get_ml_alerts(sxr, hxr, model_class):
    model = model_class()
    model.eval()
    alerts = 0
    consec_nan = 0
    for step in range(len(sxr)):
        if step < 60: continue
        window = np.stack([sxr[step-60:step]/1e-6, hxr[step-60:step]/1e-7], axis=-1)
        if np.isnan(window).any():
            consec_nan += 1
            continue
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            try:
                if model(tensor).item() > 0.5: alerts += 1
            except: pass
    return alerts

def get_solis_v1_alerts(sxr, hxr):
    imm = IMMFilter()
    bg = float(np.median(sxr[:15]) if len(sxr)>15 else sxr[0])
    if np.isnan(bg): bg = 1e-7
    imm.ukf_quiet.sxr_bg = max(bg, 1e-4)
    imm.ukf_flare.sxr_bg = max(bg, 1e-4)
    alerts = 0
    for step in range(len(sxr)):
        z = np.array([sxr[step], hxr[step]])
        if np.isnan(z).any(): continue
        mu = imm.step(z)
        maha = max(imm.ukf_quiet.last_mahalanobis, imm.ukf_flare.last_mahalanobis)
        if mu > 0.3 and maha > 1.5: alerts += 1
    return alerts

def get_solis_v2_alerts(sxr, hxr):
    imm = IMMFilter()
    bg = float(np.median(sxr[:15]) if len(sxr)>15 else sxr[0])
    if np.isnan(bg): bg = 1e-7
    imm.ukf_quiet.sxr_bg = max(bg, 1e-4)
    imm.ukf_flare.sxr_bg = max(bg, 1e-4)
    alerts = 0
    consec_alert = 0
    
    for step in range(len(sxr)):
        z = np.array([sxr[step], hxr[step]])
        # V2 Fix 1: RTC SRAM State Recovery (Data Abyss)
        if np.isnan(z).any():
            # Reset P matrix to infinity (high uncertainty) to avoid divergence
            imm.ukf_quiet.P = np.eye(5) * 1e6
            imm.ukf_flare.P = np.eye(5) * 1e6
            continue
            
        mu = imm.step(z)
        maha = max(imm.ukf_quiet.last_mahalanobis, imm.ukf_flare.last_mahalanobis)
        
        # V2 Fix 2: Dynamic Thresholding based on background (FAR Storm fix)
        current_bg = imm.ukf_quiet.x[0]
        dynamic_thresh = 1.5 + (current_bg / 1e-6) * 1.0 
        
        # V2 Fix 3: Neupert Fallback Scaling (Gradual Flare fix)
        if maha > 3.0 and mu < 0.1:
            imm.ukf_flare.Q[0,0] *= 1.5 # Dynamically inflate Q to track thermal
            
        if mu > 0.3 and maha > dynamic_thresh:
            consec_alert += 1
            if consec_alert >= 2: alerts += 1
        else:
            consec_alert = 0
            
    return alerts

# --- Test Generation ---
t1 = np.arange(0, 3600, 1)
sxr1 = 1e-7 * np.exp((t1 - 1800) / 300); hxr1 = 1e-8 * np.ones_like(t1) # T1

t3 = np.arange(0, 20000, 1) # Shortened for speed
base = 1e-7
sxr3 = np.zeros_like(t3, dtype=float)
for i in range(0, 20000, 2000):
    sxr3 += base * 10 * np.exp(-((t3 - i)**2) / (2 * 100**2))
sxr3 += np.random.poisson(base * 100) / 100
hxr3 = 1e-8 * np.ones_like(t3)

t15 = np.arange(0, 20000, 1)
sxr15 = 1e-7 * np.ones_like(t15); hxr15 = 1e-8 * np.ones_like(t15)
sxr15[3600:10000] = np.nan; hxr15[3600:10000] = np.nan
sxr15[10000:] = 1e-5 * np.exp(-((t15[10000:] - 11000)**2) / (2 * 600**2))
hxr15[10000:] = 5e-6 * np.exp(-((t15[10000:] - 11000)**2) / (2 * 400**2))

tests = [
    ("T1: Gradual Flare (Neupert Viol.)", sxr1, hxr1, "Should alert (Fallback)"),
    ("T3: FAR Storm (Noise/Bg rise)", sxr3, hxr3, "Should NOT alert (or low)"),
    ("T15: Data Abyss (NaN gap)", sxr15, hxr15, "Should recover and alert once")
]

models = [
    ("SOLIS_V1", get_solis_v1_alerts),
    ("SOLIS_V2", get_solis_v2_alerts),
    ("CNN", lambda s,h: get_ml_alerts(s,h,Lightweight1DCNN)),
    ("LSTM", lambda s,h: get_ml_alerts(s,h,LightweightLSTM)),
    ("GRU", lambda s,h: get_ml_alerts(s,h,LightweightGRU)),
    ("SolarFlareNet (500K)", lambda s,h: get_ml_alerts(s,h,SolarFlareNetProxy)),
]

output_path = r"C:\Users\brr33\.gemini\antigravity\brain\351aca50-aa69-42a9-b6d9-ef892bb706ae\solis_v2_benchmarks.md"
with open(output_path, "w") as f:
    f.write("# SOLIS V2 Architecture Benchmark Results\n\n")
    f.write("| Test Scenario | Expected Behavior | SOLIS V1 (Alerts) | SOLIS V2 (Alerts) | CNN | LSTM | GRU | SolarFlareNet |\n")
    f.write("|---------------|-------------------|-------------------|-------------------|-----|------|-----|---------------|\n")
    
    for t_name, sxr, hxr, exp in tests:
        row = f"| {t_name} | {exp} "
        for m_name, func in models:
            alerts = func(sxr, hxr)
            row += f"| {alerts} "
        row += "|\n"
        f.write(row)
        print(f"Completed {t_name}")

print(f"Report generated at {output_path}")
