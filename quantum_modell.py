# quantum_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
import sys, os
import pennylane as qml

import numpy as np

# Define your drug and protein character sets.
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

def label_token(line, vocab, max_len):
    tokens = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_len]):
        tokens[i] = vocab.get(ch, 0)
    return tokens

# --- Quantum Model Setup ---

n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

@qml.transforms.merge_rotations
@qml.qnode(dev, interface="torch", diff_method="backprop", batch=True)
def quantum_circuit(params, inputs):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RY(params[layer, i], wires=i)
            qml.RZ(params[layer, i + n_qubits], wires=i)
            qml.RX(params[layer, i + 2 * n_qubits], wires=i)
        for i in range(n_qubits):
            qml.S(wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.CZ(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        qml.CZ(wires=[n_qubits - 1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"params": (n_layers, 3 * n_qubits)}
qlayer1 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
qlayer2 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
qlayer3 = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class QuantumFeatureExtractor(nn.Module):
    def __init__(self, n_qubits, quantum_hidden_dim, dropout_prob=0.3):
        super(QuantumFeatureExtractor, self).__init__()
        self.qlayer1 = qlayer1
        self.qlayer2 = qlayer2
        self.qlayer3 = qlayer3
        self.fc = nn.Linear(3 * n_qubits, quantum_hidden_dim)
        self.bn = nn.BatchNorm1d(quantum_hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        q_out1 = self.qlayer1(x)
        q_out2 = self.qlayer2(x)
        q_out3 = self.qlayer3(x)
        q_out = torch.cat((q_out1, q_out2, q_out3), dim=1)
        q_feat = self.fc(q_out)
        q_feat = self.bn(q_feat)
        q_feat = F.relu(q_feat)
        q_feat = self.dropout(q_feat)
        return q_feat

class QuantumDrugProteinModel(nn.Module):
    def __init__(self, drug_vocab_size, protein_vocab_size, embed_dim=128, quantum_hidden_dim=64, final_hidden_dim=128, dropout_prob=0.2, normalize_targets=True):
        super(QuantumDrugProteinModel, self).__init__()
        self.drug_embedding = nn.Embedding(drug_vocab_size + 1, embed_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size + 1, embed_dim, padding_idx=0)
        self.drug_pre_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU()
        )
        self.protein_pre_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU()
        )
        self.drug_fc = nn.Linear(embed_dim // 2, 2**n_qubits)
        self.protein_fc = nn.Linear(embed_dim // 2, 2**n_qubits)
        self.drug_quantum = QuantumFeatureExtractor(n_qubits, quantum_hidden_dim, dropout_prob)
        self.protein_quantum = QuantumFeatureExtractor(n_qubits, quantum_hidden_dim, dropout_prob)
        self.fc_final = nn.Sequential(
            nn.Linear(2 * quantum_hidden_dim, final_hidden_dim),
            nn.BatchNorm1d(final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(final_hidden_dim, final_hidden_dim // 2),
            nn.BatchNorm1d(final_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(final_hidden_dim // 2, 1)
        )
        self.normalize_targets = normalize_targets

    def forward(self, drug, protein):
        drug_emb = self.drug_embedding(drug).mean(dim=1)
        drug_emb = self.drug_pre_fc(drug_emb)
        drug_q_input = self.drug_fc(drug_emb)
        drug_features = self.drug_quantum(drug_q_input)
        protein_emb = self.protein_embedding(protein).mean(dim=1)
        protein_emb = self.protein_pre_fc(protein_emb)
        protein_q_input = self.protein_fc(protein_emb)
        protein_features = self.protein_quantum(protein_q_input)
        combined_features = torch.cat([drug_features, protein_features], dim=1)
        out = self.fc_final(combined_features)
        if self.normalize_targets:
            out = torch.tanh(out)
            out = (out + 1) / 2 * (16 - 1) + 1
        return out

# --- Load the Pretrained Model ---
drug_vocab_size = len(CHARISOSMISET) + 1
protein_vocab_size = len(CHARPROTSET)  # Assumes checkpoint used this size
model = QuantumDrugProteinModel(drug_vocab_size, protein_vocab_size)
state_dict = torch.load(r"D:\final_conversion\final_conversion\qbap_cnn_v16_81.25.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
device = next(model.parameters()).device
model.to(device)

def predict_binding_affinities(drug_smiles_list, protein_seq):
    # Tokenize protein sequence
    protein_tokens = label_token(protein_seq, CHARPROTSET, 1200)
    protein_tensor = torch.tensor(protein_tokens, dtype=torch.long).unsqueeze(0).to(device)
    results = []
    for smile in drug_smiles_list:
        drug_tokens = label_token(smile, CHARISOSMISET, 100)
        drug_tensor = torch.tensor(drug_tokens, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            affinity = model(drug_tensor, protein_tensor).squeeze().item()
        results.append({"drug": smile, "predicted_affinity": affinity})
    sorted_results = sorted(results, key=lambda x: x["predicted_affinity"], reverse=True)
    return sorted_results
