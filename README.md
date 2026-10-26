# LLM Positional Encodings Lab: Sinusoidal Waves to Modern RoPE

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Pages Ready](https://img.shields.io/badge/GitHub_Pages-Ready-brightgreen.svg)](#-how-to-host-on-github-pages)
[![Tests Passing](https://img.shields.io/badge/Tests-14%20Passed-emerald.svg)](#-local-testing--verification)

An interactive, zero-backend, client-side web application designed to be hosted directly on **GitHub Pages**. 
It visually and mathematically unpacks how **Positional Encodings** allow Large Language Models to understand sequential word order, directly inspired by:
- 📄 **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Ashish Vaswani et al., 2017 - *Original Sinusoidal Encodings*)
- 🔬 **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)** (Jianlin Su et al., 2021 - *RoPE*)
- 🚀 **[The LLaMA 3 Architecture & Long-Context Scaling](https://ai.meta.com/blog/meta-llama-3/)** (Meta AI, 2024 - *128k Context Extrapolation via RoPE Base Frequency Scaling*)

---

## 🌐 Live GitHub Pages Demo
- **Live Interactive Demo:** [https://martianunlimited.github.io/llm_positional/](https://martianunlimited.github.io/llm_positional/)
- **Companion Tokenizer Demo:** [https://martianunlimited.github.io/llm_tokenizer/](https://martianunlimited.github.io/llm_tokenizer/)
- **Companion Embeddings Demo:** [https://martianunlimited.github.io/llm_embedding/](https://martianunlimited.github.io/llm_embedding/)
- **Companion Attention Demo:** [https://martianunlimited.github.io/llm_attention/](https://martianunlimited.github.io/llm_attention/)
- **Companion Sampling Demo:** [https://martianunlimited.github.io/llm_sampling/](https://martianunlimited.github.io/llm_sampling/)
- **GitHub Repository:** [https://github.com/martianunlimited/llm_positional](https://github.com/martianunlimited/llm_positional)

---

## 🧭 Interactive Modules in the Demo

### 1. The "Bag-of-Words" Permutation Dilemma
* Compares *"Dog bites man"* vs *"Man bites dog"*.
* Demonstrates that without positional encodings, self-attention matrices are mathematically symmetric and permutation-invariant.
* Turning Positional Encoding **ON** breaks symmetry and encodes grammatical syntax.

### 2. The Vaswani Sinusoidal Clock (2017)
* Continuous multi-frequency harmonic waves across dimensions:
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
* **The Clock Hands Metaphor:**
  - Dimension 0 (Seconds Hand): Period $\approx 6.28$ tokens (tracks immediate adjacent grammar).
  - Dimension 14 (Hours Hand): Period $\approx 62,800$ tokens (tracks long document structure).
* Interactive 2D Heatmap & dynamic wave visualizer for any dimension.

### 3. Modern RoPE: 2D Complex Unit Circle Rotations
* Used in **LLaMA 1/2/3**, **Mistral**, **Gemma**, and **Qwen**:
  $$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$
* Preserves vector lengths ($\|R_m \vec{q}\| = \|\vec{q}\|$).
* **Relative Distance Property:** Inner product $\langle R_m \vec{q}, R_n \vec{k} \rangle = \vec{q}^T R_{m-n} \vec{k}$ depends strictly on relative distance $\Delta = m - n$, producing natural recency bias!

### 4. LLaMA 3 128k Long-Context Scaling
* Compares standard $\theta_{\text{base}} = 10,000$ (4k context) vs $\theta_{\text{base}} = 500,000$ (LLaMA 3, 128k context).
* Visualizes angle compression: 50x slower rotation prevents out-of-distribution phase wrapping.

---

## 🚀 How to Host on GitHub Pages

This project is 100% static (HTML, CSS, JS) and requires **no build step, no Node server, and no Python backend**:

```bash
# 1. Initialize git and commit
git init
git add .
git commit -m "Initial commit: LLM Positional Encodings Lab"

# 2. Push to your GitHub repository
git branch -M main
git remote add origin https://github.com/martianunlimited/llm_positional.git
git push -u origin main

# 3. Enable GitHub Pages in your repo:
# Settings -> Pages -> Source: Deploy from a branch -> Branch: main -> / (root) -> Save!
# Your live site is now ready at: https://martianunlimited.github.io/llm_positional/
```

---

## 🧪 Local Testing & Verification

### 1. Run Automated Tests
```bash
python -m pytest -q
# 14 passed in 0.20s

python -m ruff check . --fix
python -m ruff format .
```

### 2. Terminal CLI Explorer
```bash
# Inspect the Bag-of-Words permutation dilemma:
python -m src.cli permutation

# View the sinusoidal positional encoding matrix:
python -m src.cli sinusoidal --pos 8 --dim 8

# Inspect 2D RoPE rotation angles across planes:
python -m src.cli rope --pos 3 --dim 8
```

### 3. Preview Locally
```bash
python -m http.server 8080
# Open http://localhost:8080 in your browser
```

---

## 📄 License
MIT License. Free for educational, academic, and open-source use.
