# 🧠 Practical Deep Learning Using PyTorch

A hands-on and evolving collection of **PyTorch notebooks** focused on understanding and implementing deep learning concepts from the ground up.  

The repository starts with the fundamentals — **tensors**, the core data structure in PyTorch — and will expand over time to include more advanced topics like autograd, neural networks, optimization, and model deployment.

---

## 📘 Current Notebook

### [`tensors-in-pytorch.ipynb`](./tensors-in-pytorch.ipynb)

An introductory **Google Colab** notebook exploring the core building block of PyTorch — **tensors**.  
It demonstrates how to create, manipulate, and perform mathematical operations on tensors, while also covering NumPy interoperability and GPU support.

#### 🧩 Topics Covered
- Import torch and GPU availability check
- Creating tensors using multiple methods  
- Inspecting tensor shapes and data types  
- Reshaping, cloning, and copying tensors  
- Scalar, element-wise, matrix, and reduction operations  
- Comparison and in-place operations  
- Conversion between **NumPy** and **PyTorch**  

---

## 🚀 Quick Example

```python
import torch
import numpy as np

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Create a tensor
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).to(device)

# NumPy ↔ PyTorch conversion
arr = t.cpu().numpy()
t2 = torch.from_numpy(arr)

# Operations
print(t + t2)       # Element-wise add
print(t.mean())     # Reduction op
print(t.reshape(4, 1))  # Reshape
```

## 🧭 Roadmap

This repository will gradually expand with more notebooks covering:

- **Autograd and gradient computation**  
- **Neural network layers and models**  
- **Loss functions and optimizers**  
- **Training pipelines and evaluation**  
- **GPU acceleration and performance tuning**  
- **Model saving, loading, and deployment**


💡 Goal
-------

To build a solid foundation in **deep learning and neural networks** by learning through **hands-on implementation, experimentation, and practical projects**.

🤝 Contributions
----------------

This is primarily a learning repository, but suggestions and discussions are always welcome!

📫 Contact
----------

If you’d like to connect or collaborate, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/shafiqshams) or open an issue/discussion.

> _"The best way to learn is by doing — and breaking things along the way."_ 🚧
