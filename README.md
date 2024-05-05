# SpatialSense

## LLM Overview

### Transformers
Transformers are neural network architectures that revolutionized natural language processing (NLP). The seminal paper, "Attention Is All You Need" by Vaswani et al., introduced this architecture, which is distinguished by self-attention mechanisms. These mechanisms enable the model to process data in parallel and capture complex word relationships, boosting performance in machine translation, text generation, and understanding.

### Transformers Architecture
![Transformers Architecture](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/static/transformers.png)
### LLaMa

Meta AI's LLaMa (Large Language Model Meta AI) is a series of foundation language models ranging from 7 billion to 65 billion parameters.  LLaMa's architecture is based on the Transformer architecture, known for its efficiency in handling sequential data in natural language processing tasks.  These models are notable for comparable performance to larger counterparts while requiring significantly fewer parameters and computational resources.

### LLM Quantization

LLM quantization is a technique for compressing large language models to reduce their memory footprint and increase computational efficiency. The core idea is to represent the weights within the model using lower-precision data types instead of the standard 32-bit floating-point format. This can dramatically reduce model size without significantly sacrificing performance.

### 4-Bit Quantization

4-bit quantization is a specific type of quantization where model weights are represented using only 4 bits (half a byte). This offers a substantial compression advantage: in theory, a model quantized to 4 bits could be up to 8 times smaller than its 32-bit counterpart. However, 4-bit quantization requires careful implementation to avoid significant performance degradation.  Techniques include:

- Mixed-Precision: Storing a small number of the most important weights in higher precision while quantizing the majority to 4-bits.
- Quantization-Aware Training: Training the model with quantization in mind, minimizing accuracy loss when switching to lower precision.

## Code Structure
The Spatial Sense repository includes directories and files critical for the project's setup and execution.

- `app.py`: The Flask application entry point.
- `requirements.txt`: Lists all the Python libraries that the project depends on.
- `index.html`: Contains the front-end user interface for interacting with the Spatial Sense system.
- `static/`: Stores static files like CSS and JavaScript that enhance the front end.
- `templates/`: Contains HTML templates used by Flask to render the UI.
  
## Steps to Reproduce the Code and Run It
1. Clone the repository from [Github](https://github.com/kabir12345/SpatialSenseWeb).
2. Install dependencies listed in `requirements.txt` using `pip install -r requirements.txt`.
3. Run `app.py` to start the Flask server.
4. Open a browser and navigate to `http://localhost:5000/` to interact with the application.


## Results
The application processes real-time video feeds to generate depth maps and textual feedback, demonstrating the effectiveness of combining visual processing and NLP for indoor navigation.

## Key Files and Their Functions
- `app.py`: Initializes and runs the Flask web server, processing video data input.
- `requirements.txt`: Specifies the necessary libraries for ensuring environment parity.
- `index.html`: Hosts the user interface, allowing real-time video input and displaying results.

<img width="1502" alt="Screenshot 2024-05-05 at 11 46 30 AM" src="https://github.com/kabir12345/SpatialSenseWeb/assets/18241637/2a6dac8d-3fbf-437c-9956-6a031a2cc367">

## References

  Bhat, S. F., Birkl, R., Wofk, D., Wonka, P., & Müller, M. (2023). ZoeDepth: Zero-shot transfer by combining relative and metric depth. arXiv. https://arxiv.org/abs/2302.12288
  Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
  Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. In Proceedings of the Neural Information Processing Systems Conference (NeurIPS).
