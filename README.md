# SpatialSense

## LLM Overview

### Transformers
Transformers are neural network architectures that revolutionized natural language processing (NLP). The seminal paper, "Attention Is All You Need" by Vaswani et al., introduced this architecture, which is distinguished by self-attention mechanisms. These mechanisms enable the model to process data in parallel and capture complex word relationships, boosting performance in machine translation, text generation, and understanding.

### Transformers Architecture
![Transformers Architecture](https://github.com/kabir12345/LLM-PEFT-Optimization/blob/main/static/transformers.png)
#### GPT-2
OpenAI's GPT-2 is a transformer-based model with a layered architecture comprising 12 to 48 transformer blocks, with parameters ranging from 117 million

to 1.5 billion, depending on the variant. Its multi-headed self-attention and fully connected layers generate coherent and contextually relevant text, facilitating various NLP tasks.

#### T5-Small
Google's T5, or Text-to-Text Transfer Transformer, frames NLP tasks as text-to-text problems. The T5-Small variant, with 60 million parameters and 6 layers in its encoder-decoder structure, efficiently performs tasks like translation, summarization, and question-answering.

#### LORA
Low-Rank Adaptation (LORA) is a technique to fine-tune large language models efficiently, by introducing a small set of additional parameters to approximate changes to the original weights, reducing trainable parameters and computational resources.

#### IA3
Infused Adapter by Inhibiting and Amplifying Inner Activations (IA3) optimizes language models by selectively inhibiting or amplifying activations within layers for better performance and computational efficiency.

## Code Structure
The Spatial Sense repository includes directories and files critical for the project's setup and execution.

- `app.py`: The Flask application entry point.
- `requirements.txt`: Lists all the Python libraries that the project depends on.
- `index.html`: Contains the front-end user interface for interacting with the Spatial Sense system.
- `static/`: Stores static files like CSS and JavaScript that enhance the front-end.
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

(Insert project page screenshot here)

For a complete guide and detailed instructions, refer to the `README.md` file in the repository. 

---

Note: This is a skeleton of the README.md file tailored to the information provided. Further details like installation instructions, usage examples, and contribution guidelines should be added based on the actual content and requirements of the project.