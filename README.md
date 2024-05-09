# LLM
A package demonstrating model implementation and deployment for solving practical problems. 

## Getting Started

### Get the package

```
# Navigate to your local folder
cd /your/local/folder

# Clone the WindML repository
git clone git@github.com:marcodigennaro/LLM.git

# Enter the folder
cd LLM/

# Create the python environment from the pyproject.toml file
poetry install

# Activate the python environment
source .venv/bin/activate

# Run tests 
poetry run pytest -v

# Start Jupyter Lab
jupyter-lab  
```

### Content of the Jupyter Notebooks

1. `sentiment_analysis` shows three strategies for sentiment analysis on a database of amazon reviews 
   1. VADER (bag of words)
   2. RoBERTa (Transformer)
   3. Transformers Pipelines

2. `RNN` demonstrate how to use a basic RNN as LLM
   1. Reads a text (Prometheus's) legend and predict the continuation of the story
   2. Uses MDS, Isomaps and t-SNE representations to project the results onto a 2D space

![VADER vs RoBERTa](https://github.com/marcodigennaro/LLM/blob/main/llm/images/vader_vs_roberta.jpeg)
![Accuracy](https://github.com/marcodigennaro/LLM/blob/main/llm/images/confusion_matrix.jpeg)
![https://github.com/marcodigennaro/LLM/blob/main/llm/images/prometheo.jpeg]

### Author

- [Marco Di Gennaro](https://github.com/marcodigennaro/CV/blob/main/MDG_CV.pdf)
- [GitHub](https://github.com/marcodigennaro)
- [Linkedin](https://www.linkedin.com/in/marcodig/)
- [Website](https://atomistic-modelling.com/)

### License

This project is licensed under the GPL v3 License - see the [LICENSE.md](https://github.com/marcodigennaro/WindML/blob/main/LICENSE.md) file for details


### Acknowledgments

- [Aurélien Géron's Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Rob Mulla web series](https://www.youtube.com/watch?v=QpzMWQvxXWk&t=129s)
