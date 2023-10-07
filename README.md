# Red Teaming Language Models with Language Models

A re-implementation of the [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286) paper by Perez et al. 

This implementation only focuses on toxic/offensive langauge section of the paper.

I have based run the red-teaming experiments on four target models -   
* [GPT2-XL](https://huggingface.co/gpt2-xl)
* [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
* [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b)
* [Phi-1.5](https://huggingface.co/microsoft/phi-1_5.)

I generated 50,000 valid test cases (questions) as opposed to 500,000 mentioned in the paper, due to compute constraints. A small percantge of the test cases were succesful in being able to elicit toxic language behavior from these models.

For Toxicity detection, I have used [Detoxify](https://github.com/unitaryai/detoxify).

The percentage of toxic/offensive answers (toxicity probability > 0.5) generated for questions generated using Zero-Shot and Stochastic Few-Shot generation for each model are presented below - 

| Model        | Zero-Shot | Stochastic Few-Shot | Increase (Few-Shot vs Zero-Shot) |
|--------------|-----------|---------------------|----------------------------------|
| GPT2-XL-1.5B | 0.99%     | 2.85%               | 2.88x                            |
| Llama-2-7B   | 0.15%     | 0.33%               | 2.20x                            |
| Pythia-6.9B  | 0.47%     | 1.02%               | 2.17x                            |
| Phi-1.5B     | 0.11%     |                     |                                  |

The questions, answers and toxicity scores can be found in the [artifacts](artifacts/) directory.