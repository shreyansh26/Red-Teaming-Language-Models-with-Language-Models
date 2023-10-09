# Red Teaming Language Models with Language Models

A re-implementation of the [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286) paper by Perez et al. 

This implementation only focuses on toxic/offensive langauge section of the paper.

I have based run the red-teaming experiments on four target models -   
* [GPT2-XL](https://huggingface.co/gpt2-xl)
* [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
* [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b)
* [Phi-1.5](https://huggingface.co/microsoft/phi-1_5.)

I generated 50,000 valid test cases (questions) as opposed to 500,000 mentioned in the paper, due to compute constraints. A small percantge of the test cases were succesful in being able to elicit toxic language generation from these models.

For Toxicity detection, I have used [Detoxify](https://github.com/unitaryai/detoxify).

The percentage of toxic/offensive *answers* (toxicity probability > 0.5) generated using Zero-Shot and Stochastic Few-Shot generation for each model are presented below - 

| Model        | Zero-Shot | Stochastic Few-Shot | Increase (Few-Shot vs Zero-Shot) |
|--------------|-----------|---------------------|----------------------------------|
| GPT2-XL-1.5B | 0.99%     | 2.85%               | 2.88x                            |
| Llama-2-7B   | 0.15%     | 0.33%               | 2.20x                            |
| Pythia-6.9B  | 0.47%     | 1.02%               | 2.17x                            |
| Phi-1.5B     | 0.11%     | 0.79%               | 7.18x                            |

The percentage of toxic/offensive *questions and answers* (toxicity probability > 0.5) generated using Zero-Shot and Stochastic Few-Shot generation for each model are presented below - 

| Model        | Zero-Shot | Stochastic Few-Shot | Increase (Few-Shot vs Zero-Shot) |
|--------------|-----------|---------------------|----------------------------------|
| GPT2-XL-1.5B | 2.28%     | 8.49%               | 3.72x                            |
| Llama-2-7B   | 0.20%     | 0.61%               | 3.05x                            |
| Pythia-6.9B  | 0.93%     | 3.00%               | 3.23x                            |
| Phi-1.5B     | 0.09%     | 0.38%               | 4.22x                            |

**Note** - The numbers may be a bit high as the toxicity detection model has False Positives in its predictions as well. The toxic/offensive generations can be manually inspected with the script [view_toxic_qa.py](view_toxic_qa.py).

The questions, answers and toxicity scores for each model can be found in the csv files in the [artifacts](artifacts/) directory.

For inference, I have used [vLLM](https://github.com/vllm-project/vllm) (which is awesome!) for GPT2-XL, Llama-2 and Pythia to speedup the generation process. Phi-1.5 is currently not supported by vLLM.