# Simple Math Agent Example

This is a simplified example of using the AgentLightning framework to train a math problem solving agent. It demonstrates the core concepts of reinforcement learning with LLMs in a lightweight, easy-to-understand format.

## Overview

The Simple Math Agent is designed to solve basic arithmetic problems using reinforcement learning. It uses the OpenAI API as the base language model and learns to improve its problem-solving capabilities through training.

## Features

1. **Lightweight Training**: Uses simple math problems that can be trained quickly
2. **Configurable Components**:
   - Multiple optimization algorithms (PPO, GRPO)
   - Different reward functions (accuracy-based, efficiency-based)
   - Adjustable maximum attempts per problem
3. **Self-improvement**: Agent learns from previous attempts to solve problems
4. **Modular Design**: Easy to extend for other types of problems

## Files

- `simple_math_agent.py`: Main agent implementation
- `train_simple_math.sh`: Training script with configurable parameters
- `README.md`: This file

## Setup

1. Ensure you have the AgentLightning framework installed
2. Set the required environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export OPENAI_API_BASE=https://api.openai.com/v1  # Optional, for custom endpoints
   ```

## Usage

### Training

To train the simple math agent:

```bash
cd rl_manus
bash train_simple_math.sh
```

### Customization

You can customize the training by modifying the agent parameters in `simple_math_agent.py`:

```python
agent = SimpleMathAgent(
    trained_agents="math_solver",
    max_attempts=3,              # Maximum attempts per problem
    optimization_algorithm="ppo", # Can be "ppo" or "grpo"
    reward_function="accuracy"   # Can be "accuracy" or "steps"
)
```

Or by passing command-line arguments to the training script:

```bash
bash train_simple_math.sh trainer.total_epochs=5 algorithm.adv_estimator=grpo
```

## How It Works

### Agent Architecture

The SimpleMathAgent follows the AgentLightning framework pattern:

1. **Initialization**: Sets up the agent with configurable parameters
2. **Rollout Execution**: For each math problem:
   - Attempts to solve it with multiple tries
   - Learns from previous mistakes
   - Generates a response using the LLM
3. **Reward Calculation**: Evaluates the solution based on accuracy and efficiency
4. **Training Loop**: Uses reinforcement learning to improve the agent's policy

### Reward Functions

1. **Accuracy-based**: Rewards based purely on how close the answer is to the correct one
2. **Steps-based**: Rewards based on both accuracy and the number of attempts taken

### Optimization Algorithms

1. **PPO (Proximal Policy Optimization)**: Standard policy optimization algorithm
2. **GRPO (Generalized Reward Policy Optimization)**: Advanced algorithm for better reward handling

## Extending the Example

You can easily extend this example for other types of problems:

1. **Modify the dataset generation** in `generate_math_dataset()` to create different types of problems
2. **Adjust the system prompt** to guide the LLM for different tasks
3. **Customize the reward function** for domain-specific evaluation
4. **Add more sophisticated error analysis** in the agent's retry logic

## Example Output

During training, you'll see output like:

```
[Rollout 1] Question: What is 15 + 27?
[Rollout 1] Ground Truth: 42
[Rollout 1] Attempt 1: Let me solve this step by step. 15 + 27 = 42. Answer: 42
[Rollout 1] Extracted Answer: 42.0
[Rollout 1] Reward: 1.0
```

## Performance Considerations

- Training with OpenAI API can be costly due to API usage
- The example is designed to be lightweight and complete quickly
- You can adjust the dataset size and training epochs to balance cost and performance

## Troubleshooting

1. **API Key Issues**: Ensure `OPENAI_API_KEY` is set correctly
2. **Rate Limiting**: If you encounter rate limits, reduce the batch size or training frequency
3. **Memory Issues**: For large models, adjust the micro-batch sizes in the training script
