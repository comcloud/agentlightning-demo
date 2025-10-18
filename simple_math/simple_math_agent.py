# Copyright (c) Microsoft. All rights reserved.
"""
Simple Math Problem Solver Agent using AgentLightning framework
"""

import os
import re
import time
import random
from typing import Any, Dict, Optional, cast
import json

import agentlightning
from agentlightning import LLM, NamedResources
from agentlightning.litagent import LitAgent

# Configure logger
logger = agentlightning.configure_logger(name=__name__)


class SimpleMathAgent(LitAgent):
    """A simple math problem solver agent using AgentLightning framework."""

    def __init__(
            self,
            trained_agents: Optional[str] = "math_solver",
            val_temperature: Optional[float] = None,
            max_attempts: int = 3,
            optimization_algorithm: str = "ppo",  # Can be "ppo" or "grpo"
            reward_function: str = "accuracy",  # Can be "accuracy" or "steps"
    ) -> None:
        """
        Initialize the SimpleMathAgent.
        
        Args:
            trained_agents: Identifier for trained agents
            val_temperature: Temperature for validation
        max_attempts: Maximum number of attempts to solve a problem
            optimization_algorithm: Algorithm to use for optimization ("ppo" or "grpo")
            reward_function: Function to calculate reward ("accuracy" or "steps")
        """
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.max_attempts = max_attempts
        self.optimization_algorithm = optimization_algorithm
        self.reward_function = reward_function

        # System prompt for the math solver
        self.system_prompt = """
You are a math problem solver. Solve the given math problem step by step.
Provide your final answer in the format: "Answer: [number]"

Examples:
Problem: What is 2 + 2?
Answer: 4

Problem: Calculate 15 * 3
Answer: 45
""".strip()

    def _execute_rollout(
            self, sample: dict[str, Any], *, resources: NamedResources, rollout_id: str, is_training: bool
    ) -> float | None:
        """
        Execute a single rollout for math problem solving.
        
        Args:
            sample: The math problem sample containing 'question' and 'answer'
            resources: Resources including LLM
            rollout_id: Identifier for this rollout
            is_training: Whether this is a training rollout
            
        Returns:
            Reward value for this rollout
        """
        question = sample["question"]
        ground_truth = sample["answer"]
        start_time = time.time()

        logger.info(f"[Rollout {rollout_id}] Question: {question}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

        # Get the LLM resource
        llm: LLM = cast(LLM, resources["main_llm"])

        # Set temperature based on training/validation mode
        temperature = llm.sampling_parameters.get("temperature", 0.0)
        if not is_training and self.val_temperature is not None:
            temperature = self.val_temperature

        # Attempt to solve the problem with multiple tries
        attempts = []
        for attempt_num in range(self.max_attempts):
            # Prepare the prompt
            if attempts:
                # If we have previous attempts, include them in the prompt
                previous_attempts = "\n".join([
                    f"Attempt {i + 1}: {attempt}" for i, attempt in enumerate(attempts)
                ])
                prompt = f"""
{self.system_prompt}

Previous attempts:
{previous_attempts}

Problem: {question}

Try again, learn from previous mistakes:
""".strip()
            else:
                # First attempt
                prompt = f"""
{self.system_prompt}

Problem: {question}
""".strip()

                # Call the LLM
                try:
                    response = llm.generate(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=200
                    )
                    answer = response.generations[0].text.strip()
                    attempts.append(answer)

                    logger.info(f"[Rollout {rollout_id}] Attempt {attempt_num + 1}: {answer}")

                    # Try to extract the numerical answer
                    extracted_answer = self._extract_answer(answer)
                    if extracted_answer is not None:
                        break  # Successfully extracted answer, exit retry loop
                except Exception as e:
                    logger.error(f"[Rollout {rollout_id}] Error in attempt {attempt_num + 1}: {e}")
                    attempts.append(f"Error: {str(e)}")

        # Use the last attempt if we couldn't extract a numerical answer
        final_answer = attempts[-1] if attempts else "No answer generated"
        extracted_answer = self._extract_answer(final_answer)

        end_time_rollout = time.time()

        # Calculate reward
        reward = self._calculate_reward(extracted_answer, ground_truth, len(attempts))
        logger.info(f"[Rollout {rollout_id}] Final Answer: {final_answer}")
        logger.info(f"[Rollout {rollout_id}] Extracted Answer: {extracted_answer}")
        logger.info(f"[Rollout {rollout_id}] Reward: {reward}")
        logger.info(f"[Rollout {rollout_id}] Time taken: {end_time_rollout - start_time:.2f} seconds")

        return reward

    def _extract_answer(self, response: str) -> Optional[float]:
        """
        Extract numerical answer from the LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted numerical answer or None if not found
        """
        # Look for "Answer: [number]" pattern
        answer_match = re.search(r"Answer:\s*([+-]?\d+\.?\d*)", response)
        if answer_match:
            try:
                return float(answer_match.group(1))
            except ValueError:
                pass

        # Look for any number at the end of the response
        number_match = re.search(r"([+-]?\d+\.?\d*)$", response.strip())
        if number_match:
            try:
                return float(number_match.group(1))
            except ValueError:
                pass

        return None

    def _calculate_reward(self, answer: Optional[float], ground_truth: float, attempts: int) -> float:
        """
        Calculate reward based on the answer accuracy and number of attempts.
        
        Args:
            answer: Extracted answer from LLM
            ground_truth: Correct answer
            attempts: Number of attempts taken
            
        Returns:
            Reward value between 0 and 1
        """
        if self.reward_function == "accuracy":
            # Reward based purely on accuracy
            if answer is None:
                return 0.0
            if abs(answer - ground_truth) < 1e-6:  # Exact match
                return 1.0
            elif abs(answer - ground_truth) < 0.01:  # Very close
                return 0.8
            elif abs(answer - ground_truth) < 0.1:  # Close
                return 0.5
            else:
                return 0.0
        elif self.reward_function == "steps":
            # Reward based on accuracy and efficiency (fewer attempts = better)
            accuracy_reward = self._calculate_reward(answer, ground_truth,
                                                     attempts) if self.reward_function != "steps" else (
                1.0 if answer is not None and abs(answer - ground_truth) < 1e-6 else 0.0
            )
            # Efficiency reward - fewer attempts get higher reward
            efficiency_reward = max(0.0, 1.0 - (attempts - 1) * 0.2)  # Max 1.0, decrease by 0.2 per attempt
            return (accuracy_reward + efficiency_reward) / 2.0
        else:
            # Default to accuracy-based reward
            return self._calculate_reward(answer, ground_truth, attempts) if answer is not None and abs(
                answer - ground_truth) < 1e-6 else 0.0

    def training_rollout(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        """Execute a training rollout."""
        return self._execute_rollout(task, resources=resources, rollout_id=rollout_id, is_training=True)

    def validation_rollout(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        """Execute a validation rollout."""
        return self._execute_rollout(task, resources=resources, rollout_id=rollout_id, is_training=False)


def generate_math_dataset(size: int = 100) -> list[Dict[str, Any]]:
    """
    Generate a simple math dataset for training.
    
    Args:
        size: Number of samples to generate
        
    Returns:
        List of math problem samples
    """
    samples = []
    operations = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]

    for i in range(size):
        # Generate random numbers
        a = random.randint(1, 100)
        b = random.randint(1, 100)

        # Select random operation
        op_symbol, op_func = random.choice(operations)

        # Create question and answer
        question = f"What is {a} {op_symbol} {b}?"
        answer = op_func(a, b)

        samples.append({
            "question": question,
            "answer": answer,
            "operation": op_symbol
        })

    return samples


def math_data_loader(mode='train'):
    """
    Create a data loader for math problems.
    
    Returns:
        DevTaskLoader with math problem samples
    """
    # Create resources
    resource = {
        "main_llm": LLM(
            model=os.environ.get("MODEL", "qwen3-max"),
            endpoint=os.environ.get("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.environ.get("API_KEY", "ms-5a9cdcce-92d1-468b-ad58-bb5d213fc0c5"),
            sampling_parameters={
                "temperature": 0.7,
                "max_tokens": 2000,
            },
        )
    }
    if mode == 'train':
        # Generate training data
        train_data = generate_math_dataset(50)
        return agentlightning.DevTaskLoader(
            train_data,
            resource
        )
    else:
        # Generate validation data
        val_data = generate_math_dataset(20)
        return agentlightning.DevTaskLoader(
            val_data,
            resource
        )


if __name__ == "__main__":
    # Load environment variables
    import dotenv

    dotenv.load_dotenv()

    # # Create agent
    # agent = SimpleMathAgent(
    #     trained_agents="math_solver",
    #     max_attempts=3,
    #     optimization_algorithm="ppo",
    #     reward_function="accuracy"
    # )

    # Create trainer
    agent, trainer = agentlightning.lightning_cli(SimpleMathAgent, agentlightning.Trainer)
    # Train the agent
    trainer.fit(
        agent,
        os.environ.get("VERL_API_BASE", "http://localhost:9999/"),
        dev_data=math_data_loader('train'),
        val_data=math_data_loader('val')
    )
