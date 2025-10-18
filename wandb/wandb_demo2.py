# Ensure your OpenAI client is available with:
# pip install openai

# Ensure that your OpenAI API key is available at:
# os.environ['OPENAI_API_KEY'] = "<your_openai_api_key>"

import os
import weave
from openai import OpenAI

weave.init('2230817302-zd/intro-example')  # 🐝


@weave.op()  # 🐝 Decorator to track requests
def create_completion(message: str) -> str:
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1',
        api_key='ms-5a9cdcce-92d1-468b-ad58-bb5d213fc0c5',  # ModelScope Token
    )

    response = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-V3.2-Exp',  # ModelScope Model-Id, required
        messages=[
            {
                'role': 'user',
                'content': message
            }
        ],
        stream=True
    )
    done_reasoning = False
    content = []
    for chunk in response:
        reasoning_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if reasoning_chunk != '':
            print(reasoning_chunk, end='', flush=True)
        elif answer_chunk != '':
            if not done_reasoning:
                print('\n\n === Final Answer ===\n')
                done_reasoning = True
            print(answer_chunk, end='', flush=True)
            content.append(answer_chunk)
    return ''.join(content)


message = "告诉我什么是verl"
create_completion(message)
