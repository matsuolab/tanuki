# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

# Magpie sample script.

# Get an API key from NVIDIA page.
# Press "Get API Key" button on the following page.
# https://build.nvidia.com/nvidia/nemotron-4-340b-instruct

# Set the API key as an environment variable.
# export NVIDIA_API_KEY="nvapi-..."


import json
import os

from openai import OpenAI

messages = [
    {
        "role": "system",
        "content": "以下は、解答するために論理的思考力とプログラミングが必要な問題です。問題は簡潔に1行で記述してください。",
    }
]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    # base_url="https://api.deepinfra.com/v1/openai",
    # api_key=os.getenv("DEEPINFRA_API_KEY"),
)

for id in range(2):
    completion = client.chat.completions.create(
        model="nvidia/nemotron-4-340b-instruct",
        # model="nvidia/Nemotron-4-340B-Instruct",
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        seed=1,
        stop=["\n\n"],
    )
    content = completion.choices[0].message.content
    role = completion.choices[0].message.role
    print(json.dumps({"id": id, "messages": messages + [{"role": role, "content": content}]}, ensure_ascii=False))
