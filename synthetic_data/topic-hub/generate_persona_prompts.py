import json

# prompt generation using Persona-Hub method.
# https://arxiv.org/abs/2406.20094

# download persona.jsonl from PersonaHub dataset
#
# huggingface-cli download proj-persona/PersonaHub persona.jsonl \
#   --repo-type=dataset --local-dir="." --local-dir-use-symlinks=False
# mv persona.jsonl data/persona.jsonl

# generate Japanese version of persona jsonl.
# python scripts/extract_persona.py > data/persona_jp.jsonl

# persona_jsonl = "data/persona.jsonl"
persona_jsonl = "data/persona_jp.jsonl"

personas = []
with open(persona_jsonl, "r") as f:
    for line in f:
        p = json.loads(line)
        personas.append(p["persona"])

tasks = [
    "deductive reasoning",
    "inductive reasoning",
    "abductive reasoning",
    "analogical reasoning",
    "spatial reasoning",
    # "logical reasoning",
    # "math",
    # "programming",
]

task = tasks[0]

# see https://github.com/tencent-ailab/persona-hub/blob/main/code/prompt_templates.py
PROMPT_TEMPLATE = """<extra_id_0>System

<extra_id_1>User
Create a {task} problem related to the following persona:

{persona}

Note:

1. The {task} problem should be simple and involve basic {task} skills and knowledge. All average high school students can solve it correctly.
2. You should make full use of the persona description to create the {task} problem to ensure that the {task} problem is unique and specific to the persona.
3. Your response should always start with "Problem:". Your response should not include a solution to the created {task} problem.
4. 日本語で回答しなさい.
<extra_id_1>Assistant
"""  # noqa: E501

for i in range(100):
    for persona in personas:
        prompt = PROMPT_TEMPLATE.format(task=task, persona=persona)
        print(json.dumps(prompt, ensure_ascii=False))
