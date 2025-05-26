import json

# count = 256
count = 1024 * 10  # 10k

prompt = """<extra_id_0>System
以下の難易度の高い質問に日本語で答えてください。
<extra_id_1>User
"""

prompt = """<extra_id_0>System
以下は、解答するために論理的思考力とプログラミングが必要な問題です。問題は簡潔に1行で記述してください。
<extra_id_1>User
"""

prompt = """<extra_id_0>System
以下は、解答するために論理的思考力と算数の知識が必要な問題です。問題は簡潔に1行で記述してください。
<extra_id_1>User
"""

prompt = """<extra_id_0>System
以下は、解答するために論理的思考力と多段推論が必要な問題です。問題は簡潔に1行で記述してください。
<extra_id_1>User
"""

prompt = """<extra_id_0>System
以下は、解答するために論理的思考力と高度な推論が必要な問題です。問題は簡潔に1行で記述してください。
<extra_id_1>User
"""

for _ in range(count):
    print(json.dumps(prompt, ensure_ascii=False))
