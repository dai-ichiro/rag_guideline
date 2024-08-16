import requests
import json

def post_request():
    url = "http://192.168.11.14:11434/api/generate"
        
    prompt = '''以下にチャットの履歴があります。最後の行が新しいユーザーの質問です。チャット履歴をもとに新しいユーザーの質問をそ れ自体で内容がわかるように簡単に書き換えて下さい。
    ```
    血液透析患者の目標Hb値は
    血液透析患者の目標Hb値は、週初めの採血で10 g/dL以上12 g/dL未満が推奨されています。これは、腎性貧血治療ガイドラインに基づいています。Hb値が10 g/dL未満の場合、複数回の検査でHb値が10 g/dL未満となった時点を治療開始の基準とすることが推奨されています。上限については、Hb値12 g/dLを超える場合には減量・休薬を考慮することが推奨されています。
    腹膜透析患者は
    ```'''
    data = {
        "model": "gemma2:27b",
        "prompt": prompt,
        "options": {
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repeat_penalty": 1.1,
            "mirostat_mode": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
        }
    }

    response = requests.post(url, data=json.dumps(data))

    if response.status_code == 200:
        response_list = list(response.iter_lines())
        answer = "".join([json.loads(x.decode("utf-8"))["response"] for x in response_list[:-1]]).strip()
    else:
        answer = ""
    return answer

if __name__ == "__main__":

    answer = post_request()
    print(answer)

