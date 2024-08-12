from mlx_lm import load, generate

# モデルとトークナイザーのロード
model, tokenizer = load("mlx-community/Phi-3-mini-128k-instruct-4bit")

def main():
    print("対話型質問応答システムへようこそ！終了するには 'exit' と入力してください。")
    while True:
        # ユーザーからの入力を受け取る
        user_input = input("質問を入力してください: ")
        
        # 'exit' が入力された場合、ループを終了
        if user_input.lower() == 'exit':
            print("終了します。")
            break
        
        # モデルを使って応答を生成
        response = generate(model, tokenizer, prompt=user_input, verbose=True)
        
        # 応答を表示
        print("応答: ", response)

if __name__ == "__main__":
    main()
