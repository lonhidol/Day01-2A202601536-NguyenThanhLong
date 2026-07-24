import os
from dotenv import load_dotenv
from openai import OpenAI

# Nạp API Key từ file .env
load_dotenv()
client = OpenAI()
if __name__ == "__main__":
    prompt = "Hãy kể cho tôi một sự thật thú vị về Hà Nội."
    temperatures = [0.0, 0.7, 1.2, 1.8]

    for temp in temperatures:
        print("=" * 60)
        print(f"🚀 Đang gọi API với Temperature = {temp}...")
        try:
            # Truyền đích danh chosen_model vào hàm call_openai
            response, latency = call_openai(
                prompt=prompt,
                model=chosen_model,
                temperature=temp
            )
            print(f"⏱️ Độ trễ: {latency:.2f} giây")
            print(f"💬 Phản hồi:\n{response}")
            
        except Exception as e:
            print(f"❌ Lỗi ở temperature {temp}: {e}")
        print()