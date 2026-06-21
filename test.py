import google.generativeai as genai

genai.configure(api_key="AQ.Ab8RN6IuziCZeaQtM4a71Na5gP5ZZfEj26qZjkQb-tiOMd2vqg")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)