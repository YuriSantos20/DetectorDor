import cv2
import time
import torch
import requests
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

print("🟢 Iniciando sistema com BLIP-2 + ESP32-CAM")

# Carregar modelo e processador BLIP-2
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",
    torch_dtype=torch.float16
)

# Captura de imagem via ESP32-CAM
def capturar_imagem_esp32():
    url = "http://192.168.1.100/capture"  # 🔁 Troque pelo IP da sua ESP32-CAM
    try:
        print("🌐 Capturando imagem da ESP32-CAM...")
        response = requests.get(url, timeout=5)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)
        img_path = "frame.jpg"
        cv2.imwrite(img_path, frame)
        print("📷 Imagem salva:", img_path)
        return img_path
    except Exception as e:
        print("❌ Erro ao capturar imagem da ESP32:", e)
        return None

# Função de análise com perguntas
def detectar_dor_blip2(imagem_path):
    image = Image.open(imagem_path).convert("RGB")
    
    perguntas = [
        "Is the person in pain?",
        "Is the person injured?",
        "Is the person suffering?",
        "Does the person look sick?"
    ]
    
    for pergunta in perguntas:
        inputs = processor(images=image, text=pergunta, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu", torch.float16)
        outputs = model.generate(**inputs)
        resposta = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"❓ {pergunta}")
        print(f"➡️ Resposta: {resposta}\n")
        
        if "yes" in resposta.lower():
            print("⚠️ ALERTA: Dor detectada!")
            return
    
    print("✅ Aparentemente, a pessoa está bem.")

# ------------------
# EXECUÇÃO PRINCIPAL
# ------------------

print("📸 Capturando imagem da ESP32-CAM...")
imagem = capturar_imagem_esp32()
if imagem:
    detectar_dor_blip2(imagem)

# 📦 Instalar os pacotes necessários (no terminal):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers opencv-python pillow requests
