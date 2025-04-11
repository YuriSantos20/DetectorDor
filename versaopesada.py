import cv2
import time
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

print("🟢 Iniciando sistema com BLIP-2")

# Carregar modelo e processador BLIP-2 (pesado!)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",  # Tenta usar GPU automaticamente
    torch_dtype=torch.float16
)

# Função para capturar imagem
def capturar_imagem():
    cam = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cam.read()
    if not ret:
        print("❌ Erro ao acessar a webcam.")
        cam.release()
        return None
    img_path = "frame.jpg"
    cv2.imwrite(img_path, frame)
    cam.release()
    print("📷 Imagem salva:", img_path)
    return img_path

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

# Execução
print("📸 Capturando imagem...")
imagem = capturar_imagem()
if imagem:
    detectar_dor_blip2(imagem)

# 📦 Instalar os pacotes necessários (no terminal):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers opencv-python pillow