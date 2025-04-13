import cv2
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("🟢 Programa iniciado")

# Carregar o modelo leve
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to("cpu")

# Captura da imagem
def capturar_imagem():
    cam = cv2.VideoCapture(0)
    time.sleep(2)  # Dá tempo de se posicionar
    ret, frame = cam.read()
    if not ret:
        print("❌ Não foi possível capturar a imagem da webcam.")
        cam.release()
        return None
    img_path = "frame.jpg"
    cv2.imwrite(img_path, frame)
    cam.release()
    print("📷 Imagem capturada e salva em:", img_path)
    return img_path

# Simulação de detecção de dor
def detectar_dor_simulada(imagem_path):
    image = Image.open(imagem_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cpu")

    print("🧠 Gerando descrição da imagem...")
    output = model.generate(**inputs)
    descricao = processor.decode(output[0], skip_special_tokens=True)

    print(f"📝 Descrição gerada: {descricao}")

    # Lista expandida de palavras que podem indicar dor
    palavras_de_dor = [
        "pain", "sad", "crying", "hurt", "injury", "injured", "sick",
        "ill", "unwell", "suffering", "discomfort", "grimace", "headache",
        "stomachache", "bandage", "wound", "bruise", "bleeding", "tear",
        "tears", "broken", "hospital", "cast", "wheelchair", "limping",
        "ambulance", "nurse", "doctor", "tired", "fatigue", "weak", "exhausted"
    ]

    if any(p in descricao.lower() for p in palavras_de_dor):
        print("⚠️ ALERTA: A pessoa pode estar com dor!")
    else:
        print("✅ Aparentemente, a pessoa está bem.")


# Execução
print("📸 Capturando imagem...")
imagem = capturar_imagem()
if imagem:
    detectar_dor_simulada(imagem)
