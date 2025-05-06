import kagglehub
import cv2
import numpy as np
import yolov5
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Configuração das classes de interesse (baseado no COCO dataset)
"""
O que é? Um conjunto de dados com milhares de imagens e anotações precisas.

Contém? Objetos do cotidiano (pessoas, carros, animais, etc.) em cenários complexos.

Para que serve? Treinar modelos como YOLO, Faster R-CNN, etc., em tarefas (ex: detecção de objetos)
"""
CLASSES_OF_INTEREST = {
    0: "pedestre",
    1: "bicicleta",
    2: "carro",
    3: "moto",
    5: "ônibus",
    7: "caminhão"
}

FIXED_DATASET_DIR= "dataset_test"

def get_image_files():
    if not os.path.exists(FIXED_DATASET_DIR):
        raise FileNotFoundError(f"Diretório não encontrado: {FIXED_DATASET_DIR}")
    
    image_files = [os.path.join(FIXED_DATASET_DIR, f) for f in os.listdir(FIXED_DATASET_DIR) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {FIXED_DATASET_DIR}")
    
    print(f"\nEncontradas {len(image_files)} imagens no diretório")
    return image_files

# 1. Função para baixar e preparar o dataset
def setup_dataset():
    try:
        print("Baixando dataset de imagens do Kaggle...")
        
        ### Datasets disponíveis
        # dataset_path = kagglehub.dataset_download("zoltanszekely/mini-traffic-detection-dataset")
        dataset_path = kagglehub.dataset_download("farzadnekouei/top-view-vehicle-detection-image-dataset")
        print(f"Dataset baixado em: {dataset_path}")
        
        # Encontrar arquivos de imagem
        image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            raise FileNotFoundError("Nenhuma imagem encontrada no dataset")
            
        return image_files
    except Exception as e:
        print(f"Erro ao configurar dataset: {e}")
        return []

# 2. Configurar o modelo YOLOv5
def setup_model():
    print("\nCarregando modelo YOLOv5...")
    # Carregamento de modelo pré-treinado
    model = yolov5.load('yolov5s.pt')  # Versão small (mais rápida)
    
    # Configurações do modelo
    model.conf = 0.8  # Limite de confiança
    model.iou = 0.45  # Limite de IoU
    #  Limite de Intersection over Union para supressão de detecções redundantes
    
    """ 
    Trata cada detecção pela classe, e não como algo genérico
    Ex: se houver uma pessoa e um carro próximos, não ignora um deles, mas sim trata os dois.
    Ignora apenas quando é detectado a mesma classe próxima (pega apenas uma)
    """
    model.agnostic = False  # Detecção de classes específicas
    
    return model

# 3. Processar imagens e contar objetos
def process_images(image_files, model, sample_size=None):
    if sample_size:
        image_files = image_files[:sample_size]  # Processar apenas uma amostra
    
    # print(image_files)
    
    # defaultdict -> Cria um dicionário especial que automaticamente inicializa valores inteiros como 0 para qualquer chave nova
    total_counts = defaultdict(int)
    results_data = []
    
    print(f"\nProcessando {len(image_files)} imagens...")
    
    for img_path in tqdm(image_files, desc="Processando imagens"):
        # Ler imagem
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Converter BGR para RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detecção de objetos
        results = model(rgb_img)
        #pred é 0 pois estamos processando imagens únicas, logo, a pred sempre será de tamanho 1 (index 0)
        detections = results.pred[0]
        
        # Contagem por classe na imagem atual
        
        """
        normalizada -> entre 0 e 1
        det[0]: coordenada x do centro da bounding box (normalizada)

        det[1]: coordenada y do centro da bounding box (normalizada)

        det[2]: largura da bounding box (normalizada)

        det[3]: altura da bounding box (normalizada)
        ^^0-3: localização
        det[4]: confiança/score da detecção (0 a 1)
        ^^4: confiança
        det[5]: ID da classe detectada (como inteiro)
        """
        
        img_counts = defaultdict(int)
        for det in detections:
            class_id = int(det[5])
            if class_id in CLASSES_OF_INTEREST:
                class_name = CLASSES_OF_INTEREST[class_id]
                img_counts[class_name] += 1
                total_counts[class_name] += 1
        
        # Armazenar resultados individuais
        results_data.append({
            "image_path": img_path,
            **img_counts
        })
    
    return total_counts, pd.DataFrame(results_data)

# 4. Visualização dos resultados
def visualize_results(total_counts, df_results):
    plt.figure(figsize=(15, 6))
    
    # Gráfico de barras
    plt.subplot(1, 2, 1)
    bars = plt.bar(total_counts.keys(), total_counts.values(), color='skyblue')
    plt.title('Contagem Total de Objetos')
    plt.xlabel('Classe')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    # Boxplot da distribuição
    plt.subplot(1, 2, 2)
    class_cols = [c for c in CLASSES_OF_INTEREST.values() if c in df_results.columns]
    if class_cols:
        df_results[class_cols].plot(kind='box', vert=False)
        plt.title('Distribuição por Imagem')
    else:
        plt.text(0.5, 0.5, 'Sem dados para boxplot', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Estatísticas
    if not df_results.empty and class_cols:
        print("\nEstatísticas por Imagem:")
        print(df_results[class_cols].describe().round(2))

# 5. Função principal
def main():
    # Baixar e configurar dataset
    image_files = setup_dataset()
    if not image_files:
        return
    
    # image_files = get_image_files()
    
    # Configurar modelo
    model = setup_model()
    
    # Processar imagens (usando apenas 100 imagens para demonstração)
    total_counts, df_results = process_images(image_files, model)
    
    # Resultados
    print("\n=== RESULTADOS FINAIS ===")
    for obj_type, count in total_counts.items():
        print(f"{obj_type}: {count}")
    
    # Visualização
    visualize_results(total_counts, df_results)
    
    # Exportar resultados
    df_results.to_csv("resultados_deteccao.csv", index=False)
    print("\nResultados salvos em 'resultados_deteccao.csv'")

if __name__ == "__main__":
    main()