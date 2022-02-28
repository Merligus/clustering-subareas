## Projeto de mestrado de clusterização das subáreas da computação

A consulta pode ser feita no site [merligus.pythonanywhere.com](https://merligus.pythonanywhere.com).

### Instalação

1. Clonar repositório
   ```sh
   git clone https://github.com/Merligus/clustering-subareas.git
   cd clustering-subareas/
   ```
2. Criar o ambiente
   ```sh
   python3 -m venv clustering_subareas_env
   ```
3. Ativar o ambiente
    ```sh
    source clustering_subareas_env/bin/activate
    ```
4. Instalar os pacotes
    ```sh
    python3 -m pip install -r requirements.txt
    ```
5. Executar
    ```sh
    python3 -m web.web
    ```

### Dados

Baixar os [dados](https://drive.google.com/drive/folders/1o9CFw8kaRpTW1oUa4SBXMNNRt5XwOXBE?usp=sharing) necessários para o algoritmo. E colocar os arquivos em clustering_subareas/data/* .
