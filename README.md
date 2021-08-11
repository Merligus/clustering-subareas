## Projeto de mestrado de clusterização das subáreas da computação

### Git LFS

    Instalar o [Git LFS](https://github.com/git-lfs/git-lfs/wiki/Installation)

### Instalação

1. Clonar repositório
   ```sh
   git clone https://github.com/Merligus/clustering_subareas.git
   cd clustering_subareas/
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