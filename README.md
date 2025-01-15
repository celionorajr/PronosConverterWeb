# Pronos Converter

## Descrição
Este projeto é um conversor de imagens médicas que permite o upload de arquivos JPG, converte-os para o formato DICOM e envia-os para um PACS (Picture Archiving and Communication System).

## Estrutura do Projeto

app.log
app.py
config.json
iniciar.bat
output/
static/
    style.css
tempCodeRunnerFile.py
templates/
    index.html


## Dependências
- Flask
- pydicom
- pynetdicom
- Pillow
- numpy

## Configuração
1. **config.json**:
    ```json
    {
        "output_folder": "output",
        "pacs": {
            "ae_title": "PACSGAMA",
            "ip": "192.168.169.230",
            "port": 11112
        }
    }
    ```
    - [output_folder](http://_vscodecontentref_/7): Pasta onde os arquivos convertidos serão salvos.
    - `pacs`: Informações de conexão com o PACS.

## Executando a Aplicação
1. Certifique-se de ter todas as dependências instaladas:
    ```sh
    pip install flask pydicom pynetdicom pillow numpy
    ```

2. Inicie a aplicação:
    ```sh
    python app.py
    ```

3. Acesse a aplicação no navegador:
    ```
    http://127.0.0.1:5000
    ```

## Funcionalidades
- **Upload de Imagens**: Permite o upload de uma pasta contendo arquivos JPG.
- **Conversão para DICOM**: Converte os arquivos JPG para o formato DICOM.
- **Envio para PACS**: Envia os arquivos DICOM para o PACS configurado.

## Logs
Os logs da aplicação são salvos no arquivo [app.log](http://_vscodecontentref_/8).


## Instalação
1. Instale o Python a partir do [site oficial](https://www.python.org/downloads/).
2. Clone o repositório:
    ```sh
    git clone https://github.com/celionorajr/ProbosConverterWeb.git
    cd PronosConverterWeb
    ```

## Contribuição
1. Faça um fork do projeto.
2. Crie uma nova branch:
    ```sh
    git checkout -b minha-nova-funcionalidade
    ```
3. Faça suas alterações e commit:
    ```sh
    git commit -m 'Adiciona nova funcionalidade'
    ```
4. Envie para o branch original:
    ```sh
    git push origin minha-nova-funcionalidade
    ```
5. Crie um pull request.#   P r o n o s C o n v e r t e r W e b  
 