# Pronos Converter

## 🩺 Descrição
Pronos Converter é uma aplicação para conversão de imagens médicas no formato **JPG** para **DICOM**, com envio direto para um sistema **PACS (Picture Archiving and Communication System)**. Ideal para uso em clínicas e hospitais que precisam digitalizar e gerenciar imagens médicas.

---

## 📂 Estrutura do Projeto

```plaintext
PronosConverterWeb/
├── app.log              # Arquivo de logs da aplicação
├── app.py               # Arquivo principal da aplicação Flask
├── config.json          # Arquivo de configuração com informações do PACS
├── iniciar.bat          # Script para iniciar a aplicação no Windows
├── output/              # Diretório onde os arquivos DICOM temporários são armazenados
├── static/              # Recursos estáticos (CSS, imagens)
│   └── style.css        # Arquivo de estilos
├── tempCodeRunnerFile.py # (Opcional) Arquivo temporário gerado por alguns editores
└── templates/           # Diretório para templates HTML
    └── index.html       # Página inicial da aplicação
```

---

## 🛠️ Dependências

As seguintes bibliotecas Python são necessárias para rodar a aplicação:

- Flask
- pydicom
- pynetdicom
- Pillow
- numpy

---

## ⚙️ Configuração

Certifique-se de configurar corretamente o arquivo `config.json` antes de iniciar a aplicação:

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

- **`output_folder`**: Pasta onde os arquivos DICOM serão temporariamente armazenados antes do envio ao PACS.
- **`pacs`**: Configuração de conexão com o sistema PACS:
  - `ae_title`: Título da aplicação no PACS.
  - `ip`: Endereço IP do PACS.
  - `port`: Porta usada para comunicação com o PACS.

---

## 🚀 Executando a Aplicação

1. **Instale as dependências:**

   ```sh
   pip install flask pydicom pynetdicom pillow numpy
   ```

2. **Inicie a aplicação:**

   ```sh
   python app.py
   ```

3. **Acesse no navegador:**

   ```
   http://127.0.0.1:5000
   ```

4. **Para usuários de Windows**, você pode usar o arquivo `iniciar.bat` para facilitar o início.

---

## 🔧 Funcionalidades

- **Upload de Imagens**: Selecione uma pasta contendo arquivos JPG para envio.
- **Conversão Automática**: Cada imagem JPG é convertida para o formato DICOM.
- **Envio ao PACS**: Os arquivos DICOM são enviados para o PACS configurado.
- **Limpeza Automática**: Após o envio, os arquivos temporários são excluídos.

---

## 📜 Logs

Todas as ações e erros são registrados no arquivo `app.log`, localizado no diretório principal.

---

## 💻 Instalação

1. **Instale o Python**:
   Baixe e instale a versão mais recente do Python no [site oficial](https://www.python.org/downloads/).

2. **Clone o repositório:**

   ```sh
   git clone https://github.com/celionorajr/PronosConverterWeb.git
   cd PronosConverterWeb
   ```

3. **Configure e execute a aplicação** (veja a seção **Executando a Aplicação**).

---

## 🤝 Contribuindo

Quer ajudar a melhorar o Pronos Converter? Siga estes passos:

1. Faça um fork do repositório.
2. Crie uma nova branch para sua funcionalidade:
   ```sh
   git checkout -b minha-nova-funcionalidade
   ```
3. Faça suas alterações e commite:
   ```sh
   git commit -m 'Adiciona nova funcionalidade'
   ```
4. Envie para o repositório remoto:
   ```sh
   git push origin minha-nova-funcionalidade
   ```
5. Crie um **pull request** e aguarde a revisão.

---

