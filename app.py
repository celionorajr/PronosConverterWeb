import logging
from flask import Flask, request, jsonify, send_file
import os
import json
import uuid
from datetime import datetime
from pydicom.dataset import Dataset, FileDataset
from pynetdicom import AE
from pynetdicom.sop_class import CTImageStorage
from PIL import Image
import numpy as np

# Configuração de logging
logging.basicConfig(
    filename="app.log",  # Nome do arquivo de log
    level=logging.INFO,  # Nível de log
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Aplicativo iniciado.")

# Carregar configurações do arquivo config.json
def load_config():
    try:
        with open("config.json", "r") as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        logging.error("Arquivo config.json não encontrado.")
        raise RuntimeError("Arquivo config.json não encontrado.")
    except json.JSONDecodeError as e:
        logging.error(f"Erro ao carregar config.json: {e}")
        raise RuntimeError(f"Erro ao carregar config.json: {e}")

# Gerar UID compatível com o padrão DICOM
def generate_uid(prefix="1.2.826.0.1.3680043.2.1125"):
    return f"{prefix}.{uuid.uuid4().int >> 64}"

# Converter JPG para DICOM
def convert_to_dicom(jpg_path, output_folder, filename, patient_name, study_uid, series_uid):
    try:
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"
        file_meta.ImplementationClassUID = "1.2.276.0.7230010.3.0.3.6.7"

        ds = FileDataset(jpg_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = patient_name
        ds.PatientID = f"ID-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        ds.StudyDate = datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.now().strftime("%H%M%S")
        ds.Modality = "OT"
        ds.StudyDescription = "ENDOSCOPIA"
        ds.SeriesDescription = "ENDOSCOPIA"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        image = Image.open(jpg_path).convert("RGB")
        pixel_array = np.array(image)
        ds.Rows, ds.Columns, _ = pixel_array.shape
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_array.tobytes()

        output_dicom = os.path.join(output_folder, filename.replace(".jpg", ".dcm"))
        ds.save_as(output_dicom)
        logging.info(f"DICOM criado: {output_dicom}")
        return output_dicom
    except Exception as e:
        logging.error(f"Erro ao converter para DICOM: {e}")
        return None

# Enviar DICOM para o PACS
def send_to_pacs(dicom_file, pacs_info):
    try:
        ae = AE()
        ae.add_requested_context(CTImageStorage, "1.2.840.10008.1.2")
        assoc = ae.associate(pacs_info["ip"], pacs_info["port"], ae_title=pacs_info["ae_title"])

        if assoc.is_established:
            status = assoc.send_c_store(dicom_file)
            assoc.release()
            if status and status.Status == 0x0000:
                logging.info(f"DICOM enviado ao PACS: {dicom_file}")
                return True
            else:
                logging.error(f"Erro ao enviar para o PACS: {status}")
                return False
        else:
            logging.error("Associação com PACS falhou.")
            return False
    except Exception as e:
        logging.error(f"Erro ao enviar para o PACS: {e}")
        return False

# Configurar o servidor Flask
app = Flask(__name__)
config = load_config()
output_folder = config.get("output_folder", "output")
os.makedirs(output_folder, exist_ok=True)

@app.route("/")
def index():
    logging.info("Endpoint '/' foi acessado.")
    return send_file("templates/index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    try:
        files = request.files.getlist("folder")
        patient_name = request.form.get("patient_name", "").strip()

        if len(files) == 0 or not patient_name:
            logging.warning("Arquivos ou nome do paciente não fornecidos.")
            return jsonify({"error": "Arquivos ou nome do paciente não fornecidos."}), 400

        study_uid = generate_uid()
        series_uid = generate_uid()

        logs = []
        for file in files:
            sanitized_filename = file.filename.replace("/", "_").replace("\\", "_")
            output_path = os.path.join(output_folder, sanitized_filename)

            file.save(output_path)
            logging.info(f"Arquivo salvo: {output_path}")

            dicom_file = convert_to_dicom(output_path, output_folder, sanitized_filename, patient_name, study_uid, series_uid)
            if dicom_file:
                if send_to_pacs(dicom_file, config["pacs"]):
                    os.remove(dicom_file)
                else:
                    logging.error(f"Erro ao enviar para o PACS: {dicom_file}")
                    return jsonify({"error": "Erro ao enviar para o PACS.", "logs": logs}), 500
            else:
                logging.error(f"Erro ao converter para DICOM: {output_path}")
                return jsonify({"error": f"Erro ao converter {file.filename} para DICOM.", "logs": logs}), 500

            os.remove(output_path)
            logging.info(f"Arquivo JPG excluído: {output_path}")

        logging.info("Todos os arquivos processados com sucesso.")
        return jsonify({"message": "Arquivos enviados e processados com sucesso!", "logs": logs}), 200
    except Exception as e:
        logging.error(f"Erro inesperado: {e}", exc_info=True)
        return jsonify({"error": f"Erro inesperado: {str(e)}"}), 500

if __name__ == "__main__":
    logging.info("Aplicativo iniciado no modo debug.")
    app.run(host="0.0.0.0", port=5000, debug=True)
