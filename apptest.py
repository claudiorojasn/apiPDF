import pytesseract
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError
import re
import base64
from io import BytesIO
import os

# Importaciones de FLASK / Flask-Smorest
from flask import Flask, request, jsonify, send_file
from flask.views import MethodView
from flask_smorest import Api, Blueprint, abort
from marshmallow import Schema, fields

# --- Configuración de Tesseract ---
# Es una buena práctica apuntar explícitamente al ejecutable, especialmente en Windows.
# Reemplaza la ruta si lo instalaste en otro lugar.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Importaciones de Firma PyHanko
from pyhanko.sign import signers
from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
from pyhanko.sign.signers import PdfSigner

# --- Configuración General ---

# Carga las claves de API desde variables de entorno para mayor seguridad.
# Ejemplo: set API_KEYS=tu_clave_secreta_12345,otra_clave_valida
API_KEYS = os.environ.get('API_KEYS', 'tu_clave_secreta_12345').split(',')

app = Flask(__name__)

# --- Configuración de Flask-Smorest (para Firma) ---
app.config["API_TITLE"] = "API de Documentos: OCR y Firma"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.2"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

api = Api(app)

# --- Blueprint de Firma (usando Flask-Smorest para Swagger) ---
blp_firma = Blueprint(
    "Firma",
    __name__,
    url_prefix="/api/firma",
    description="Operaciones de firma de documentos digitales"
)

# --- Esquemas para la Firma ---
class SignPdfSchema(Schema):
    pdf_file_base64 = fields.String(required=True, metadata={"description": "El archivo PDF en formato Base64"})
    pfx_file_base64 = fields.String(required=True, metadata={"description": "El archivo PFX del certificado en formato Base64"})
    pfx_passphrase = fields.String(required=True, metadata={"description": "La contraseña del archivo PFX"})


@blp_firma.route("/sign-pdf-base64")
class SignPdf(MethodView):
    @blp_firma.arguments(SignPdfSchema, location="json")
    @blp_firma.response(200, content_type="application/pdf")
    def post(self, args):
        """
        Firma un PDF digitalmente usando un certificado PFX.
        """
        api_key = request.headers.get('X-API-KEY')
        
        if api_key not in API_KEYS:
            # Usar abort de Smorest para integrarse con el manejo de errores del framework
            abort(401, message="Acceso no autorizado. Clave de API inválida.")
        
        try:
            # 1. Decodificar Base64
            pdf_bytes = base64.b64decode(args['pdf_file_base64'])
            pfx_bytes = base64.b64decode(args['pfx_file_base64'])
            pfx_passphrase = args['pfx_passphrase']

            # 2. Cargar el firmante
            signer = signers.SimpleSigner.load_pkcs12(
                pfx_file=pfx_bytes,
                passphrase=pfx_passphrase.encode('utf-8')
            )
            
            # 3. Leer el PDF y firmarlo
            pdf_in_memory = BytesIO(pdf_bytes)
            w = IncrementalPdfFileWriter(pdf_in_memory)
            
            meta = signers.PdfSignatureMetadata(field_name='Signature1')
            pdf_signer = PdfSigner(meta, signer)
            out = pdf_signer.sign_pdf(w)
            
            # 4. Devolver el PDF firmado
            signed_pdf_bytes = out.getvalue()
            return send_file(
                BytesIO(signed_pdf_bytes),
                mimetype='application/pdf',
                as_attachment=True,
                download_name='documento_firmado.pdf'
            )
        
        except base64.binascii.Error:
            abort(400, message="Los datos Base64 (PDF o PFX) no son válidos.")
        except Exception as e:
            # PyHanko lanza errores detallados, lo capturamos
            abort(500, message=f"Error al firmar el documento: {str(e)}")

api.register_blueprint(blp_firma)


## 📄 Endpoint de OCR y Extracción de Datos (Flask Básico)

## Este endpoint utiliza el método Flask tradicional (`@app.route`) y se mantiene separado de Flask-Smorest.

@app.route("/api/ocr/extract-from-pdf", methods=["POST"])
def extract_data():
    """
    Recibe un PDF, realiza OCR y extrae información clave.
    """
    # Verificación de clave API (Opcional si usas middleware, pero lo incluimos aquí)
    api_key = request.headers.get('X-API-KEY')
    if api_key not in API_KEYS:
        return jsonify({"error": "Acceso no autorizado. Clave de API inválida."}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo 'file' en la solicitud."}), 400

    file = request.files['file']
    
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Por favor, sube un archivo PDF válido."}), 400

    try:
        # Lógica de OCR y Extracción
        pdf_bytes = file.read()
        
        try:
            # DPI más bajo para velocidad, 300 es estándar.
            images = convert_from_bytes(pdf_bytes, 200) 
        except PDFInfoNotInstalledError:
            return jsonify({"error": "Poppler no está instalado o no se encuentra en el PATH del sistema. Es necesario para convertir el PDF a imágenes."}), 500

        full_text = ""
        
        for image in images:
            text = pytesseract.image_to_string(image, lang='spa')
            full_text += text + "\n---\n"

        # Extracción de Información Específica con RegEx
        date_pattern = r'(\d{2}[-/]\d{2}[-/]\d{4})'
        doc_pattern = r'ID\s*:\s*(\d{6})\b' # Ejemplo: buscar "ID: 123456"
        
        fecha_encontrada = re.search(date_pattern, full_text)
        doc_encontrado = re.search(doc_pattern, full_text)

        extracted_data = {
            "file_name": file.filename,
            "date": fecha_encontrada.group(1) if fecha_encontrada else "No encontrada",
            "document_id": doc_encontrado.group(1) if doc_encontrado else "No encontrado",
            "full_ocr_text_snippet": full_text[:500] + "..." if len(full_text) > 500 else full_text
        }
        
        return jsonify(extracted_data), 200

    except pytesseract.TesseractError:
        return jsonify({"error": "Error de Tesseract/OCR. Asegúrate de que Tesseract-OCR esté instalado y la ruta sea correcta."}), 500
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error en el procesamiento: {e}"}), 500

# --- Ejecución de la Aplicación ---

if __name__ == '__main__':
    app.run(debug=True)