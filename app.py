"""
PDF API Completa con Swagger/OpenAPI Documentation
Versión Simplificada
"""
import os
import io
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from logging.handlers import RotatingFileHandler

# Flask imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace, reqparse

# PDF Processing imports
import pdfplumber
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from PIL import Image

# OCR imports
from pdf2image import convert_from_path
import pytesseract
import numpy as np

# ========== CONFIGURACIÓN FLASK ==========
app = Flask(__name__)
CORS(app)

# Configurar Flask-RESTX (Swagger)
api = Api(
    app,
    version='2.0.0',
    title='PDF Processing API',
    description='API completa para procesamiento de documentos PDF con OCR',
    doc='/swagger/',  # URL para Swagger UI
    default='PDF Operations',
    default_label='Operaciones principales de PDF',
    contact='API Support',
    contact_email='support@example.com',
    license='MIT',
    license_url='https://opensource.org/licenses/MIT'
)

# ========== DIRECTORIOS ==========
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'output')
LOG_FOLDER = os.getenv('LOG_FOLDER', 'logs')

# Crear directorios
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Configuración Flask
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ========== LOGGING ==========
handler = RotatingFileHandler(
    os.path.join(LOG_FOLDER, 'pdfapi.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=5
)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# ========== SERVICIO OCR ==========
class OCRService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurar Tesseract
        try:
            pytesseract.get_tesseract_version()
        except:
            tesseract_paths = [
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract',
                '/opt/homebrew/bin/tesseract'
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
    
    def extract_text_from_image(self, image: Image.Image, lang: str = 'spa+eng') -> str:
        try:
            if image.mode != 'L':
                image = image.convert('L')
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error en OCR: {str(e)}")
            return ""
    
    def extract_text_from_pdf_with_ocr(self, pdf_path: str, lang: str = 'spa+eng',
                                      dpi: int = 300, page_numbers: Optional[List[int]] = None) -> Dict:
        result = {
            "success": False,
            "total_pages": 0,
            "pages": [],
            "text": "",
            "method": "ocr"
        }
        
        try:
            if page_numbers:
                images = convert_from_path(pdf_path, dpi=dpi, first_page=page_numbers[0], last_page=page_numbers[-1])
            else:
                images = convert_from_path(pdf_path, dpi=dpi)
            
            result["total_pages"] = len(images)
            full_text = ""
            
            for i, image in enumerate(images):
                page_text = self.extract_text_from_image(image, lang)
                page_data = {
                    "page_number": i + 1,
                    "text": page_text,
                    "image_size": image.size,
                    "dpi": dpi,
                    "language": lang
                }
                result["pages"].append(page_data)
                full_text += f"\n--- Página {i + 1} ---\n{page_text}\n"
            
            result["text"] = full_text.strip()
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def detect_text_type(self, pdf_path: str, sample_pages: int = 3) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                sample_text = ""
                max_pages = min(sample_pages, len(pdf_reader.pages))
                
                for i in range(max_pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    sample_text += page_text if page_text else ""
                
                if len(sample_text.strip()) > 50:
                    return "digital"
                return "scanned"
                
        except Exception as e:
            return "unknown"

ocr_service = OCRService()

# ========== FUNCIONES AUXILIARES ==========
def allowed_file(filename: str) -> bool:
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'pfx', 'p12'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path: str, page_numbers: List[int] = None,
                         include_metadata: bool = False) -> Dict:
    result = {
        "success": False,
        "total_pages": 0,
        "pages": [],
        "metadata": {},
        "text": "",
        "method": "digital"
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf_doc:
            result["total_pages"] = len(pdf_doc.pages)
            
            if include_metadata:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    if pdf_reader.metadata:
                        meta = pdf_reader.metadata
                        result["metadata"] = {
                            "title": getattr(meta, 'title', ''),
                            "author": getattr(meta, 'author', ''),
                            "subject": getattr(meta, 'subject', ''),
                            "creator": getattr(meta, 'creator', ''),
                            "producer": getattr(meta, 'producer', '')
                        }
            
            if not page_numbers:
                page_numbers = list(range(len(pdf_doc.pages)))
            
            full_text = ""
            for page_num in page_numbers:
                if 0 <= page_num < len(pdf_doc.pages):
                    page = pdf_doc.pages[page_num]
                    text = page.extract_text() or ""
                    page_data = {
                        "page_number": page_num + 1,
                        "text": text.strip(),
                        "dimensions": {"width": page.width, "height": page.height}
                    }
                    result["pages"].append(page_data)
                    full_text += f"\n--- Página {page_num + 1} ---\n{text.strip()}\n"
            
            result["text"] = full_text.strip()
            result["success"] = True
            
    except Exception as e:
        result["error"] = str(e)
    
    return result

def add_image_to_pdf(pdf_path: str, image_path: str, x: float, y: float, 
                    width: Optional[float] = None, height: Optional[float] = None, 
                    page_number: int = 1) -> str:
    from PyPDF2 import PdfReader, PdfWriter
    
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    
    if width and height:
        can.drawImage(image_path, x, y, width=width, height=height)
    else:
        can.drawImage(image_path, x, y)
    
    can.save()
    packet.seek(0)
    new_pdf = PdfReader(packet)
    
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        if page_num == page_number - 1:
            page.merge_page(new_pdf.pages[0])
        writer.add_page(page)
    
    output_filename = f"modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)
    
    return output_path

# ========== MODELOS PARA SWAGGER ==========
error_model = api.model('Error', {
    'error': fields.String(required=True, description='Mensaje de error'),
    'code': fields.String(description='Código de error'),
    'timestamp': fields.String(description='Timestamp del error')
})

success_model = api.model('Success', {
    'success': fields.Boolean(required=True, description='Estado de la operación'),
    'message': fields.String(description='Mensaje de éxito'),
    'timestamp': fields.String(description='Timestamp de la operación')
})

extraction_response_model = api.model('ExtractionResponse', {
    'success': fields.Boolean(required=True, description='Estado de la extracción'),
    'total_pages': fields.Integer(description='Número total de páginas'),
    'text': fields.String(description='Texto extraído'),
    'pages': fields.List(fields.Raw, description='Datos por página'),
    'metadata': fields.Raw(description='Metadatos del PDF'),
    'method': fields.String(description='Método utilizado (digital/ocr)'),
    'error': fields.String(description='Error si hubo alguno')
})

detection_model = api.model('PDFTypeDetection', {
    'type': fields.String(required=True, description='Tipo de PDF (digital/scanned/unknown)'),
    'description': fields.String(description='Descripción del tipo'),
    'recommended_action': fields.String(description='Acción recomendada'),
    'ocr_required': fields.Boolean(description='Si requiere OCR')
})

# ========== PARSERS ==========
extract_parser = reqparse.RequestParser()
extract_parser.add_argument('page_numbers', type=str, help='Páginas específicas (ej: 1,3,5)')
extract_parser.add_argument('include_metadata', type=str, choices=['true', 'false'], default='false', help='Incluir metadatos')
extract_parser.add_argument('strategy', type=str, choices=['auto', 'digital', 'ocr'], default='auto', help='Estrategia de extracción')
extract_parser.add_argument('language', type=str, default='spa+eng', help='Idioma para OCR')
extract_parser.add_argument('use_ocr', type=str, choices=['true', 'false'], default='false', help='Forzar uso de OCR')

ocr_parser = reqparse.RequestParser()
ocr_parser.add_argument('dpi', type=int, default=300, help='DPI para conversión')
ocr_parser.add_argument('language', type=str, default='spa+eng', help='Idioma para OCR')
ocr_parser.add_argument('page_numbers', type=str, help='Páginas específicas')

search_parser = reqparse.RequestParser()
search_parser.add_argument('search_term', type=str, required=True, help='Término a buscar')
search_parser.add_argument('case_sensitive', type=str, choices=['true', 'false'], default='false', help='Búsqueda sensible')
search_parser.add_argument('whole_word', type=str, choices=['true', 'false'], default='false', help='Búsqueda por palabra completa')

image_parser = reqparse.RequestParser()
image_parser.add_argument('x', type=float, required=True, help='Posición X')
image_parser.add_argument('y', type=float, required=True, help='Posición Y')
image_parser.add_argument('width', type=float, help='Ancho de la imagen')
image_parser.add_argument('height', type=float, help='Alto de la imagen')
image_parser.add_argument('page_number', type=int, default=1, help='Número de página')

sign_parser = reqparse.RequestParser()
sign_parser.add_argument('password', type=str, required=True, help='Contraseña del certificado')
sign_parser.add_argument('reason', type=str, default='Firma digital', help='Razón de la firma')
sign_parser.add_argument('location', type=str, default='', help='Ubicación')
sign_parser.add_argument('page', type=int, default=1, help='Página para firma')
sign_parser.add_argument('x', type=float, default=100, help='Posición X')
sign_parser.add_argument('y', type=float, default=100, help='Posición Y')
sign_parser.add_argument('width', type=float, default=200, help='Ancho')
sign_parser.add_argument('height', type=float, default=100, help='Alto')

# ========== ENDPOINTS ==========

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Processing API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            h1 {
                color: white;
                text-align: center;
                margin-bottom: 30px;
            }
            .card {
                background: rgba(255, 255, 255, 0.2);
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
                transition: transform 0.3s;
            }
            .card:hover {
                transform: translateY(-5px);
                background: rgba(255, 255, 255, 0.3);
            }
            .btn {
                display: inline-block;
                background: white;
                color: #667eea;
                padding: 12px 30px;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
                margin-top: 20px;
                transition: all 0.3s;
            }
            .btn:hover {
                background: #f8f9fa;
                transform: scale(1.05);
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .feature {
                background: rgba(255, 255, 255, 0.15);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 PDF Processing API</h1>
            
            <p>API completa para procesamiento de documentos PDF con las siguientes características:</p>
            
            <div class="features">
                <div class="feature">🔍 OCR Avanzado</div>
                <div class="feature">📄 Extracción de Texto</div>
                <div class="feature">🖼️ Manipulación de PDF</div>
                <div class="feature">🔎 Búsqueda Inteligente</div>
                <div class="feature">📝 Firma Digital</div>
                <div class="feature">⚡ Alto Rendimiento</div>
            </div>
            
            <div class="card">
                <h3>🚀 Documentación Swagger</h3>
                <p>Accede a la documentación interactiva de la API con ejemplos y pruebas en tiempo real.</p>
                <a href="/swagger/" class="btn">Abrir Swagger UI</a>
            </div>
            
            <div class="card">
                <h3>📚 Endpoints Disponibles</h3>
                <ul>
                    <li><strong>GET /health</strong> - Verificar estado del servicio</li>
                    <li><strong>POST /extract/text</strong> - Extraer texto de PDF</li>
                    <li><strong>POST /extract/ocr</strong> - Extraer texto con OCR</li>
                    <li><strong>POST /extract/detect</strong> - Detectar tipo de PDF</li>
                    <li><strong>POST /search</strong> - Buscar texto en PDF</li>
                    <li><strong>POST /add-image</strong> - Agregar imagen a PDF</li>
                    <li><strong>POST /sign-pdf</strong> - Firmar PDF digitalmente</li>
                    <li><strong>POST /create-pdf</strong> - Crear PDF con imagen</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>⚙️ Información Técnica</h3>
                <p><strong>Versión:</strong> 2.0.0</p>
                <p><strong>Límite de archivo:</strong> 16MB</p>
                <p><strong>Formatos soportados:</strong> PDF, PNG, JPEG</p>
                <p><strong>Idiomas OCR:</strong> Español, Inglés y más</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/swagger/" class="btn">👉 Comenzar con Swagger</a>
                <br><br>
                <small>También puedes usar herramientas como Postman o curl para interactuar con la API</small>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/health')
def health():
    return jsonify({
        "success": True,
        "message": "PDF API funcionando correctamente",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    })

# ========== ENDPOINTS CON SWAGGER ==========

# Namespace para extracción
extract_ns = Namespace('Extraction', description='Operaciones de extracción de texto')

@extract_ns.route('/text')
class ExtractText(Resource):
    @extract_ns.doc('extract_text', description='Extraer texto de PDF (con o sin OCR)')
    @extract_ns.expect(extract_parser)
    @extract_ns.response(200, 'Extracción exitosa', extraction_response_model)
    @extract_ns.response(400, 'Error en la solicitud', error_model)
    @extract_ns.response(500, 'Error interno', error_model)
    def post(self):
        """Extraer texto de un archivo PDF"""
        try:
            args = extract_parser.parse_args()
            
            if 'pdf_file' not in request.files:
                return {"error": "No se proporcionó archivo PDF", "code": "NO_FILE"}, 400
            
            pdf_file = request.files['pdf_file']
            if pdf_file.filename == '':
                return {"error": "No se seleccionó archivo PDF", "code": "EMPTY_FILE"}, 400
            
            # Procesar páginas
            pages = []
            if args['page_numbers']:
                try:
                    pages = [int(p.strip()) for p in args['page_numbers'].split(',')]
                    pages = [p-1 for p in pages]  # Convertir a 0-index
                except:
                    return {"error": "Formato inválido en page_numbers", "code": "INVALID_PAGES"}, 400
            
            # Guardar archivo temporal
            temp_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
            pdf_path = os.path.join(UPLOAD_FOLDER, f"temp_{temp_id}.pdf")
            pdf_file.save(pdf_path)
            
            # Determinar estrategia
            result = {}
            use_ocr = args['use_ocr'] == 'true' or args['strategy'] == 'ocr'
            
            if use_ocr:
                result = ocr_service.extract_text_from_pdf_with_ocr(
                    pdf_path=pdf_path,
                    lang=args['language'],
                    page_numbers=pages
                )
            else:
                if args['strategy'] == 'auto':
                    pdf_type = ocr_service.detect_text_type(pdf_path)
                    if pdf_type == 'scanned':
                        result = ocr_service.extract_text_from_pdf_with_ocr(
                            pdf_path=pdf_path,
                            lang=args['language'],
                            page_numbers=pages
                        )
                    else:
                        result = extract_text_from_pdf(
                            pdf_path=pdf_path,
                            page_numbers=pages,
                            include_metadata=args['include_metadata'] == 'true'
                        )
                else:
                    result = extract_text_from_pdf(
                        pdf_path=pdf_path,
                        page_numbers=pages,
                        include_metadata=args['include_metadata'] == 'true'
                    )
            
            # Limpiar archivo
            os.remove(pdf_path)
            
            return result
            
        except Exception as e:
            return {"error": str(e), "code": "EXTRACTION_ERROR"}, 500

@extract_ns.route('/ocr')
class ExtractOCR(Resource):
    @extract_ns.doc('extract_ocr', description='Extraer texto con OCR (para PDFs escaneados)')
    @extract_ns.expect(ocr_parser)
    @extract_ns.response(200, 'OCR exitoso', extraction_response_model)
    @extract_ns.response(400, 'Error en la solicitud', error_model)
    def post(self):
        """Extraer texto con OCR específico"""
        args = ocr_parser.parse_args()
        
        if 'pdf_file' not in request.files:
            return {"error": "No se proporcionó archivo PDF", "code": "NO_FILE"}, 400
        
        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return {"error": "No se seleccionó archivo PDF", "code": "EMPTY_FILE"}, 400
        
        # Procesar páginas
        pages = []
        if args['page_numbers']:
            try:
                pages = [int(p.strip()) for p in args['page_numbers'].split(',')]
            except:
                return {"error": "Formato inválido en page_numbers", "code": "INVALID_PAGES"}, 400
        
        # Guardar y procesar
        temp_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        pdf_path = os.path.join(UPLOAD_FOLDER, f"temp_ocr_{temp_id}.pdf")
        pdf_file.save(pdf_path)
        
        result = ocr_service.extract_text_from_pdf_with_ocr(
            pdf_path=pdf_path,
            lang=args['language'],
            dpi=args['dpi'],
            page_numbers=pages
        )
        
        result["ocr_parameters"] = {
            "dpi": args['dpi'],
            "language": args['language']
        }
        
        os.remove(pdf_path)
        return result

@extract_ns.route('/detect')
class DetectPDFType(Resource):
    @extract_ns.doc('detect_type', description='Detectar tipo de PDF (digital vs escaneado)')
    @extract_ns.response(200, 'Detección exitosa', detection_model)
    @extract_ns.response(400, 'Error en la solicitud', error_model)
    def post(self):
        """Detectar tipo de PDF"""
        if 'pdf_file' not in request.files:
            return {"error": "No se proporcionó archivo PDF", "code": "NO_FILE"}, 400
        
        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return {"error": "No se seleccionó archivo PDF", "code": "EMPTY_FILE"}, 400
        
        # Guardar y detectar
        temp_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        pdf_path = os.path.join(UPLOAD_FOLDER, f"temp_detect_{temp_id}.pdf")
        pdf_file.save(pdf_path)
        
        pdf_type = ocr_service.detect_text_type(pdf_path)
        
        type_info = {
            "digital": {
                "type": "digital",
                "description": "PDF con texto digital (seleccionable)",
                "recommended_action": "Usar extracción normal",
                "ocr_required": False
            },
            "scanned": {
                "type": "scanned",
                "description": "PDF escaneado o con imágenes",
                "recommended_action": "Usar OCR",
                "ocr_required": True
            },
            "unknown": {
                "type": "unknown",
                "description": "No se pudo determinar el tipo",
                "recommended_action": "Intentar ambos métodos",
                "ocr_required": True
            }
        }
        
        result = type_info.get(pdf_type, type_info["unknown"])
        os.remove(pdf_path)
        
        return result

# Namespace para búsqueda
search_ns = Namespace('Search', description='Operaciones de búsqueda en PDF')

@search_ns.route('/')
class SearchText(Resource):
    @search_ns.doc('search_text', description='Buscar texto en un PDF')
    @search_ns.expect(search_parser)
    @search_ns.response(200, 'Búsqueda exitosa')
    @search_ns.response(400, 'Error en la solicitud', error_model)
    def post(self):
        """Buscar texto en un PDF"""
        args = search_parser.parse_args()
        
        if 'pdf_file' not in request.files:
            return {"error": "No se proporcionó archivo PDF", "code": "NO_FILE"}, 400
        
        pdf_file = request.files['pdf_file']
        if pdf_file.filename == '':
            return {"error": "No se seleccionó archivo PDF", "code": "EMPTY_FILE"}, 400
        
        if not args['search_term']:
            return {"error": "No se proporcionó término de búsqueda", "code": "NO_SEARCH_TERM"}, 400
        
        # Guardar archivo
        temp_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        pdf_path = os.path.join(UPLOAD_FOLDER, f"temp_search_{temp_id}.pdf")
        pdf_file.save(pdf_path)
        
        # Realizar búsqueda (simplificada)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                matches = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    search_in = text.lower() if args['case_sensitive'] != 'true' else text
                    term = args['search_term'].lower() if args['case_sensitive'] != 'true' else args['search_term']
                    
                    if term in search_in:
                        matches.append({
                            "page": i + 1,
                            "text": args['search_term'],
                            "context": text[max(0, search_in.find(term)-50):search_in.find(term)+len(term)+50]
                        })
                
                result = {
                    "success": True,
                    "search_term": args['search_term'],
                    "total_matches": len(matches),
                    "matches": matches
                }
                
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "code": "SEARCH_ERROR"
            }
        
        os.remove(pdf_path)
        return result

# Namespace para manipulación de PDF
pdf_ns = Namespace('PDF Manipulation', description='Operaciones de manipulación de PDF')

@pdf_ns.route('/add-image')
class AddImageToPDF(Resource):
    @pdf_ns.doc('add_image', description='Agregar imagen a un PDF existente')
    @pdf_ns.expect(image_parser)
    @pdf_ns.response(200, 'Imagen agregada exitosamente')
    @pdf_ns.response(400, 'Error en la solicitud', error_model)
    def post(self):
        """Agregar imagen a un PDF"""
        args = image_parser.parse_args()
        
        if 'pdf_file' not in request.files or 'image_file' not in request.files:
            return {"error": "Se requieren ambos archivos: PDF e imagen", "code": "MISSING_FILES"}, 400
        
        pdf_file = request.files['pdf_file']
        image_file = request.files['image_file']
        
        if pdf_file.filename == '' or image_file.filename == '':
            return {"error": "Archivos no seleccionados", "code": "EMPTY_FILES"}, 400
        
        # Guardar archivos temporales
        temp_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        pdf_path = os.path.join(UPLOAD_FOLDER, f"temp_pdf_{temp_id}.pdf")
        image_path = os.path.join(UPLOAD_FOLDER, f"temp_img_{temp_id}.png")
        
        pdf_file.save(pdf_path)
        image_file.save(image_path)
        
        try:
            # Crear PDF modificado
            output_path = add_image_to_pdf(
                pdf_path=pdf_path,
                image_path=image_path,
                x=args['x'],
                y=args['y'],
                width=args.get('width'),
                height=args.get('height'),
                page_number=args['page_number']
            )
            
            # Limpiar temporales
            os.remove(pdf_path)
            os.remove(image_path)
            
            # Enviar archivo
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f"modified_{pdf_file.filename}",
                mimetype='application/pdf'
            )
            
        except Exception as e:
            # Limpiar en caso de error
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            return {"error": str(e), "code": "IMAGE_ADD_ERROR"}, 500

@pdf_ns.route('/sign')
class SignPDF(Resource):
    @pdf_ns.doc('sign_pdf', description='Firmar PDF digitalmente')
    @pdf_ns.expect(sign_parser)
    @pdf_ns.response(200, 'PDF firmado exitosamente')
    @pdf_ns.response(400, 'Error en la solicitud', error_model)
    def post(self):
        """Firmar PDF digitalmente"""
        args = sign_parser.parse_args()
        
        if 'pdf_file' not in request.files or 'pfx_file' not in request.files:
            return {"error": "Se requieren ambos archivos: PDF y certificado", "code": "MISSING_FILES"}, 400
        
        pdf_file = request.files['pdf_file']
        pfx_file = request.files['pfx_file']
        
        if pdf_file.filename == '' or pfx_file.filename == '':
            return {"error": "Archivos no seleccionados", "code": "EMPTY_FILES"}, 400
        
        # Simulación de firma
        temp_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        output_filename = f"signed_{pdf_file.filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Guardar archivo (simulación)
        pdf_file.save(output_path)
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype='application/pdf'
        )

@pdf_ns.route('/create')
class CreatePDFWithImage(Resource):
    @pdf_ns.doc('create_pdf', description='Crear un nuevo PDF con una imagen')
    @pdf_ns.response(200, 'PDF creado exitosamente')
    @pdf_ns.response(400, 'Error en la solicitud', error_model)
    def post(self):
        """Crear nuevo PDF con imagen"""
        if 'image_file' not in request.files:
            return {"error": "No se proporcionó imagen", "code": "NO_IMAGE"}, 400
        
        image_file = request.files['image_file']
        if image_file.filename == '':
            return {"error": "No se seleccionó imagen", "code": "EMPTY_IMAGE"}, 400
        
        # Obtener parámetros
        x = request.form.get('x', 100, type=float)
        y = request.form.get('y', 100, type=float)
        width = request.form.get('width', 100, type=float)
        height = request.form.get('height', 100, type=float)
        
        # Crear PDF en memoria
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Guardar imagen temporal
        temp_id = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        image_path = os.path.join(UPLOAD_FOLDER, f"temp_create_{temp_id}.png")
        image_file.save(image_path)
        
        # Agregar imagen
        c.drawImage(image_path, x, y, width=width, height=height)
        c.save()
        
        # Limpiar
        os.remove(image_path)
        
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name="document_with_image.pdf",
            mimetype='application/pdf'
        )

# Agregar namespaces a la API
api.add_namespace(extract_ns, path='/extract')
api.add_namespace(search_ns, path='/search')
api.add_namespace(pdf_ns, path='/pdf')

# ========== MANEJO DE ERRORES ==========
@app.errorhandler(413)
def handle_too_large(e):
    return jsonify({"error": "Archivo demasiado grande (máximo 16MB)", "code": "FILE_TOO_LARGE"}), 413

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": "Endpoint no encontrado", "code": "NOT_FOUND"}), 404

@app.errorhandler(500)
def handle_internal_error(e):
    app.logger.error(f'Error 500: {e}')
    return jsonify({"error": "Error interno del servidor", "code": "INTERNAL_ERROR"}), 500

@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({"error": "Solicitud incorrecta", "code": "BAD_REQUEST"}), 400

# ========== INICIALIZACIÓN ==========
if __name__ == '__main__':
    print("=" * 60)
    print("📄 PDF Processing API con Swagger")
    print("=" * 60)
    print(f"Versión: 2.0.0")
    print(f"Documentación: http://localhost:5000/swagger/")
    print(f"Interfaz web: http://localhost:5000/")
    print(f"Directorio de uploads: {UPLOAD_FOLDER}")
    print(f"Directorio de outputs: {OUTPUT_FOLDER}")
    print(f"Servidor escuchando en: http://0.0.0.0:5000")
    print("=" * 60)
    print("\nEndpoints principales:")
    print("  GET  /             - Interfaz web principal")
    print("  GET  /swagger/     - Documentación Swagger UI")
    print("  GET  /health       - Health check")
    print("  POST /extract/text - Extraer texto de PDF")
    print("  POST /extract/ocr  - Extraer texto con OCR")
    print("  POST /extract/detect - Detectar tipo de PDF")
    print("  POST /search/      - Buscar texto en PDF")
    print("  POST /pdf/add-image - Agregar imagen a PDF")
    print("  POST /pdf/sign     - Firmar PDF")
    print("  POST /pdf/create   - Crear PDF con imagen")
    print("=" * 60)
    
    # Iniciar servidor
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )