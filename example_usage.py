import requests
import json

class PDFAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def add_image_to_pdf(self, pdf_path, image_path, x, y, width=None, height=None, page_number=1):
        """
        Agrega una imagen a un PDF en coordenadas específicas
        """
        url = f"{self.base_url}/add-image"
        
        files = {
            'pdf_file': open(pdf_path, 'rb'),
            'image_file': open(image_path, 'rb')
        }
        
        data = {
            'x': x,
            'y': y,
            'page_number': page_number
        }
        
        if width:
            data['width'] = width
        if height:
            data['height'] = height
        
        response = requests.post(url, files=files, data=data)
        
        # Cerrar archivos
        files['pdf_file'].close()
        files['image_file'].close()
        
        if response.status_code == 200:
            with open('output_with_image.pdf', 'wb') as f:
                f.write(response.content)
            print("PDF con imagen creado exitosamente")
            return True
        else:
            print(f"Error: {response.json()}")
            return False
    
    def sign_pdf(self, pdf_path, pfx_path, password, reason="Firma digital", location=""):
        """
        Firma un PDF con certificado digital
        """
        url = f"{self.base_url}/sign-pdf"
        
        files = {
            'pdf_file': open(pdf_path, 'rb'),
            'pfx_file': open(pfx_path, 'rb')
        }
        
        data = {
            'password': password,
            'reason': reason,
            'location': location,
            'page': 1,
            'x': 100,
            'y': 100,
            'width': 200,
            'height': 100
        }
        
        response = requests.post(url, files=files, data=data)
        
        files['pdf_file'].close()
        files['pfx_file'].close()
        
        if response.status_code == 200:
            with open('signed_document.pdf', 'wb') as f:
                f.write(response.content)
            print("PDF firmado exitosamente")
            return True
        else:
            print(f"Error: {response.json()}")
            return False
    
    def extract_text(self, pdf_path, pages=None, include_metadata=True, extract_tables=True):
        """
        Extrae texto de un PDF
        """
        url = f"{self.base_url}/extract-text"
        
        files = {
            'pdf_file': open(pdf_path, 'rb')
        }
        
        data = {
            'include_metadata': str(include_metadata).lower(),
            'extract_tables': str(extract_tables).lower()
        }
        
        if pages:
            data['page_numbers'] = ','.join(map(str, pages))
        
        response = requests.post(url, files=files, data=data)
        files['pdf_file'].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"Extracción exitosa. {result['total_pages']} páginas procesadas.")
            return result
        else:
            print(f"Error: {response.json()}")
            return None
    
    def search_text(self, pdf_path, search_term, case_sensitive=False, whole_word=False):
        """
        Busca texto en un PDF
        """
        url = f"{self.base_url}/search-text"
        
        files = {
            'pdf_file': open(pdf_path, 'rb')
        }
        
        data = {
            'search_term': search_term,
            'case_sensitive': str(case_sensitive).lower(),
            'whole_word': str(whole_word).lower()
        }
        
        response = requests.post(url, files=files, data=data)
        files['pdf_file'].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"Búsqueda completada. {result['total_matches']} coincidencias encontradas.")
            return result
        else:
            print(f"Error: {response.json()}")
            return None

# Ejemplos de uso
if __name__ == "__main__":
    api = PDFAPI()
    
    # Ejemplo: Agregar imagen a PDF
    # api.add_image_to_pdf('documento.pdf', 'imagen.png', 100, 100, 50, 50)
    
    # Ejemplo: Firmar PDF
    # api.sign_pdf('documento.pdf', 'certificado.pfx', 'mi_password')
    
    # Ejemplo: Extraer texto
    # result = api.extract_text('documento.pdf')
    # if result:
    #     print(f"Texto extraído: {result['text'][:500]}...")
    
    # Ejemplo: Buscar texto
    # result = api.search_text('documento.pdf', 'contrato')
    # if result:
    #     for match in result['matches'][:3]:
    #         print(f"Página {match['page']}: {match['context']}")