#!/bin/bash
# Script de instalación completa para PDF API en Ubuntu 20.04

set -e

echo "📦 Instalando PDF API en producción..."

# 1. Actualizar sistema
echo "🔄 Actualizando sistema..."
sudo apt update && sudo apt upgrade -y

# 2. Instalar dependencias
echo "📦 Instalando dependencias..."
sudo apt install -y python3 python3-pip python3-venv python3-dev
sudo apt install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng
sudo apt install -y poppler-utils libgl1-mesa-glx
sudo apt install -y nginx supervisor git curl wget

# 3. Crear usuario
echo "👤 Creando usuario..."
sudo useradd -m -s /bin/bash pdfapi || true

# 4. Configurar proyecto
echo "📁 Configurando proyecto..."
sudo -u pdfapi mkdir -p /home/pdfapi/apiPDF
cd /home/pdfapi/apiPDF

# 5. Crear entorno virtual
echo "🐍 Creando entorno virtual..."
sudo -u pdfapi python3 -m venv venv

# 6. Copiar archivos (debes tenerlos en el mismo directorio que este script)
echo "📄 Copiando archivos..."
# Aquí copiarías tus archivos o clonarías desde Git
sudo -u pdfapi git clone https://github.com/tu-usuario/apiPDF.git .

# 7. Instalar dependencias Python
echo "📦 Instalando dependencias Python..."
sudo -u pdfapi ./venv/bin/pip install --upgrade pip
sudo -u pdfapi ./venv/bin/pip install gunicorn
# sudo -u pdfapi ./venv/bin/pip install -r requirements.txt

# 8. Crear directorios
echo "📂 Creando directorios..."
sudo -u pdfapi mkdir -p uploads output logs static
chmod 755 uploads output logs

# 9. Configurar Supervisor
echo "👮 Configurando Supervisor..."
sudo tee /etc/supervisor/conf.d/pdfapi.conf > /dev/null << 'EOF'
[program:pdfapi]
directory=/home/pdfapi/apiPDF
command=/home/pdfapi/apiPDF/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:8000 --timeout 120 app:app
user=pdfapi
autostart=true
autorestart=true
environment=PYTHONPATH="/home/pdfapi/apiPDF",PATH="/home/pdfapi/apiPDF/venv/bin"
stderr_logfile=/var/log/pdfapi/error.log
stdout_logfile=/var/log/pdfapi/access.log

[group:pdfapi]
programs=pdfapi
EOF

sudo mkdir -p /var/log/pdfapi
sudo chown pdfapi:pdfapi /var/log/pdfapi

# 10. Configurar Nginx
echo "🌐 Configurando Nginx..."
sudo tee /etc/nginx/sites-available/pdfapi > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;
    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/pdfapi /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# 11. Iniciar servicios
echo "🚀 Iniciando servicios..."
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start pdfapi

sudo nginx -t
sudo systemctl restart nginx

# 12. Configurar firewall
echo "🛡️ Configurando firewall..."
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

echo "✅ Instalación completada!"
echo ""
echo "📊 URLs de acceso:"
echo "   - API: http://$(curl -s ifconfig.me)/"
echo "   - Health: http://$(curl -s ifconfig.me)/health"
echo "   - Swagger: http://$(curl -s ifconfig.me)/swagger/"
echo ""
echo "🔧 Comandos útiles:"
echo "   sudo supervisorctl status pdfapi"
echo "   sudo tail -f /var/log/pdfapi/access.log"
echo "   sudo systemctl reload nginx"