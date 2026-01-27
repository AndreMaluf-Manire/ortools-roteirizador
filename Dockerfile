# Dockerfile para OR-Tools Route Optimizer
# Deploy no Railway/Cloud Run

FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para OR-Tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY main.py .

# Expor porta (Railway usa a variável PORT)
EXPOSE 8000

# Comando para iniciar o servidor
CMD ["python", "main.py"]
