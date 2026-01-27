# OR-Tools Route Optimizer

API de otimização de rotas usando Google OR-Tools para o Roteirizador Manirê/Fruleve.

## O que faz

Resolve o problema **CVRPTW** (Capacitated Vehicle Routing Problem with Time Windows):
- Minimiza número de veículos usados
- Respeita capacidade dos veículos (caixas/kg)
- Respeita janelas de atendimento dos clientes
- Garante 100% das entregas alocadas (sem órfãs)

## Endpoints

### `GET /`
Health check básico.

### `GET /health`
Health check detalhado.

### `POST /optimize`
Otimiza rotas. Recebe:
- `depot`: Base de origem
- `customers`: Lista de clientes com janelas
- `deliveries`: Entregas a serem alocadas
- `vehicles`: Veículos disponíveis
- `time_matrix`: Matriz de tempos (do Google Distance Matrix)
- `mode`: `"minimize_vehicles"` ou `"fixed_fleet"`

### `POST /validate`
Valida os dados antes de otimizar (útil para debug).

## Deploy no Railway

1. Conecte este repositório no Railway
2. Configure a variável de ambiente `ORTOOLS_API_KEY`
3. Deploy automático

## Variáveis de Ambiente

| Variável | Descrição |
|----------|-----------|
| `ORTOOLS_API_KEY` | Chave de API para autenticação |
| `PORT` | Porta do servidor (Railway configura automaticamente) |

## Desenvolvimento Local

```bash
pip install -r requirements.txt
python main.py
```

Acesse: http://localhost:8000/docs (Swagger UI)
